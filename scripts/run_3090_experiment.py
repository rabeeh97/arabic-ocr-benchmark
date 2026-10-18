"""Self-contained RTX-3090 fine-tuning runner (Windows-first, also POSIX).

One command takes a fresh box from nothing to a scored fine-tune:

    python scripts/run_3090_experiment.py --dir C:\\Users\\User\\a-experiment\\exp --base qwen3vl4b --detach

Everything lives under ``--dir``: the uv venv, the HF cache, the uv cache, the
dataset, checkpoints, merged weights, eval scores and plots. Deleting that one
folder removes every trace (no writes anywhere else). The run is:

  * **resumable**    — re-run the same command and it continues from the last
                       finished phase; training itself resumes from the newest
                       ``checkpoint-*``.
  * **detachable**   — ``--detach`` re-spawns the process free of the SSH
                       session (Windows: WMI ``Win32_Process.Create``, which
                       leaves OpenSSH's kill-on-close Job; POSIX:
                       ``start_new_session``) and returns immediately.
  * **observable**   — ``status`` prints the phase table, the newest checkpoint
                       and the tail of the log; everything streams to
                       ``<dir>/logs/run.log``.

Phases (each idempotent, tracked in ``<dir>/state.json``):

    bootstrap -> fetch -> prepare -> train -> merge -> eval -> report

See docs/experiments/3090-experiment-runbook.md for the SSH transfer / retrieval
/ cleanup instructions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

UTC = timezone.utc  # datetime.UTC alias; not present before Python 3.11
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

PHASES = ["bootstrap", "fetch", "prepare", "train", "merge", "eval", "report"]

# The base-model registry is shared with run_3090_pipeline.py,
# run_3090_benchmark.py and run_experiment_suite.py — see
# _experiment_bases.py's own docstring for why it's plain Python, not YAML.
from _detach_spawn import (  # noqa: E402
    emit_log_line, pid_alive, pid_is_cmd, pidfile_is_other_worker,
    spawn_surviving_logout, stdout_is_regular_file,
)
from _envfile import load_experiment_env  # noqa: E402
from _experiment_bases import BASES, is_local_checkpoint  # noqa: E402

# CUDA wheel index for torch on the 3090 box (cu124 covers Ada/Ampere on
# driver >= 550). Override with --torch-index if the box has an older driver.
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
UV_INSTALL_SH = "https://astral.sh/uv/install.sh"
UV_INSTALL_PS1 = "https://astral.sh/uv/install.ps1"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _download(url: str, dest) -> None:
    """Fetch ``url`` to ``dest`` with a browser-ish UA.

    ``urllib``'s default ``Python-urllib/x.y`` agent gets a blanket **403** from
    GitHub's asset CDN (``objects.githubusercontent.com``), which is where the
    ``astral.sh`` installer URLs redirect to.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def _log(dirs: dict, msg: str) -> None:
    emit_log_line(dirs["logs"] / "run.log", f"[{_now()}] {msg}")


def _rotate_log(dirs: dict) -> None:
    """Move a non-empty ``run.log`` aside so each launch starts on a clean log.

    Old sessions are kept as ``run.<UTC-timestamp>.log`` next to it; `status`
    always tails the current ``run.log`` only.
    """
    log = dirs["logs"] / "run.log"
    if log.exists() and log.stat().st_size > 0:
        stamp = _now().replace(":", "").replace("-", "")
        log.rename(dirs["logs"] / f"run.{stamp}.log")


def _dirs(root: Path) -> dict:
    d = {
        "root": root,
        "venv": root / "venv",
        "hf_cache": root / "hf_cache",
        "uv_cache": root / "uv_cache",
        "bin": root / "bin",
        "data": root / "data",
        "runs": root / "runs",
        "merged": root / "merged",
        "eval": root / "eval",
        "plots": root / "plots",
        "logs": root / "logs",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _pin_env(dirs: dict) -> None:
    os.environ["HF_HOME"] = str(dirs["hf_cache"])
    os.environ["UV_CACHE_DIR"] = str(dirs["uv_cache"])
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    load_experiment_env(dirs["root"])


def _state_path(dirs: dict) -> Path:
    return dirs["root"] / "state.json"


def _read_state(dirs: dict) -> dict:
    p = _state_path(dirs)
    if not p.exists():
        return {"phases": {}, "created": _now()}
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"phases": {}, "created": _now()}
    # Reconcile phase state against disk. state.json phase records for a
    # per-experiment phase are only trusted when merged/<exp>/config.json proves
    # that experiment actually trained + merged here — earlier code tracked
    # phases in a flat bucket and a second --base could make its phases look
    # done, or a botched migration could file them under the wrong experiment.
    trained = {
        md.name for md in (dirs["merged"].iterdir() if dirs["merged"].exists() else [])
        if (md / "config.json").exists()
    }
    flat = st.pop("phases", {})
    boot = {"bootstrap": flat["bootstrap"]} if "bootstrap" in flat else {}
    # Migration shim for state.json files written before phase records
    # existed. Must prove the bootstrap actually FINISHED (import check),
    # not just that `uv venv` created an interpreter file -- see
    # _bootstrap_complete()'s docstring for why bare existence is a false
    # positive that's bitten repeatedly on this box.
    if not boot and _bootstrap_complete(dirs):
        boot = {"bootstrap": {"done": True, "at": "reconciled-from-disk"}}
    st["phases"] = boot
    exps = st.setdefault("experiments", {})
    # legacy flat phases -> the experiment that has merged weights, if any
    legacy_exp = st.get("experiment")
    if legacy_exp in trained:
        b = exps.setdefault(legacy_exp, {}).setdefault("phases", {})
        for k, v in flat.items():
            if k != "bootstrap":
                b.setdefault(k, v)
    # Reconcile each experiment's bucket against disk:
    #  - a merged/<exp>/config.json proves fetch->prepare->train->merge finished;
    #  - without it, merge/eval/report cannot legitimately be "done" (clear them);
    #  - fetch/prepare/train records are written by that experiment's own run and
    #    are trusted as-is (buckets are per-experiment, so no cross-base bleed).
    for name in set(list(exps) + list(trained)):
        ph = exps.setdefault(name, {}).setdefault("phases", {})
        if name in trained:
            for k in ("fetch", "prepare", "train", "merge"):
                ph.setdefault(k, {"done": True, "at": "reconciled-from-disk"})
        else:
            for k in ("merge", "eval", "report"):
                ph.pop(k, None)
    return st


def _write_state(dirs: dict, state: dict) -> None:
    _state_path(dirs).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _phases(state: dict, exp: str | None) -> dict:
    """Phase-record dict. ``exp=None`` -> the shared bucket (``bootstrap``, since
    the venv/deps are per-folder). ``exp`` set -> that experiment's own bucket,
    so running a second ``--base`` in the same ``--dir`` doesn't see the first
    one's ``train``/``merge``/``eval`` as already done."""
    if exp is None:
        return state.setdefault("phases", {})
    return state.setdefault("experiments", {}).setdefault(exp, {}).setdefault("phases", {})


def _mark(dirs: dict, state: dict, phase: str, *, exp: str | None = None, **extra) -> None:
    _phases(state, exp)[phase] = {"done": True, "at": _now(), **extra}
    _write_state(dirs, state)


def _done(state: dict, phase: str, exp: str | None = None) -> bool:
    return bool(_phases(state, exp).get(phase, {}).get("done"))


def _venv_python(dirs: dict) -> Path:
    return dirs["venv"] / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _bootstrap_complete(dirs: dict) -> bool:
    """True only if the venv's interpreter can actually import this package
    and its core deps -- proof the FULL bootstrap finished, not just that
    `uv venv` created an interpreter file. `uv venv` itself is fast and
    rarely fails; the slow steps after it (CUDA torch, the editable install,
    the transformers/trl pins) are what actually go wrong (WDAC blocks,
    missing build-time deps, a flaky download) -- and when one of them does,
    python.exe is still sitting on disk from the successful `uv venv` step,
    making a partial bootstrap indistinguishable from a finished one to a
    bare existence check. That false positive is exactly what silently
    skipped straight to `fetch` (missing `pandas`/`huggingface_hub`) on a
    supposedly-fresh re-run here more than once."""
    py = _venv_python(dirs)
    if not py.exists():
        return False
    try:
        r = subprocess.run(
            [str(py), "-c", "import arabic_ocr_bench, torch, pandas, transformers"],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except OSError:
        return False


def _uv_bin(dirs: dict) -> str:
    """uv on PATH, else the copy we install into <dir>/bin."""
    onpath = shutil.which("uv")
    if onpath:
        return onpath
    local = dirs["bin"] / ("uv.exe" if IS_WIN else "uv")
    return str(local)


HEARTBEAT_S = 90  # log a liveness line after this long with no subprocess output


def _run(dirs: dict, cmd: list[str], *, phase: str, env: dict | None = None) -> int:
    """Run a subprocess, streaming combined output to the log and stdout. A
    background reader thread + queue means a long silent stretch (a big
    download, a slow training step) gets an explicit heartbeat line instead of
    looking indistinguishable from a hang."""
    _log(dirs, f"$ {' '.join(str(c) for c in cmd)}")
    e = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}  # keep a crash traceback in run.log
    tee = not stdout_is_regular_file()  # tty or pipe -> echo; detached log file -> don't dup
    fh = (dirs["logs"] / "run.log").open("a", encoding="utf-8") if tee else None
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(REPO), env=e,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        q: queue.Queue = queue.Queue()

        def _reader() -> None:
            for ln in proc.stdout:
                q.put(ln)
            q.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        last = time.monotonic()
        while True:
            try:
                ln = q.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                _log(dirs, f"...  {phase}: still running (pid {proc.pid}, "
                           f"{int(time.monotonic() - last)}s since last output)")
                continue
            if ln is None:
                break
            last = time.monotonic()
            sys.stdout.write(ln)
            sys.stdout.flush()
            if fh is not None:
                fh.write(ln)
        proc.wait()
    finally:
        if fh is not None:
            fh.close()
    if proc.returncode != 0:
        _log(dirs, f"!! phase {phase}: command exited {proc.returncode}")
    return proc.returncode


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def phase_bootstrap(dirs: dict, state: dict, spec: dict, args) -> None:
    # "done" is only trusted if the venv can actually import this package and
    # its core deps -- not just that a python.exe file exists (it may have
    # been deleted, e.g. rebuilding from a different --venv-python; or it may
    # be a survivor of a PARTIAL bootstrap where `uv venv` succeeded but a
    # later pip-install step failed, which looks identical to a finished
    # bootstrap under a bare existence check). See _bootstrap_complete().
    if _done(state, "bootstrap") and not args.force and _bootstrap_complete(dirs):
        _log(dirs, "bootstrap: already done")
        return
    _log(dirs, "bootstrap: install uv / venv / deps")

    uv = _uv_bin(dirs)
    if not Path(uv).exists() and shutil.which("uv") is None:
        _log(dirs, f"bootstrap: fetching standalone uv -> {dirs['bin']}")
        env = {**os.environ, "UV_INSTALL_DIR": str(dirs["bin"]), "UV_UNMANAGED_INSTALL": str(dirs["bin"])}
        if IS_WIN:
            ps1 = dirs["bin"] / "install-uv.ps1"
            _download(UV_INSTALL_PS1, ps1)
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
                check=True, env=env,
            )
        else:
            sh = dirs["bin"] / "install-uv.sh"
            _download(UV_INSTALL_SH, sh)
            subprocess.run(["sh", str(sh)], check=True, env=env)
        uv = _uv_bin(dirs)

    py = _venv_python(dirs)
    # --venv-python: build the venv from a specific interpreter (e.g. a signed
    # python.org 3.11) instead of uv's managed build — needed on boxes where
    # Smart App Control / WDAC blocks the unrecognised uv CPython (WinError 4551).
    venv_py = getattr(args, "venv_python", None) or "3.11"
    if not py.exists() and _run(
        dirs, [uv, "venv", str(dirs["venv"]), "--python", venv_py], phase="bootstrap"
    ):
        raise SystemExit("bootstrap: `uv venv` failed")

    pipbase = [uv, "pip", "install", "--python", str(py)]
    # 1. CUDA torch first so the CUDA build wins over the CPU wheel a later
    #    resolve would pick.
    if _run(dirs, [*pipbase, "torch", "torchvision", "--index-url", args.torch_index], phase="bootstrap"):
        raise SystemExit("bootstrap: torch (CUDA) install failed - check --torch-index / driver")
    # 2. the benchmark + training extras (editable, from the SSH-copied repo).
    #    --no-build-isolation: by default `uv pip install -e` builds this
    #    package's PEP 517 backend (hatchling) in a throwaway venv of its own
    #    under <dir>\uv_cache\builds-v0\ -- a SEPARATE interpreter from the
    #    one `--venv-python` just pinned, so on a WDAC/Smart-App-Control box
    #    it hits the exact same "unrecognised python.exe blocked" (WinError
    #    4551) that --venv-python was supposed to have already solved, one
    #    step later. With isolation skipped, `uv` no longer auto-fetches the
    #    build backend's own requirements into a throwaway env, so they have
    #    to be installed into the main venv by hand first: hatchling itself,
    #    plus `editables` -- hatchling's build_editable() imports it lazily
    #    (hatchling/builders/wheel.py) only for an EDITABLE build specifically
    #    (a plain `pip install .` wouldn't need it), so it's easy to miss.
    if _run(dirs, [*pipbase, "hatchling", "editables"], phase="bootstrap"):
        raise SystemExit("bootstrap: `pip install hatchling editables` failed")
    if _run(
        dirs, [*pipbase, "--no-build-isolation", "-e", f"{REPO}[local,train]"], phase="bootstrap"
    ):
        raise SystemExit("bootstrap: `pip install -e .[local,train]` failed")
    # 3. pin the training stack to the era this code targets, then plotting.
    #    The pyproject floors (`transformers>=4.49`, `trl>=0.12`) are unbounded,
    #    so a fresh resolve drifts to transformers 5.x / trl 1.x — whose
    #    `SFTConfig` dropped `warmup_ratio` and rewrote the SFT data pipeline.
    #    Cap both below the next major; bump the transformers floor for bases
    #    that need a newer minor (Qwen3-VL wants >=4.57).
    #    `-P name` upgrades ONLY that package, not its deps — a bare `-U` would
    #    also pull a fresh PyPI `torch`, replacing the CUDA wheel from step 1
    #    (and on WDAC-locked boxes the new torch DLLs fail to load, WinError 4551).
    tfloor = spec["needs_transformers"] or "4.49"
    tmax = spec.get("transformers_max") or "5"
    if _run(dirs, [*pipbase, "-P", "transformers", "-P", "trl",
                   f"transformers>={tfloor},<{tmax}", "trl>=0.21,<1.0"], phase="bootstrap"):
        raise SystemExit("bootstrap: transformers/trl pin install failed")
    _run(dirs, [*pipbase, "matplotlib>=3.7", "hf_transfer"], phase="bootstrap")
    # WDAC on this box blocks pandas 3.x indexing.pyd and pyarrow 25.x
    # _dataset.pyd. Re-pin after the editable install in case the resolver
    # still pulled a newer wheel.
    if _run(dirs, [*pipbase, "-P", "pandas", "-P", "pyarrow",
                   "pandas>=2.0,<3", "pyarrow>=15,<22"], phase="bootstrap"):
        raise SystemExit("bootstrap: pandas/pyarrow WDAC pin failed")

    # 4. verify CUDA + bitsandbytes + the WDAC-sensitive native wheels
    check = (
        "import torch,bitsandbytes,pandas,pyarrow;"
        "from pandas.core.frame import DataFrame;"
        "import pyarrow.dataset as pds;"
        "assert torch.cuda.is_available(), 'CUDA not available in the venv';"
        "print('cuda', torch.version.cuda, torch.cuda.get_device_name(0));"
        "print('bitsandbytes', bitsandbytes.__version__);"
        "print('pandas', pandas.__version__);"
        "print('pyarrow', pyarrow.__version__)"
    )
    if _run(dirs, [str(py), "-c", check], phase="bootstrap"):
        raise SystemExit("bootstrap: CUDA / bitsandbytes self-check failed (see log)")
    _mark(dirs, state, "bootstrap", uv=str(uv), python=str(py))


def phase_fetch(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "fetch", spec["experiment"]) and not args.force:
        _log(dirs, "fetch: already done")
        return
    py = _venv_python(dirs)
    hf_id = spec["hf_id"]
    _log(dirs, f"fetch: base weights + dataset + fonts ({hf_id})")

    if is_local_checkpoint(hf_id):
        # Stacked fine-tune: LoRA goes on already-merged local weights. Do not
        # snapshot_download a Windows path, and do not re-pull MBZUAI/AIN.
        _log(dirs, f"fetch: local checkpoint, skip Hub download ({hf_id})")
        probe = (
            "import json;from pathlib import Path;"
            f"d=json.load(open(Path({json.dumps(hf_id)})/'config.json',encoding='utf-8'));"
            "print('architectures:', d.get('architectures'));"
            "print('model_type:', d.get('model_type'))"
        )
        _run(dirs, [str(py), "-c", probe], phase="fetch")
    else:
        # base model — snapshot_download gives a clean progress bar and fails fast
        # on a gated repo without a valid token.
        dl = (
            "import os;from huggingface_hub import snapshot_download;"
            f"p=snapshot_download({hf_id!r}, token=os.environ.get('HF_TOKEN'));"
            "print('base ->', p)"
        )
        if _run(dirs, [str(py), "-c", dl], phase="fetch"):
            gated = " (gated - is HF_TOKEN in experiment.env and has it accepted the licence?)" if spec["gated"] else ""
            raise SystemExit(f"fetch: could not download {hf_id}{gated}")

        # probe the real architecture class so a wrong --base fails here, not after
        # a 20-minute train start.
        probe = (
            "import json,glob,os;from huggingface_hub import hf_hub_download;"
            f"c=hf_hub_download({hf_id!r}, 'config.json', token=os.environ.get('HF_TOKEN'));"
            "d=json.load(open(c));print('architectures:', d.get('architectures'));"
            "print('model_type:', d.get('model_type'))"
        )
        _run(dirs, [str(py), "-c", probe], phase="fetch")

    extra = spec.get("fetch_datasets")
    if extra:
        for repo, split in extra:
            ds = (
                "import os;from datasets import load_dataset;"
                f"d=load_dataset({repo!r}, split={split!r}, token=os.environ.get('HF_TOKEN'));"
                f"print({repo!r}, {split!r}, 'rows:', len(d))"
            )
            if _run(dirs, [str(py), "-c", ds], phase="fetch"):
                raise SystemExit(
                    f"fetch: {repo} {split} download failed (HF_TOKEN for the gated set?)"
                )
    else:
        # training dataset (Muharaf train split) — default for the original
        # exp3090 / suite_v1 jobs.
        ds = (
            "from datasets import load_dataset;"
            "d=load_dataset('aamijar/muharaf-public', split='train');"
            "print('muharaf train rows:', len(d))"
        )
        if _run(dirs, [str(py), "-c", ds], phase="fetch"):
            raise SystemExit("fetch: muharaf-public train split download failed (HF_TOKEN for the gated set?)")

    fonts = REPO / "scripts" / "fetch_fonts.py"
    if fonts.exists():
        _run(dirs, [str(py), str(fonts)], phase="fetch")
    _mark(dirs, state, "fetch", exp=spec["experiment"])


def _effective_config(dirs: dict, spec: dict, args) -> Path:
    """Base training config with output_dir (+ optional limit/epochs) overridden.

    The rewrite runs in the venv python (which has pyyaml) so this launcher
    stays stdlib-only and works on a bare box before bootstrap.
    """
    py = _venv_python(dirs)
    src = REPO / spec["config"]
    out = dirs["root"] / "effective_config.yaml"
    outdir = str(dirs["runs"] / spec["experiment"])
    code = (
        "import sys,yaml;"
        f"c=yaml.safe_load(open(r'{src}',encoding='utf-8').read());"
        f"c['output_dir']=r'{outdir}';"
        f"c['base_model']['hf_id']={json.dumps(spec['hf_id'])};"
        + (f"[s.__setitem__('limit',{args.limit}) for s in c['data']['sources']];" if args.limit is not None else "")
        + (f"c['train']['epochs']={args.epochs};" if args.epochs is not None else "")
        + f"open(r'{out}','w',encoding='utf-8').write(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))"
    )
    if _run(dirs, [str(py), "-c", code], phase="prepare"):
        raise SystemExit("prepare: could not build the effective config")
    return out


def phase_prepare(dirs: dict, state: dict, spec: dict, args) -> None:
    # Always rewrite effective_config.yaml from the source config, even when
    # prepare is cached as done: it's the file `train` reads on every run, and
    # a resumed experiment must pick up config edits (e.g. max_seq_len) without
    # needing --force, which would also redo the slow fetch/bootstrap phases.
    eff = _effective_config(dirs, spec, args)
    if _done(state, "prepare", spec["experiment"]) and not args.force:
        _log(dirs, "prepare: already done (effective_config.yaml refreshed from source config)")
        return
    py = _venv_python(dirs)
    _log(dirs, "prepare: assemble counts + contamination gate")
    cli = [str(py), "-m", "arabic_ocr_bench.training.cli"]
    _run(dirs, [*cli, "assemble", "--config", str(eff)], phase="prepare")
    report_path = dirs["root"] / "contamination_report.json"
    rc = _run(dirs, [*cli, "check-contam", "--config", str(eff),
                     "--out", str(report_path)], phase="prepare")
    if rc:
        # exit 1 == "flagged" *and* exit 1 == "uncaught exception" — disambiguate
        # by whether a report with n_flagged>0 was actually written.
        try:
            n_flagged = json.loads(report_path.read_text(encoding="utf-8")).get("n_flagged")
        except (OSError, ValueError):
            n_flagged = None
        if n_flagged:
            if not args.allow_contam:
                raise SystemExit(
                    "prepare: contamination check FLAGGED a train/eval overlap. Inspect "
                    f"{report_path} - pass --allow-contam only if it is a false positive."
                )
        else:
            raise SystemExit(
                f"prepare: check-contam exited {rc} without a flagged report - it "
                "crashed (see the traceback in the log above), this is not a "
                "contamination hit and --allow-contam will not help."
            )
    _mark(dirs, state, "prepare", exp=spec["experiment"], config=str(eff))


def phase_train(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "train", spec["experiment"]) and not args.force:
        _log(dirs, "train: already done")
        return
    py = _venv_python(dirs)
    eff = dirs["root"] / "effective_config.yaml"
    _log(dirs, "train: bench-train sft (resumes from newest checkpoint-* if present)")
    rc = _run(dirs, [str(py), "-m", "arabic_ocr_bench.training.cli", "sft",
                     "--config", str(eff)], phase="train")
    if rc:
        raise SystemExit("train: sft exited non-zero - fix the cause and re-run (it resumes)")
    _mark(dirs, state, "train", exp=spec["experiment"])


def phase_merge(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "merge", spec["experiment"]) and not args.force:
        _log(dirs, "merge: already done")
        return
    py = _venv_python(dirs)
    exp = spec["experiment"]
    adapter = dirs["runs"] / exp
    out = dirs["merged"] / exp
    _log(dirs, f"merge: {adapter} -> {out}")
    rc = _run(dirs, [str(py), "-m", "arabic_ocr_bench.training.cli", "merge",
                     "--adapter", str(adapter), "--out", str(out),
                     "--base", spec["hf_id"], "--arch", spec["arch"]], phase="merge")
    if rc:
        raise SystemExit("merge: failed")
    _mark(dirs, state, "merge", exp=spec["experiment"], merged=str(out))


def phase_eval(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "eval", spec["experiment"]) and not args.force:
        _log(dirs, "eval: already done")
        return
    py = _venv_python(dirs)
    exp = spec["experiment"]
    merged = dirs["merged"] / exp
    name = f"{exp}-ft"
    _log(dirs, f"eval: register {name} + score the held-out eval config")
    _run(dirs, [str(py), "-m", "arabic_ocr_bench.training.cli", "register",
                "--name", name, "--path", str(merged), "--arch", spec["arch"]], phase="eval")
    rc = _run(dirs, [str(py), "-m", "arabic_ocr_bench.cli", "run",
                     "--config", str(REPO / spec["run_config"]), "--report"], phase="eval")
    if rc:
        raise SystemExit("eval: `bench run` failed")
    # copy the newest scored summary into <dir>/eval/
    runs = sorted((REPO / "results").glob("*/summary.csv"), key=lambda p: p.stat().st_mtime)
    if runs:
        src = runs[-1].parent
        for f in ("summary.csv", "summary.md", "scores.jsonl", "predictions.jsonl"):
            if (src / f).exists():
                shutil.copy2(src / f, dirs["eval"] / f)
        _log(dirs, f"eval: copied {src.name}/ -> {dirs['eval']}")
    _mark(dirs, state, "eval", exp=spec["experiment"])


_REPORT_PY = r'''
"""Written by run_3090_experiment.py; renders SFT loss + eval-CER plots."""
import csv, glob, json, os, sys

RUN, PLOTS, EVAL = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # noqa: BLE001
    print("report: matplotlib unavailable, skipping plots:", e)
    sys.exit(0)

cands = glob.glob(os.path.join(RUN, "**", "trainer_state.json"), recursive=True)
direct = os.path.join(RUN, "trainer_state.json")
if os.path.exists(direct):
    cands.append(direct)
if cands:
    hist = json.load(open(max(cands, key=os.path.getmtime)))["log_history"]
    tr = [(x["step"], x["loss"]) for x in hist if "loss" in x]
    ev = [(x["step"], x["eval_loss"]) for x in hist if "eval_loss" in x]
    if tr:
        plt.figure(figsize=(8, 4))
        plt.plot(*zip(*tr), label="train")
        if ev:
            plt.plot(*zip(*ev), marker="o", label="eval")
        plt.xlabel("step"); plt.ylabel("loss"); plt.legend()
        plt.title("SFT loss"); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "loss.png"), dpi=120)
        print("wrote loss.png")

sc = os.path.join(EVAL, "summary.csv")
if os.path.exists(sc):
    rows = list(csv.DictReader(open(sc, encoding="utf-8")))
    lab = [f"{r.get('model','?')} / {r.get('dataset','?')}" for r in rows]
    val = [float(r.get("cer_micro_light") or "nan") for r in rows]
    plt.figure(figsize=(8, 3 + 0.3 * len(rows)))
    plt.barh(lab, val); plt.xlabel("CER-light (micro)")
    plt.title("held-out eval"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "eval_cer.png"), dpi=120)
    print("wrote eval_cer.png")
'''


def phase_report(dirs: dict, state: dict, spec: dict, args) -> None:
    py = _venv_python(dirs)
    exp = spec["experiment"]
    _log(dirs, "report: loss curve + eval CER -> plots/")
    rp = dirs["root"] / "_report.py"
    rp.write_text(_REPORT_PY, encoding="utf-8")
    _run(dirs, [str(py), str(rp), str(dirs["runs"] / exp), str(dirs["plots"]),
                str(dirs["eval"])], phase="report")
    # per-experiment report bundle: shared plots/ & eval/ get overwritten by the
    # next --base, so snapshot everything the leaderboard's Fine-tuning tab needs.
    rd = dirs["root"] / "reports" / exp
    rd.mkdir(parents=True, exist_ok=True)
    cands = list((dirs["runs"] / exp).glob("**/trainer_state.json"))
    if (dirs["runs"] / exp / "trainer_state.json").exists():
        cands.append(dirs["runs"] / exp / "trainer_state.json")
    if cands:
        shutil.copy2(max(cands, key=lambda p: p.stat().st_mtime), rd / "trainer_state.json")
    for src in (dirs["root"] / "effective_config.yaml",
                dirs["eval"] / "summary.csv", dirs["eval"] / "summary.md",
                dirs["plots"] / "loss.png", dirs["plots"] / "eval_cer.png"):
        if src.exists():
            shutil.copy2(src, rd / src.name)
    _log(dirs, f"report: snapshot -> {rd}")
    _mark(dirs, state, "report", exp=spec["experiment"])


PHASE_FN = {
    "bootstrap": phase_bootstrap, "fetch": phase_fetch, "prepare": phase_prepare,
    "train": phase_train, "merge": phase_merge, "eval": phase_eval, "report": phase_report,
}


# --------------------------------------------------------------------------- #
# detach + status + drive
# --------------------------------------------------------------------------- #
def _respawn_detached(dirs: dict, argv: list[str]) -> int:
    _rotate_log(dirs)
    child_argv = [sys.executable, str(Path(__file__).resolve()), *[a for a in argv if a != "--detach"]]
    pid = spawn_surviving_logout(
        child_argv,
        cwd=REPO,
        log_path=dirs["logs"] / "run.log",
        pid_path=dirs["logs"] / "run.pid",
    )
    print(f"detached pid {pid}")
    print(f"log : {dirs['logs'] / 'run.log'}")
    # Users always launch this as `python scripts\run_3090_experiment.py ...`
    # (or scripts/... on posix) from the repo root -- Path(__file__).name alone
    # prints a bare filename that only works if the cwd happens to already be
    # scripts/, which it normally isn't, producing a confusing "No such file".
    _script_rel = ("scripts\\" if IS_WIN else "scripts/") + Path(__file__).name
    print(f"poll: python {_script_rel} --dir {dirs['root']} status")
    return 0


def _pid_alive(pid: int) -> bool:
    return pid_alive(pid)


def _active_worker(dirs: dict) -> int | None:
    """PID in ``run.pid`` if it names a live process other than us, else None."""
    pidf = dirs["logs"] / "run.pid"
    if not pidf.exists():
        return None
    try:
        pid = int(pidf.read_text(encoding="utf-8").strip() or 0)
    except ValueError:
        return None
    if pidfile_is_other_worker(
        pid, os.getpid(), os.getppid(),
        alive=_pid_alive(pid), is_cmd=pid_is_cmd(pid),
    ):
        return pid
    return None


def cmd_status(dirs: dict) -> int:
    state = _read_state(dirs)
    spec_name = state.get("base", "?")
    print(f"experiment dir : {dirs['root']}")
    print(f"base           : {spec_name}")
    pidf = dirs["logs"] / "run.pid"
    if pidf.exists():
        pid = int(pidf.read_text().strip() or 0)
        print(f"worker pid     : {pid}  ({'alive' if _pid_alive(pid) else 'not running'})")
    exp = state.get("experiment")
    print(f"\nphases ({exp or '?'}):")
    exp_ph = _phases(state, exp) if exp else {}
    for ph in PHASES:
        rec = state["phases"].get(ph, {}) if ph == "bootstrap" else exp_ph.get(ph, {})
        mark = "done" if rec.get("done") else "--"
        print(f"  {ph:<10} {mark:<6} {rec.get('at', '')}")
    if state.get("experiments"):
        others = [e for e in state["experiments"] if e != exp]
        if others:
            print(f"  other experiments in this folder: {', '.join(others)}")
    if exp:
        cks = sorted((dirs["runs"] / exp).glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        if cks:
            print(f"\nnewest checkpoint: {cks[-1].name}")
    log = dirs["logs"] / "run.log"
    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        print("\n--- log tail (40) ---")
        print("\n".join(tail))
    return 0


def cmd_clean(root: Path, yes: bool) -> int:
    cmd = f'rmdir /s /q "{root}"' if IS_WIN else f'rm -rf "{root}"'
    if not yes:
        print("this would remove the entire experiment folder (venv, caches, weights, logs):")
        print(f"  {cmd}")
        print("re-run with --yes to actually delete.")
        return 0
    shutil.rmtree(root, ignore_errors=True)
    print(f"removed {root}")
    return 0


def drive(dirs: dict, spec: dict, args, argv: list[str]) -> int:
    state = _read_state(dirs)
    state["base"] = args.base
    state["experiment"] = spec["experiment"]
    state["config"] = spec["config"]
    state.setdefault("started", _now())
    _write_state(dirs, state)

    active = _active_worker(dirs)
    if active and not args.force:
        print(
            f"a run is already active in this folder (pid {active}).\n"
            f"  watch it : python {Path(__file__).name} --dir {dirs['root']} status\n"
            f"  stop it  : {'taskkill /F /PID' if IS_WIN else 'kill'} {active}\n"
            f"  or pass --force to start anyway (only if you know it is dead)."
        )
        return 1

    if args.detach:
        return _respawn_detached(dirs, argv)

    if not stdout_is_regular_file():
        _rotate_log(dirs)
    (dirs["logs"] / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    start = args.from_phase or ("bootstrap" if not args.only else args.only)
    plan = [args.only] if args.only else PHASES[PHASES.index(start):]
    _log(dirs, f"=== run: base={args.base} phases={plan} ===")
    try:
        for ph in plan:
            try:
                PHASE_FN[ph](dirs, state, spec, args)
            except SystemExit as e:
                _log(dirs, f"STOP at phase {ph}: {e}")
                return 1
            state = _read_state(dirs)
        _log(dirs, "=== all phases complete ===")
        _log(dirs, f"merged weights : {dirs['merged']}")
        _log(dirs, f"eval scores    : {dirs['eval']}")
        _log(dirs, f"plots          : {dirs['plots']}")
        return 0
    finally:
        pidf = dirs["logs"] / "run.pid"
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="run_3090_experiment", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="self-contained experiment folder (created if absent)")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "status", "clean"],
                    help="run (default) | status | clean")
    ap.add_argument("--base", default="qwen3vl4b", choices=list(BASES),
                    help="base model (default: qwen3vl4b)")
    ap.add_argument("--config", default=None, help="override the training config path")
    ap.add_argument("--detach", action="store_true", help="re-spawn free of this shell and return")
    ap.add_argument("--from", dest="from_phase", default=None, choices=PHASES,
                    help="start at this phase")
    ap.add_argument("--only", default=None, choices=PHASES, help="run just this one phase")
    ap.add_argument("--limit", type=int, default=None, help="override training-line limit")
    ap.add_argument("--epochs", type=float, default=None, help="override epochs")
    ap.add_argument("--venv-python", dest="venv_python", default=None,
                    help="interpreter for `uv venv` (path or version); default 3.11. "
                         "Use a signed python.org 3.11 if Smart App Control blocks uv's build")
    ap.add_argument("--torch-index", default=DEFAULT_TORCH_INDEX,
                    help="pip index-url for the CUDA torch wheel")
    ap.add_argument("--allow-contam", action="store_true",
                    help="proceed even if the contamination check flags an overlap")
    ap.add_argument("--force", action="store_true", help="re-run phases already marked done")
    ap.add_argument("--yes", action="store_true", help="clean: actually delete")
    args = ap.parse_args(argv)

    root = Path(args.dir).expanduser().resolve()
    if args.cmd == "clean":
        return cmd_clean(root, args.yes)

    dirs = _dirs(root)
    _pin_env(dirs)
    if args.cmd == "status":
        return cmd_status(dirs)

    spec = dict(BASES[args.base])
    if args.config:
        spec["config"] = args.config
    return drive(dirs, spec, args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
