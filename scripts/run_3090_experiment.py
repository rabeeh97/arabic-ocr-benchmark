"""Self-contained RTX-3090 fine-tuning runner (Windows-first, also POSIX).

One command takes a fresh box from nothing to a scored fine-tune:

    python scripts/run_3090_experiment.py --dir C:\\Users\\User\\a-experiment\\exp --base qwen3vl4b --detach

Everything lives under ``--dir``: the uv venv, the HF cache, the uv cache, the
dataset, checkpoints, merged weights, eval scores and plots. Deleting that one
folder removes every trace (no writes anywhere else). The run is:

  * **resumable**    — re-run the same command and it continues from the last
                       finished phase; training itself resumes from the newest
                       ``checkpoint-*``.
  * **detachable**   — ``--detach`` re-spawns the process free of the SSH session
                       (Windows ``DETACHED_PROCESS``; POSIX ``start_new_session``)
                       and returns immediately. It keeps running after logout.
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
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

UTC = timezone.utc  # datetime.UTC alias; not present before Python 3.11
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

PHASES = ["bootstrap", "fetch", "prepare", "train", "merge", "eval", "report"]

BASES = {
    "qwen3vl4b": {
        "experiment": "exp3090-qwen3vl4b",
        "config": "configs/training/exp3090_qwen3vl4b.yaml",
        "run_config": "configs/training/exp3090_qwen3vl4b.run.yaml",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "arch": "qwen3_vl",
        "needs_transformers": "4.57",
        "gated": False,
    },
    "baseer": {
        "experiment": "exp3090-baseer-nakba",
        "config": "configs/training/exp3090_baseer_nakba.yaml",
        "run_config": "configs/training/exp3090_baseer_nakba.run.yaml",
        "hf_id": "Misraj/Baseer__Nakba",
        "arch": "qwen2_5_vl",
        "needs_transformers": None,
        "gated": False,
    },
    "paligemma": {
        "experiment": "exp3090-paligemma2-3b",
        "config": "configs/training/exp3090_paligemma2_3b.yaml",
        "run_config": "configs/training/exp3090_paligemma2_3b.run.yaml",
        "hf_id": "google/paligemma2-3b-pt-448",
        "arch": "paligemma",
        "needs_transformers": "4.49",
        "gated": True,
    },
}

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
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with (dirs["logs"] / "run.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from <dir>/experiment.env into os.environ."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _pin_env(dirs: dict) -> None:
    os.environ["HF_HOME"] = str(dirs["hf_cache"])
    os.environ["UV_CACHE_DIR"] = str(dirs["uv_cache"])
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    _load_env_file(dirs["root"] / "experiment.env")


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
    if not boot and (dirs["venv"] / ("Scripts/python.exe" if IS_WIN else "bin/python")).exists():
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
    # every experiment: keep train-ish marks only if its merged dir exists
    for name in list(exps) + list(trained):
        ph = exps.setdefault(name, {}).setdefault("phases", {})
        if name in trained:
            for k in ("fetch", "prepare", "train", "merge"):
                ph.setdefault(k, {"done": True, "at": "reconciled-from-disk"})
        else:
            for k in ("fetch", "prepare", "train", "merge", "eval", "report"):
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


def _uv_bin(dirs: dict) -> str:
    """uv on PATH, else the copy we install into <dir>/bin."""
    onpath = shutil.which("uv")
    if onpath:
        return onpath
    local = dirs["bin"] / ("uv.exe" if IS_WIN else "uv")
    return str(local)


def _run(dirs: dict, cmd: list[str], *, phase: str, env: dict | None = None) -> int:
    """Run a subprocess, streaming combined output to the log and stdout."""
    _log(dirs, f"$ {' '.join(str(c) for c in cmd)}")
    e = {**os.environ, **(env or {})}
    # When detached, sys.stdout IS run.log (the child's fd 1) — mirroring to it
    # again would double every line. Only echo to stdout when it's a real tty.
    tee = sys.stdout.isatty()
    with (dirs["logs"] / "run.log").open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(REPO), env=e,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for ln in proc.stdout:
            if tee:
                sys.stdout.write(ln)
                sys.stdout.flush()
            fh.write(ln)
        proc.wait()
    if proc.returncode != 0:
        _log(dirs, f"!! phase {phase}: command exited {proc.returncode}")
    return proc.returncode


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def phase_bootstrap(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "bootstrap") and not args.force:
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
    if not py.exists() and _run(
        dirs, [uv, "venv", str(dirs["venv"]), "--python", "3.11"], phase="bootstrap"
    ):
        raise SystemExit("bootstrap: `uv venv` failed")

    pipbase = [uv, "pip", "install", "--python", str(py)]
    # 1. CUDA torch first so the CUDA build wins over the CPU wheel a later
    #    resolve would pick.
    if _run(dirs, [*pipbase, "torch", "torchvision", "--index-url", args.torch_index], phase="bootstrap"):
        raise SystemExit("bootstrap: torch (CUDA) install failed - check --torch-index / driver")
    # 2. the benchmark + training extras (editable, from the SSH-copied repo)
    if _run(dirs, [*pipbase, "-e", f"{REPO}[local,train]"], phase="bootstrap"):
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
    if _run(dirs, [*pipbase, "-P", "transformers", "-P", "trl",
                   f"transformers>={tfloor},<5", "trl>=0.21,<1.0"], phase="bootstrap"):
        raise SystemExit("bootstrap: transformers/trl pin install failed")
    _run(dirs, [*pipbase, "matplotlib>=3.7", "hf_transfer"], phase="bootstrap")

    # 4. verify CUDA + bitsandbytes
    check = (
        "import torch,bitsandbytes;"
        "assert torch.cuda.is_available(), 'CUDA not available in the venv';"
        "print('cuda', torch.version.cuda, torch.cuda.get_device_name(0));"
        "print('bitsandbytes', bitsandbytes.__version__)"
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

    # base model — snapshot_download gives a clean progress bar and fails fast on
    # a gated repo without a valid token.
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

    # training dataset (Muharaf train split)
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
        + (f"[s.__setitem__('limit',{args.limit}) for s in c['data']['sources']];" if args.limit is not None else "")
        + (f"c['train']['epochs']={args.epochs};" if args.epochs is not None else "")
        + f"open(r'{out}','w',encoding='utf-8').write(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))"
    )
    if _run(dirs, [str(py), "-c", code], phase="prepare"):
        raise SystemExit("prepare: could not build the effective config")
    return out


def phase_prepare(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "prepare", spec["experiment"]) and not args.force:
        _log(dirs, "prepare: already done")
        return
    py = _venv_python(dirs)
    eff = _effective_config(dirs, spec, args)
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
    _log(dirs, f"eval: register {name} + score on the held-out muharaf_lines test split")
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
    logf = (dirs["logs"] / "run.log").open("a", encoding="utf-8")
    child_argv = [sys.executable, str(Path(__file__).resolve()), *[a for a in argv if a != "--detach"]]
    kwargs: dict = {"cwd": str(REPO), "stdout": logf, "stderr": subprocess.STDOUT,
                    "stdin": subprocess.DEVNULL, "close_fds": True}
    if IS_WIN:
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(child_argv, **kwargs)
    (dirs["logs"] / "run.pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"detached pid {proc.pid}")
    print(f"log : {dirs['logs'] / 'run.log'}")
    print(f"poll: python {Path(__file__).name} --dir {dirs['root']} status")
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        if IS_WIN:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _active_worker(dirs: dict) -> int | None:
    """PID in ``run.pid`` if it names a live process other than us, else None."""
    pidf = dirs["logs"] / "run.pid"
    if not pidf.exists():
        return None
    try:
        pid = int(pidf.read_text(encoding="utf-8").strip() or 0)
    except ValueError:
        return None
    if pid and pid != os.getpid() and _pid_alive(pid):
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
