"""Held-out benchmark for the RTX-3090 fine-tune (Windows-first, also POSIX).

Sibling to ``run_3090_experiment.py``. It **reuses that run's folder** — point
``--dir`` at the same experiment dir and it borrows the venv, the HF cache and
the merged weights, adds two model rows to ``configs/models.yaml`` and scores
**fine-tune vs its untuned base** on datasets the fine-tune never saw:

    python scripts/run_3090_benchmark.py --dir C:\\Users\\User\\a-experiment\\exp --detach

Everything it writes lives under ``<dir>/bench/`` (never inside the git repo, so
re-syncing the repo can't wipe the cache or results).

Phases (idempotent, tracked in ``<dir>/bench_state.json``):

    check -> register -> smoke -> full -> collect

  * **check**    — the shared venv imports torch (CUDA) and the merged
                   fine-tune dir exists.
  * **register** — add ``qwen3-vl-4b-bf16`` + ``exp3090-qwen3vl4b-ft-bf16``
                   (``--with-4bit`` also adds ``exp3090-qwen3vl4b-ft-4bit``).
  * **smoke**    — dataset by dataset: score every model on the first
                   ``--smoke-n`` (20) lines, copy the summary into
                   ``<dir>/bench/smoke/<dataset>/``, then the next dataset.
  * **full**     — dataset by dataset in **accumulate** mode: each run scores the
                   next ``--full-n`` (200) *un-scored* lines per dataset and
                   copies the batch summary into ``<dir>/bench/full/<dataset>/``.
                   Re-run to continue where it left off; done when every dataset
                   is exhausted.
  * **collect**  — render ``<dir>/bench/leaderboard.html`` over everything scored
                   so far.

Same ergonomics as the training runner: ``--detach`` survives logout, re-running
resumes, one run per folder (a second launch is refused, and a stray orphaned
``bench run`` is reaped on start), each launch rotates ``<dir>/logs/bench.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc  # datetime.UTC alias; not present before Python 3.11

REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

PHASES = ["check", "register", "smoke", "full", "collect"]

DEFAULT_EXPERIMENT = "exp3090-qwen3vl4b"
DEFAULT_BASE_HF_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_BASE_NAME = "qwen3-vl-4b-bf16"   # models.yaml row for the untuned base
DEFAULT_ARCH = "qwen2_5_vl"
MATRIX_FILE = "configs/training/exp3090_bench.run.yaml"
BANNED_DATASETS = ("muharaf_lines", "kitab_muharaf")  # overlap the fine-tune's training data
PROMPT_VERSION = "line-v2"
NORM_PROFILES = ("raw", "light", "diac_insensitive")

_CHILD: subprocess.Popen | None = None  # current `bench run` subprocess, for cleanup


# --------------------------------------------------------------------------- #
# small helpers (mirrors run_3090_experiment.py)
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dirs(root: Path, shared: Path | None = None) -> dict:
    """Same layout rule as run_3090_experiment.py's `_dirs`: `shared` (from
    `--shared`) moves the venv and the HF/uv caches out to a path reused across
    experiments, so a multi-model queue keeps ONE venv and ONE model cache.
    Must be given the same value the training run used, or this would build a
    second venv and re-download every base model it scores."""
    reusable = shared or root
    d = {
        "root": root,
        "venv": reusable / "venv",
        "hf_cache": reusable / "hf_cache",
        "uv_cache": reusable / "uv_cache",
        "merged": root / "merged",
        "bench": root / "bench",
        "results": root / "bench" / "results",
        "cfg": root / "bench" / "_cfg",
        "logs": root / "logs",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _log(dirs: dict, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with (dirs["logs"] / "bench.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _rotate_log(dirs: dict) -> None:
    log = dirs["logs"] / "bench.log"
    if log.exists() and log.stat().st_size > 0:
        stamp = _now().replace(":", "").replace("-", "")
        log.rename(dirs["logs"] / f"bench.{stamp}.log")


def _load_env_file(path: Path) -> None:
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
    return dirs["root"] / "bench_state.json"


def _read_state(dirs: dict) -> dict:
    p = _state_path(dirs)
    if not p.exists():
        return {"experiments": {}, "created": _now()}
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"experiments": {}, "created": _now()}
    # Reconcile against disk. Phases are trusted per experiment only if
    # merged/<exp>/config.json exists — a second --base could have clobbered
    # state["experiment"] and left stale phase records under the wrong name.
    trained = {
        md.name for md in (dirs["merged"].iterdir() if dirs["merged"].exists() else [])
        if (md / "config.json").exists()
    }
    flat = st.pop("phases", None)
    exp = st.get("experiment")
    if flat and exp in trained:
        b = st.setdefault("experiments", {}).setdefault(exp, {}).setdefault("phases", {})
        for k, v in flat.items():
            b.setdefault(k, v)
    for name, edata in list(st.get("experiments", {}).items()):
        if name not in trained:
            edata["phases"] = {}
    return st


def _write_state(dirs: dict, state: dict) -> None:
    _state_path(dirs).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _phases(state: dict, exp: str) -> dict:
    """This experiment's phase-record dict (per-experiment, so a second --base in
    the same --dir doesn't inherit the first one's check/smoke/full)."""
    return state.setdefault("experiments", {}).setdefault(exp, {}).setdefault("phases", {})


def _mark(dirs: dict, state: dict, phase: str, exp: str, **extra) -> None:
    _phases(state, exp)[phase] = {"done": True, "at": _now(), **extra}
    _write_state(dirs, state)


def _done(state: dict, phase: str, exp: str) -> bool:
    return bool(_phases(state, exp).get(phase, {}).get("done"))


def _venv_python(dirs: dict) -> Path:
    return dirs["venv"] / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _stdout_is_regular_file() -> bool:
    """True only when this process's stdout is a plain file (the detached
    worker's own log). A tty or a pipe (parent script reading us) is not — we
    must still echo to stdout in that case or the parent/log sees nothing."""
    try:
        import stat as _s
        return _s.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode)
    except Exception:  # noqa: BLE001
        return False


HEARTBEAT_S = 90  # log a liveness line after this long with no subprocess output


def _run(dirs: dict, cmd: list[str], *, phase: str) -> int:
    global _CHILD
    _log(dirs, f"$ {' '.join(str(c) for c in cmd)}")
    tee = not _stdout_is_regular_file()  # tty or pipe -> echo; detached log file -> don't dup
    with (dirs["logs"] / "bench.log").open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(REPO), env={**os.environ},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        _CHILD = proc
        assert proc.stdout is not None
        q: queue.Queue = queue.Queue()

        def _reader() -> None:
            for ln in proc.stdout:
                q.put(ln)
            q.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        last = time.monotonic()
        try:
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
                if tee:
                    sys.stdout.write(ln)
                    sys.stdout.flush()
                fh.write(ln)
            proc.wait()
        finally:
            _CHILD = None
    if proc.returncode != 0:
        _log(dirs, f"!! phase {phase}: command exited {proc.returncode}")
    return proc.returncode


def _kill_child() -> None:
    if _CHILD and _CHILD.poll() is None:
        _CHILD.terminate()
        try:
            _CHILD.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _CHILD.kill()


def _reap_orphans() -> None:
    """Kill a stray `bench run` left behind by a previously hard-killed worker
    (it holds the GPU and a fresh run would fight it)."""
    if IS_WIN:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -match 'arabic_ocr_bench\\.cli run' } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, check=False)
    else:
        subprocess.run(["pkill", "-f", "arabic_ocr_bench.cli run"],
                       capture_output=True, check=False)


def _datasets(dirs: dict, training_config: str | None) -> list[str]:
    """The held-out dataset list to sweep.

    With `--training-config <path>` (the model's own training YAML), this
    calls `safe_eval_datasets()` in the venv — every registered line-track
    dataset EXCEPT `synthetic` and anything that overlaps one of that config's
    `data.sources` (either an exact scored (hf_id, split) match, or a declared
    `overlaps_source`, e.g. `kitab_muharaf` when Muharaf is a training
    source). This generalizes the sweep to "every dataset" per model, instead
    of the fixed 6-dataset MATRIX_FILE list every model used to share
    regardless of what it was actually trained on.

    Without `--training-config` (back-compat / ad-hoc standalone use), falls
    back to parsing MATRIX_FILE's curated list (stdlib, no yaml — this
    function itself must stay launcher-safe; the training-config path below
    shells out to the venv precisely because `safe_eval_datasets` needs
    pydantic/yaml that may not exist in the bare launcher Python).
    """
    if training_config:
        py = _venv_python(dirs)
        code = (
            "from arabic_ocr_bench.training.config import load_training_config;"
            "from arabic_ocr_bench.training.data import safe_eval_datasets;"
            f"cfg = load_training_config({training_config!r});"
            "print('\\n'.join(safe_eval_datasets(cfg.data)))"
        )
        out = subprocess.run(
            [str(py), "-c", code], cwd=str(REPO), capture_output=True, text=True, check=False
        )
        if out.returncode != 0:
            raise SystemExit(
                f"_datasets: failed to compute safe_eval_datasets for {training_config}:\n{out.stderr}"
            )
        return [ln for ln in out.stdout.splitlines() if ln.strip()]

    datasets: list[str] = []
    in_block = False
    for raw in (REPO / MATRIX_FILE).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith((" ", "\t")):
            in_block = line.strip().rstrip(":") == "datasets"
            continue
        if in_block and line.lstrip().startswith("- "):
            datasets.append(line.lstrip()[2:].strip())
    return datasets


def _bench_models(spec: dict, with_4bit: bool) -> list[str]:
    m = [spec["base_name"], f"{spec['experiment']}-ft-bf16"]
    if with_4bit:
        m.append(f"{spec['experiment']}-ft-4bit")
    return m


def _write_run_cfg(path: Path, *, models: list[str], dataset: str, limit: int,
                   accumulate: bool, results_dir: Path) -> None:
    path.write_text(
        "# generated by run_3090_benchmark.py — do not edit\n"
        f"models: [{', '.join(models)}]\n"
        f"datasets: [{dataset}]\n"
        f"limit: {limit}\n"
        f"offset: 0\n"
        f"accumulate: {'true' if accumulate else 'false'}\n"
        f"prompt_version: {PROMPT_VERSION}\n"
        f"normalization_profiles: [{', '.join(NORM_PROFILES)}]\n"
        "concurrency: 1\n"
        "max_retries: 2\n"
        "cache: true\n"
        f"results_dir: {results_dir.as_posix()}\n",
        encoding="utf-8",
    )


def _collect_latest(dirs: dict, stage: str, dataset: str) -> str | None:
    """Copy the newest results/<run_id>/ into <dir>/bench/<stage>/<dataset>/."""
    runs = sorted(dirs["results"].glob("*/summary.csv"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    src = runs[-1].parent
    dst = dirs["bench"] / stage / dataset / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summary.csv", "summary.md", "scores.jsonl", "predictions.jsonl",
              "run_config.snapshot.yaml"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    _log(dirs, f"{stage}: {dataset} -> {dst}")
    return src.name


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def phase_check(dirs: dict, state: dict, spec: dict, args) -> None:
    py = _venv_python(dirs)
    if not py.exists():
        raise SystemExit(
            f"check: no venv at {dirs['venv']} — run scripts/run_3090_experiment.py "
            f"--dir {dirs['root']} first (it builds the venv this script reuses)."
        )
    merged = dirs["merged"] / spec["experiment"]
    if not (merged / "config.json").exists():
        raise SystemExit(
            f"check: merged weights not found at {merged} — finish the training run "
            f"(its `merge` phase) before benchmarking."
        )
    if _run(dirs, [str(py), "-c",
                   "import torch,arabic_ocr_bench;"
                   "assert torch.cuda.is_available(), 'CUDA not available in the venv';"
                   "print('cuda', torch.version.cuda, torch.cuda.get_device_name(0))"],
            phase="check"):
        raise SystemExit("check: venv sanity import failed")
    _mark(dirs, state, "check", spec["experiment"], merged=str(merged))


def phase_register(dirs: dict, state: dict, spec: dict, args) -> None:
    # Always runs: it writes into aocr-repo/configs/models.yaml, which a repo
    # re-sync resets — a persistent "done" mark would leave the rows missing.
    # The register calls are idempotent overwrites, ~1 s total.
    py = _venv_python(dirs)
    merged = str(dirs["merged"] / spec["experiment"])
    # 640-token cap (vs the 160 default) so page-scale datasets (kitab_hindawi,
    # kitab_historicalbooks — GT ~350+ tokens) aren't hard-truncated. Line
    # datasets stop at EOS well before this, so no slowdown there.
    reg = [str(py), "-m", "arabic_ocr_bench.training.cli", "register",
           "--arch", spec["arch"], "--max-new-tokens", str(args.max_new_tokens)]
    rows = [
        (spec["base_name"], spec["base_hf_id"], [],
         f"Untuned {spec['base_hf_id']} at bf16 — base row for the "
         f"{spec['experiment']} held-out benchmark."),
        (f"{spec['experiment']}-ft-bf16", merged, [],
         f"{spec['experiment']} RTX-3090 fine-tune (bf16), merged weights."),
    ]
    if args.with_4bit:
        rows.append((f"{spec['experiment']}-ft-4bit", merged, ["--load-in-4bit"],
                     f"{spec['experiment']} RTX-3090 fine-tune (4-bit), merged weights."))
    for name, path, extra, card in rows:
        if _run(dirs, [*reg, "--name", name, "--path", path, "--card", card, *extra],
                phase="register"):
            raise SystemExit(f"register: `bench-train register {name}` failed")
    _mark(dirs, state, "register", spec["experiment"], models=[r[0] for r in rows])


def _sweep(dirs: dict, state: dict, spec: dict, args, *, stage: str,
           limit: int, accumulate: bool) -> bool:
    """Run one pass over every dataset. Returns True if the stage is complete."""
    training_config = spec.get("training_config")
    datasets = _datasets(dirs, training_config)
    models = _bench_models(spec, args.with_4bit)
    if not training_config:
        # Legacy MATRIX_FILE path only — it has no `overlaps_source` check of
        # its own, so this hardcoded net is still load-bearing there. The
        # training-config-driven path above already resolved this correctly
        # (and deliberately INCLUDES muharaf_lines when it's genuinely safe —
        # e.g. a model trained on `muharaf_train`, scored on the disjoint
        # `test` split — so this same check must not run for that path).
        banned = sorted(set(datasets) & set(BANNED_DATASETS))
        if banned:
            raise SystemExit(f"{stage}: {MATRIX_FILE} lists {banned} (Muharaf = training data).")
    py = _venv_python(dirs)
    exhausted: list[str] = []
    for ds in datasets:
        key = f"{stage}:{ds}"
        if _done(state, key, spec["experiment"]) and not args.force:
            _log(dirs, f"{stage}: {ds} already done")
            exhausted.append(ds)
            continue
        cfg = dirs["cfg"] / f"{stage}__{ds}.run.yaml"
        _write_run_cfg(cfg, models=models, dataset=ds, limit=limit,
                       accumulate=accumulate, results_dir=dirs["results"])
        if _run(dirs, [str(py), "-m", "arabic_ocr_bench.cli", "run",
                       "--config", str(cfg), "--report"], phase=stage):
            raise SystemExit(f"{stage}: `bench run` failed on {ds} — fix and re-run (resumes).")
        run_id = _collect_latest(dirs, stage, ds)
        # dataset is exhausted when the last batch scored fewer NEW lines than asked
        new_lines = _run_new_line_count(dirs, run_id, models[0], ds) if run_id else 0
        if not accumulate or new_lines < limit:
            _mark(dirs, state, key, spec["experiment"], run_id=run_id)
            exhausted.append(ds)
        else:
            _log(dirs, f"{stage}: {ds} — {new_lines} more lines scored; re-run to continue")
    return len(exhausted) == len(datasets)


def _run_new_line_count(dirs: dict, run_id: str, model: str, dataset: str) -> int:
    pred = dirs["results"] / run_id / "predictions.jsonl"
    if not pred.exists():
        return 0
    n = 0
    for line in pred.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("model") == model and r.get("dataset") == dataset and not r.get("cached"):
            n += 1
    return n


def phase_smoke(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "smoke", spec["experiment"]) and not args.force:
        _log(dirs, "smoke: already done")
        return
    if _sweep(dirs, state, spec, args, stage="smoke", limit=args.smoke_n, accumulate=False):
        _mark(dirs, state, "smoke", spec["experiment"])


def phase_full(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "full", spec["experiment"]) and not args.force:
        _log(dirs, "full: already done")
        return
    if _sweep(dirs, state, spec, args, stage="full", limit=args.full_n, accumulate=True):
        _mark(dirs, state, "full", spec["experiment"])
        _log(dirs, "full: every dataset exhausted")
    else:
        _log(dirs, "full: batch done — re-run this command to score the next batch")


def phase_collect(dirs: dict, state: dict, spec: dict, args) -> None:
    py = _venv_python(dirs)
    _run(dirs, [str(py), str(REPO / "scripts" / "make_leaderboard.py"),
                str(dirs["results"]), "-o", str(dirs["bench"] / "leaderboard.html")],
         phase="collect")
    # One fetchable bundle: the curated per-batch summaries + the leaderboard.
    # Excludes bench/results/ and its _cache/ (hundreds of hash-named JSON that
    # only matter on the box, for resume).
    bundle = dirs["bench"] / "bundle.zip"
    reports = dirs["root"] / "reports"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("smoke", "full"):
            root = dirs["bench"] / sub
            for f in root.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(dirs["bench"]))
        lb = dirs["bench"] / "leaderboard.html"
        if lb.exists():
            z.write(lb, lb.name)
        # per-experiment training snapshots (loss curve JSON, effective config,
        # in-domain eval, plots) for the Fine-tuning tab
        for f in (reports.rglob("*") if reports.exists() else []):
            if f.is_file():
                z.write(f, Path("reports") / f.relative_to(reports))
    _log(dirs, f"collect: bundle -> {bundle} ({bundle.stat().st_size // 1024} KB)")
    _mark(dirs, state, "collect", spec["experiment"], at=_now(), bundle=str(bundle))


PHASE_FN = {
    "check": phase_check, "register": phase_register,
    "smoke": phase_smoke, "full": phase_full, "collect": phase_collect,
}


# --------------------------------------------------------------------------- #
# detach + status + drive (mirrors run_3090_experiment.py)
# --------------------------------------------------------------------------- #
def _respawn_detached(dirs: dict, argv: list[str]) -> int:
    _rotate_log(dirs)
    logf = (dirs["logs"] / "bench.log").open("a", encoding="utf-8")
    child_argv = [sys.executable, str(Path(__file__).resolve()), *[a for a in argv if a != "--detach"]]
    kwargs: dict = {"cwd": str(REPO), "stdout": logf, "stderr": subprocess.STDOUT,
                    "stdin": subprocess.DEVNULL, "close_fds": True}
    if IS_WIN:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(child_argv, **kwargs)
    (dirs["logs"] / "bench.pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"detached pid {proc.pid}")
    print(f"log : {dirs['logs'] / 'bench.log'}")
    print(f"poll: python {Path(__file__).name} --dir {dirs['root']} status")
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        if IS_WIN:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, check=False).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _active_worker(dirs: dict) -> int | None:
    pidf = dirs["logs"] / "bench.pid"
    if not pidf.exists():
        return None
    try:
        pid = int(pidf.read_text(encoding="utf-8").strip() or 0)
    except ValueError:
        return None
    if pid and pid != os.getpid() and _pid_alive(pid):
        return pid
    return None


def cmd_status(dirs: dict, exp: str) -> int:
    state = _read_state(dirs)
    ph_map = _phases(state, exp)
    print(f"experiment dir : {dirs['root']}")
    print(f"experiment     : {exp}")
    pidf = dirs["logs"] / "bench.pid"
    if pidf.exists():
        pid = int(pidf.read_text(encoding="utf-8").strip() or 0)
        print(f"worker pid     : {pid}  ({'alive' if _pid_alive(pid) else 'not running'})")
    print("\nphases:")
    for ph in PHASES:
        rec = ph_map.get(ph, {})
        print(f"  {ph:<9} {'done' if rec.get('done') else '--':<6} {rec.get('at', '')}")
    sub = {k: v for k, v in ph_map.items() if ":" in k}
    if sub:
        print("\nper-dataset:")
        for k in sorted(sub):
            print(f"  {k:<22} done  {sub[k].get('at', '')}")
    for stage in ("smoke", "full"):
        got = sorted((dirs["bench"] / stage).glob("*/*/summary.csv")) if (dirs["bench"] / stage).exists() else []
        if got:
            print(f"\n{stage} results : {dirs['bench'] / stage}  ({len(got)} batch summaries)")
    log = dirs["logs"] / "bench.log"
    if log.exists():
        print("\n--- log tail (40) ---")
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
    return 0


def drive(dirs: dict, spec: dict, args, argv: list[str]) -> int:
    state = _read_state(dirs)
    state["experiment"] = spec["experiment"]
    state.setdefault("started", _now())
    _write_state(dirs, state)

    active = _active_worker(dirs)
    if active and not args.force:
        print(
            f"a benchmark run is already active in this folder (pid {active}).\n"
            f"  watch it : python {Path(__file__).name} --dir {dirs['root']} status\n"
            f"  stop it  : {'taskkill /F /PID' if IS_WIN else 'kill'} {active}\n"
            f"  or pass --force to start anyway (only if you know it is dead)."
        )
        return 1

    if args.detach:
        return _respawn_detached(dirs, argv)

    _reap_orphans()
    _rotate_log(dirs)
    (dirs["logs"] / "bench.pid").write_text(str(os.getpid()), encoding="utf-8")
    start = args.from_phase or ("check" if not args.only else args.only)
    plan = [args.only] if args.only else PHASES[PHASES.index(start):]
    _log(dirs, f"=== benchmark: experiment={spec['experiment']} phases={plan} ===")
    try:
        for ph in plan:
            try:
                PHASE_FN[ph](dirs, state, spec, args)
            except SystemExit as e:
                _log(dirs, f"STOP at phase {ph}: {e}")
                return 1
            state = _read_state(dirs)
        _log(dirs, "=== pass complete ===")
        _log(dirs, f"results : {dirs['bench']}")
        return 0
    finally:
        _kill_child()
        pidf = dirs["logs"] / "bench.pid"
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="run_3090_benchmark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="the run_3090_experiment.py folder to reuse (venv + merged weights)")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "status"],
                    help="run (default) | status")
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                    help=f"merged-dir name under <dir>/merged/ (default: {DEFAULT_EXPERIMENT})")
    ap.add_argument("--base-hf-id", dest="base_hf_id", default=DEFAULT_BASE_HF_ID,
                    help="hf_id of the untuned base to register + score")
    ap.add_argument("--base-name", dest="base_name", default=DEFAULT_BASE_NAME,
                    help=f"models.yaml row name for the base (default: {DEFAULT_BASE_NAME})")
    ap.add_argument("--arch", default=DEFAULT_ARCH)
    ap.add_argument(
        "--training-config",
        dest="training_config",
        default=None,
        help="the model's own training YAML — when given, the dataset sweep "
        "is every safe_eval_datasets() result for it (every registered "
        "line-track dataset except what it trained on), instead of the "
        "fixed MATRIX_FILE list",
    )
    ap.add_argument("--smoke-n", dest="smoke_n", type=int, default=20,
                    help="lines per dataset per model in the smoke stage (default 20)")
    ap.add_argument("--full-n", dest="full_n", type=int, default=200,
                    help="new lines per dataset per full-stage batch (default 200)")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=640,
                    help="generation cap for the registered rows (default 640)")
    ap.add_argument("--with-4bit", dest="with_4bit", action="store_true",
                    help="also score the 4-bit fine-tune row")
    ap.add_argument("--detach", action="store_true", help="re-spawn free of this shell and return")
    ap.add_argument("--from", dest="from_phase", default=None, choices=PHASES,
                    help="start at this phase")
    ap.add_argument("--only", default=None, choices=PHASES, help="run just this one phase")
    ap.add_argument("--force", action="store_true",
                    help="re-run phases already marked done, and ignore the run lock")
    ap.add_argument(
        "--shared", default=None,
        help="shared venv + HF/uv cache path — pass the SAME value the training run "
             "used (run_3090_experiment.py --shared), or this builds a second venv "
             "and re-downloads every base model it scores",
    )
    args = ap.parse_args(argv)

    dirs = _dirs(
        Path(args.dir).expanduser().resolve(),
        Path(args.shared).expanduser().resolve() if args.shared else None,
    )
    _pin_env(dirs)
    if args.cmd == "status":
        return cmd_status(dirs, args.experiment)

    spec = {"experiment": args.experiment, "base_hf_id": args.base_hf_id,
            "base_name": args.base_name, "arch": args.arch,
            "training_config": args.training_config}
    return drive(dirs, spec, args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
