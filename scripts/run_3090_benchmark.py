"""Held-out benchmark for the RTX-3090 fine-tune (Windows-first, also POSIX).

Sibling to ``run_3090_experiment.py``. It **reuses that run's folder** — point
``--dir`` at the same experiment dir and it borrows the venv, the HF cache and
the merged weights, adds a few model rows to ``configs/models.yaml`` and scores
**base vs fine-tune** on datasets the fine-tune never saw:

    python scripts/run_3090_benchmark.py --dir C:\\Users\\User\\a-experiment\\exp --detach

Phases (idempotent, tracked in ``<dir>/bench_state.json``):

    check -> register -> run -> collect

  * **check**    — the shared venv imports torch (CUDA) and the merged
                   fine-tune dir exists.
  * **register** — add ``qwen3-vl-4b-bf16`` + ``exp3090-qwen3vl4b-ft-{4bit,bf16}``
                   to ``configs/models.yaml`` (the 4-bit base ``qwen3-vl-4b`` is
                   already there).
  * **run**      — ``bench run --config <benchmark run config> --report`` over the
                   models x datasets matrix. Refuses to start if the config's
                   ``datasets`` names a Muharaf set (the fine-tune trained on it).
  * **collect**  — copy the newest ``results/<run_id>/`` into ``<dir>/bench/`` and
                   render a standalone ``<dir>/bench/leaderboard.html``.

Same ergonomics as the training runner: ``--detach`` survives logout, re-running
resumes from the last finished phase, one run per folder (a second launch is
refused), and each launch rotates ``<dir>/logs/bench.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc  # datetime.UTC alias; not present before Python 3.11

REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

PHASES = ["check", "register", "run", "collect"]

# Defaults for the qwen3vl4b training run; flags below override for other bases.
DEFAULT_EXPERIMENT = "exp3090-qwen3vl4b"
DEFAULT_BASE_HF_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_ARCH = "qwen2_5_vl"
DEFAULT_RUN_CONFIG = "configs/training/exp3090_bench.run.yaml"
BANNED_DATASETS = ("muharaf_lines", "kitab_muharaf")  # overlap the fine-tune's training data


# --------------------------------------------------------------------------- #
# small helpers (mirrors run_3090_experiment.py)
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dirs(root: Path) -> dict:
    d = {
        "root": root,
        "venv": root / "venv",
        "hf_cache": root / "hf_cache",
        "uv_cache": root / "uv_cache",
        "merged": root / "merged",
        "bench": root / "bench",
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
    """Move a non-empty ``bench.log`` aside so each launch starts clean."""
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
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"phases": {}, "created": _now()}


def _write_state(dirs: dict, state: dict) -> None:
    _state_path(dirs).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _mark(dirs: dict, state: dict, phase: str, **extra) -> None:
    state["phases"][phase] = {"done": True, "at": _now(), **extra}
    _write_state(dirs, state)


def _done(state: dict, phase: str) -> bool:
    return bool(state["phases"].get(phase, {}).get("done"))


def _venv_python(dirs: dict) -> Path:
    return dirs["venv"] / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _run(dirs: dict, cmd: list[str], *, phase: str) -> int:
    _log(dirs, f"$ {' '.join(str(c) for c in cmd)}")
    with (dirs["logs"] / "bench.log").open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(REPO), env={**os.environ},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for ln in proc.stdout:
            sys.stdout.write(ln)
            sys.stdout.flush()
            fh.write(ln)
        proc.wait()
    if proc.returncode != 0:
        _log(dirs, f"!! phase {phase}: command exited {proc.returncode}")
    return proc.returncode


def _config_datasets(path: Path) -> list[str]:
    """List items under a top-level ``datasets:`` key (stdlib, no yaml)."""
    out: list[str] = []
    in_block = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith((" ", "\t")):
            in_block = line.strip() == "datasets:"
            continue
        if in_block and line.lstrip().startswith("- "):
            out.append(line.lstrip()[2:].strip())
    return out


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
    rc = _run(dirs, [str(py), "-c",
                     "import torch,arabic_ocr_bench;"
                     "assert torch.cuda.is_available(), 'CUDA not available in the venv';"
                     "print('cuda', torch.version.cuda, torch.cuda.get_device_name(0))"],
              phase="check")
    if rc:
        raise SystemExit("check: venv sanity import failed")
    _mark(dirs, state, "check", merged=str(merged))


def phase_register(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "register") and not args.force:
        _log(dirs, "register: already done")
        return
    py = _venv_python(dirs)
    merged = str(dirs["merged"] / spec["experiment"])
    reg = [str(py), "-m", "arabic_ocr_bench.training.cli", "register", "--arch", spec["arch"]]
    rows = [
        # name, path, extra flags
        ("qwen3-vl-4b-bf16", spec["base_hf_id"], []),
        (f"{spec['experiment']}-ft-4bit", merged, ["--load-in-4bit"]),
        (f"{spec['experiment']}-ft-bf16", merged, []),
    ]
    for name, path, extra in rows:
        card = (
            f"Untuned {spec['base_hf_id']} at bf16 — base row for the "
            f"{spec['experiment']} held-out benchmark."
            if path == spec["base_hf_id"]
            else f"{spec['experiment']} RTX-3090 fine-tune "
                 f"({'4-bit' if extra else 'bf16'}), merged weights."
        )
        if _run(dirs, [*reg, "--name", name, "--path", path, "--card", card, *extra],
                phase="register"):
            raise SystemExit(f"register: `bench-train register {name}` failed")
    _mark(dirs, state, "register", models=[r[0] for r in rows])


def phase_run(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "run") and not args.force:
        _log(dirs, "run: already done")
        return
    cfg = REPO / spec["run_config"]
    if not cfg.exists():
        raise SystemExit(f"run: config not found: {cfg}")
    banned = sorted(set(_config_datasets(cfg)) & set(BANNED_DATASETS))
    if banned:
        raise SystemExit(
            f"run: {cfg.name} lists {banned} — the fine-tune trained on the Muharaf "
            f"train split, so these would be contaminated. Remove them from `datasets:`."
        )
    py = _venv_python(dirs)
    rc = _run(dirs, [str(py), "-m", "arabic_ocr_bench.cli", "run",
                     "--config", str(cfg), "--report"], phase="run")
    if rc:
        raise SystemExit("run: `bench run` failed")
    _mark(dirs, state, "run", config=str(cfg))


def phase_collect(dirs: dict, state: dict, spec: dict, args) -> None:
    runs = sorted((REPO / "results").glob("*/summary.csv"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("collect: no results/<run_id>/summary.csv found — did `run` succeed?")
    src = runs[-1].parent
    dst = dirs["bench"] / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summary.csv", "summary.md", "scores.jsonl", "predictions.jsonl",
              "run_config.snapshot.yaml"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    _log(dirs, f"collect: {src.name}/ -> {dst}")
    py = _venv_python(dirs)
    _run(dirs, [str(py), str(REPO / "scripts" / "make_leaderboard.py"),
                str(REPO / "results"), "-o", str(dirs["bench"] / "leaderboard.html")],
         phase="collect")
    _mark(dirs, state, "collect", run_id=src.name, out=str(dst))


PHASE_FN = {
    "check": phase_check, "register": phase_register,
    "run": phase_run, "collect": phase_collect,
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


def cmd_status(dirs: dict) -> int:
    state = _read_state(dirs)
    print(f"experiment dir : {dirs['root']}")
    pidf = dirs["logs"] / "bench.pid"
    if pidf.exists():
        pid = int(pidf.read_text(encoding="utf-8").strip() or 0)
        print(f"worker pid     : {pid}  ({'alive' if _pid_alive(pid) else 'not running'})")
    print("\nphases:")
    for ph in PHASES:
        rec = state["phases"].get(ph, {})
        print(f"  {ph:<9} {'done' if rec.get('done') else '--':<6} {rec.get('at', '')}")
    collected = sorted(dirs["bench"].glob("*/summary.csv"))
    if collected:
        print(f"\ncollected      : {collected[-1].parent}")
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
        _log(dirs, "=== all phases complete ===")
        _log(dirs, f"collected results : {dirs['bench']}")
        return 0
    finally:
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
                    help="base model hf_id for the bf16 base row")
    ap.add_argument("--arch", default=DEFAULT_ARCH, help="arch shim for the registry rows")
    ap.add_argument("--config", dest="run_config", default=DEFAULT_RUN_CONFIG,
                    help="bench run config (models x datasets matrix)")
    ap.add_argument("--detach", action="store_true", help="re-spawn free of this shell and return")
    ap.add_argument("--from", dest="from_phase", default=None, choices=PHASES,
                    help="start at this phase")
    ap.add_argument("--only", default=None, choices=PHASES, help="run just this one phase")
    ap.add_argument("--force", action="store_true",
                    help="re-run phases already marked done, and ignore the run lock")
    args = ap.parse_args(argv)

    dirs = _dirs(Path(args.dir).expanduser().resolve())
    _pin_env(dirs)
    if args.cmd == "status":
        return cmd_status(dirs)

    spec = {"experiment": args.experiment, "base_hf_id": args.base_hf_id,
            "arch": args.arch, "run_config": args.run_config}
    return drive(dirs, spec, args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
