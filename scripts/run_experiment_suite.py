"""Sequential multi-model training + full-benchmark queue (suite_v1, Windows-first).

    python scripts/run_experiment_suite.py --dir C:\\Users\\User\\suite-v1 --detach

Runs every entry in `_experiment_bases.round1_queue()`, IN ORDER, on ONE GPU
— never concurrently, since a single 3090 can't usefully train and evaluate
at the same time anyway. For each model:

  1. train -> merge -> in-domain eval, via `run_3090_experiment.py --dir
     <this-dir>/<model>` — reuses that script's bootstrap/fetch/prepare/
     train/merge/eval/report phase machine, crash-safety and resume-on-reboot
     state file UNCHANGED. Blocking (no `--detach` passed to the child).
  2. a curated cross-domain benchmark, via `run_3090_benchmark.py
     --datasets-file configs/training/suite_v1/bench.run.yaml` (kitab_khatt,
     nakba_lines, rasam1, rasam2 — one 200-line batch each, not every
     registered line-track dataset and not accumulate-until-exhausted).

Only after BOTH finish does the queue move to the next model. A model whose
manifest entry (`_experiment_bases.py`) is `gated` with `license_check` not
`"ok"` is skipped — logged, not queue-blocking. `fanar2_oryx` was confirmed
Apache-2.0 on 2026-09-18 and is in the queue.

Resumable: suite-level progress is tracked in `<dir>/suite_state.json`
(queued/running/done/failed/skipped per model); re-running the same command
skips already-`done` entries and resumes the in-progress one via ITS OWN
per-model `--dir` state (unchanged from `run_3090_experiment.py`/
`run_3090_benchmark.py`'s own resume logic — nothing new here, this script
only remembers which model it's on).

Watch it:  powershell -NoProfile -Command "Get-Content <dir>\\logs\\suite.log -Wait -Tail 50"
Status:    python scripts\\run_experiment_suite.py --dir <dir> status
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _detach_spawn import emit_log_line, spawn_surviving_logout, stdout_is_regular_file  # noqa: E402
from _envfile import load_experiment_env  # noqa: E402
from _experiment_bases import BASES, is_runnable, round1_queue  # noqa: E402

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"
HEARTBEAT_S = 90


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(logs: Path, msg: str) -> None:
    emit_log_line(logs / "suite.log", f"[{_now()}] {msg}")


def _rotate(logs: Path) -> None:
    p = logs / "suite.log"
    if p.exists() and p.stat().st_size > 0:
        p.rename(logs / f"suite.{_now().replace(':', '').replace('-', '')}.log")


def _state_path(root: Path) -> Path:
    return root / "suite_state.json"


def _read_state(root: Path) -> dict:
    p = _state_path(root)
    if not p.exists():
        return {"created": _now(), "queue": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"created": _now(), "queue": {}}


def _write_state(root: Path, state: dict) -> None:
    _state_path(root).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _set(root: Path, state: dict, name: str, status: str, **extra) -> None:
    state.setdefault("queue", {})[name] = {"status": status, "at": _now(), **extra}
    _write_state(root, state)


def _venv_python(root: Path) -> Path:
    """Python that has `arabic_ocr_bench` installed — not the system 3.10 launcher."""
    rel = Path("Scripts") / "python.exe" if IS_WIN else Path("bin") / "python"
    candidates = [root / "_shared" / "venv" / rel]
    candidates.extend(root / name / "venv" / rel for name in round1_queue())
    for p in candidates:
        if p.exists():
            return p
    return Path(sys.executable)


def _run(logs: Path, cmd: list[str], *, step: str) -> int:
    """Stream a child process's output to the suite log, with a heartbeat."""
    _log(logs, f"--- {step}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    q: "queue.Queue[str | None]" = queue.Queue()

    def _reader():
        assert proc.stdout is not None
        for ln in proc.stdout:
            q.put(ln)
        q.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    last = time.monotonic()
    # cmd.exe >> suite.log already holds the file; a second open('a') is
    # PermissionError 13 on Windows. When stdout is that log, write it only.
    fh = None if stdout_is_regular_file() else (logs / "suite.log").open("a", encoding="utf-8")
    try:
        while True:
            try:
                ln = q.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                _log(
                    logs,
                    f"...  {step}: still running (pid {proc.pid}, "
                    f"{int(time.monotonic() - last)}s since last output)",
                )
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
    _log(logs, f"=== {step}: exit {proc.returncode}")
    return proc.returncode or 0


def _bench_full_done(model_dir: Path, experiment: str) -> bool:
    p = model_dir / "bench_state.json"
    if not p.exists():
        return False
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(
        st.get("experiments", {}).get(experiment, {}).get("phases", {}).get("full", {}).get("done")
    )


def _run_one(root: Path, logs: Path, name: str, args) -> int:
    spec = BASES[name]
    model_dir = root / name
    py = sys.executable

    train_cmd = [
        py, str(REPO / "scripts" / "run_3090_experiment.py"),
        "--dir", str(model_dir), "--base", name, "--allow-contam",
    ]
    if args.venv_python:
        train_cmd += ["--venv-python", args.venv_python]
    rc = _run(logs, train_cmd, step=f"{name}: train")
    if rc:
        return rc

    bench_cmd = [
        py, str(REPO / "scripts" / "run_3090_benchmark.py"),
        "--dir", str(model_dir),
        "--experiment", spec["experiment"],
        "--base-hf-id", spec["hf_id"],
        "--base-name", spec["base_model_row"],
        "--arch", spec["arch"],
        "--training-config", spec["config"],
        "--datasets-file", str(REPO / "configs" / "training" / "suite_v1" / "bench.run.yaml"),
        "--no-full-accumulate",
    ]
    # One 200-line full batch per curated dataset (see bench.run.yaml). The
    # loop remains so a crash mid-sweep still resumes; with accumulate off,
    # `full` is marked done after a single successful pass.
    for i in range(args.max_bench_iters):
        rc = _run(logs, bench_cmd, step=f"{name}: bench (pass {i + 1}/{args.max_bench_iters})")
        if rc:
            return rc
        if _bench_full_done(model_dir, spec["experiment"]):
            return 0
    _log(
        logs,
        f"{name}: bench did not report `full` exhausted after {args.max_bench_iters} passes — "
        f"re-run the suite to continue (fully resumable, nothing lost)",
    )
    return 1


def _respawn_detached(logs: Path, argv: list[str]) -> int:
    _rotate(logs)
    child = [sys.executable, str(Path(__file__).resolve()), *[a for a in argv if a != "--detach"]]
    pid = spawn_surviving_logout(
        child, cwd=REPO, log_path=logs / "suite.log", pid_path=logs / "suite.pid"
    )
    print(f"detached pid {pid}")
    print(f"log : {logs / 'suite.log'}")
    print(f"watch: powershell -NoProfile -Command \"Get-Content {logs / 'suite.log'} -Wait -Tail 50\"")
    print(f"status: python {Path(__file__).name} --dir {logs.parent} status")
    return 0


def _pid_alive(pid: int) -> bool:
    if IS_WIN:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def cmd_status(root: Path) -> int:
    state = _read_state(root)
    print(f"suite dir: {root}")
    for name in round1_queue():
        q = state.get("queue", {}).get(name, {})
        print(f"  {name:<20} {q.get('status', 'queued'):<10} {q.get('at', '')} {q.get('reason', '')}")
    log = root / "logs" / "suite.log"
    if log.exists():
        print("\n--- log tail (40) ---")
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
    return 0


def drive(root: Path, args) -> int:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pidf = logs / "suite.pid"
    if pidf.exists():
        try:
            other = int(pidf.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            other = 0
        # WMI wrapper writes cmd.exe's pid first; the python child must not
        # treat its own parent as a rival run.
        if other and other not in (os.getpid(), os.getppid()) and _pid_alive(other):
            print(
                f"a suite run is already active here (pid {other}). Watch it, or "
                f"{'taskkill /F /PID' if IS_WIN else 'kill'} {other}."
            )
            return 1

    if args.detach:
        return _respawn_detached(logs, sys.argv[1:])

    # Parent already rotated before WMI spawn; renaming while cmd.exe holds
    # suite.log fails on Windows.
    if not stdout_is_regular_file():
        _rotate(logs)
    pidf.write_text(str(os.getpid()), encoding="utf-8")
    load_experiment_env(root)
    _log(
        logs,
        "env: HF_TOKEN=" + ("set" if os.environ.get("HF_TOKEN") else "MISSING"),
    )
    state = _read_state(root)
    try:
        for name in round1_queue():
            if state.get("queue", {}).get(name, {}).get("status") == "done" and not args.force:
                _log(logs, f"{name}: already done, skipping")
                continue
            ok, reason = is_runnable(name)
            if not ok:
                _log(logs, f"{name}: SKIP — {reason}")
                _set(root, state, name, "skipped", reason=reason)
                continue
            _set(root, state, name, "running")
            _log(logs, f"=== {name}: starting ===")
            rc = _run_one(root, logs, name, args)
            if rc:
                _set(root, state, name, "failed")
                _log(logs, f"{name}: FAILED (exit {rc}) — re-run the suite to resume from here")
                return 1
            _set(root, state, name, "done")
            _log(logs, f"=== {name}: done ===")
        _log(logs, "=== suite complete — rebuilding the top-level leaderboard ===")
        rc = _run(
            logs,
            [str(_venv_python(root)), str(REPO / "scripts" / "make_leaderboard.py"),
             "results", "-o", "results/leaderboard.html"],
            step="leaderboard",
        )
        if rc:
            _log(logs, f"leaderboard rebuild failed (exit {rc}) — models are done; re-run make_leaderboard.py")
        return rc
    finally:
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="run_experiment_suite", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", required=True, help="the suite's shared directory")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "status"])
    ap.add_argument("--detach", action="store_true", help="re-spawn free of this shell and return")
    ap.add_argument("--force", action="store_true", help="re-run entries already marked done")
    ap.add_argument(
        "--venv-python", dest="venv_python", default=None,
        help="forwarded to run_3090_experiment.py — see its own --venv-python help",
    )
    ap.add_argument(
        "--max-bench-iters", dest="max_bench_iters", type=int, default=50,
        help="cap on repeated full-stage bench passes per model before giving up "
        "(each pass scores --full-n new lines per curated dataset; with "
        "--no-full-accumulate a single pass marks full done)",
    )
    args = ap.parse_args(argv)

    root = Path(args.dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.cmd == "status":
        return cmd_status(root)
    return drive(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
