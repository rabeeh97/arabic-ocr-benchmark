"""One detached command: train a base + benchmark it vs its untuned self.

Chains ``run_3090_experiment.py`` (bootstrap -> train -> merge -> in-domain eval)
and ``run_3090_benchmark.py`` (register -> smoke -> full batch -> leaderboard +
bundle.zip) for one base model, so a whole ~10-16 h experiment is a single
line and one file to fetch:

    python scripts/run_3090_pipeline.py --dir C:\\Users\\User\\a-experiment\\exp --base baseer --detach

``--base`` is any name in ``_experiment_bases.BASES`` (see that module for the
full, single source-of-truth list — originally ``qwen3vl4b`` | ``qwen3vl4b-visft``
| ``baseer`` | ``paligemma``, plus the suite_v1 round-1 entries ``qwen3vl4b_suite``
| ``qwen3vl8b`` | ``ain7b`` | ``fanar2_oryx``, plus the later stacked job
``ain7b_nakba``). Re-run the same command
to resume (both sub-steps are individually resumable); the ``full`` benchmark
stage scores another 200 lines/dataset each run until every set is exhausted
unless the base sets ``no_full_accumulate``. For running the whole round-1 queue
in sequence, use ``run_experiment_suite.py`` instead of calling this per base
by hand.

Watch it:  Get-Content <dir>\\logs\\pipeline.log -Wait -Tail 50
Fetch:     <dir>\\bench\\bundle.zip   (curated summaries + leaderboard, no cache junk)
"""

from __future__ import annotations

import argparse
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

# Shared with run_3090_experiment.py / run_3090_benchmark.py / run_experiment_suite.py
# — see _experiment_bases.py's own docstring for why it's plain Python.
from _detach_spawn import (  # noqa: E402
    emit_log_line, pid_alive, pid_is_cmd, pidfile_is_other_worker,
    spawn_surviving_logout, stdout_is_regular_file,
)
from _experiment_bases import BASES, pipeline_bench_args  # noqa: E402


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(logs: Path, msg: str) -> None:
    emit_log_line(logs / "pipeline.log", f"[{_now()}] {msg}")


def _rotate(logs: Path) -> None:
    p = logs / "pipeline.log"
    if p.exists() and p.stat().st_size > 0:
        p.rename(logs / f"pipeline.{_now().replace(':', '').replace('-', '')}.log")


def _pid_alive(pid: int) -> bool:
    return pid_alive(pid)


HEARTBEAT_S = 90  # log a liveness line after this long with no subprocess output


def _run(logs: Path, cmd: list[str], *, step: str) -> int:
    _log(logs, f"=== {step}: {' '.join(cmd)}")
    tee = not stdout_is_regular_file()
    fh = (logs / "pipeline.log").open("a", encoding="utf-8") if tee else None
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
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
                _log(logs, f"...  {step}: still running (pid {proc.pid}, "
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
    _log(logs, f"=== {step}: exit {proc.returncode}")
    return proc.returncode


def _respawn_detached(logs: Path, argv: list[str]) -> int:
    _rotate(logs)
    child = [sys.executable, str(Path(__file__).resolve()), *[a for a in argv if a != "--detach"]]
    pid = spawn_surviving_logout(
        child, cwd=REPO, log_path=logs / "pipeline.log", pid_path=logs / "pipeline.pid"
    )
    print(f"detached pid {pid}")
    print(f"log : {logs / 'pipeline.log'}")
    print(f"watch: powershell -NoProfile -Command \"Get-Content {logs / 'pipeline.log'} -Wait -Tail 50\"")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="run_3090_pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="self-contained experiment folder")
    ap.add_argument("--base", required=True, choices=list(BASES))
    ap.add_argument("--limit", type=int, default=None, help="training-line count (default: config's 8000)")
    ap.add_argument("--venv-python", dest="venv_python", default=None,
                    help="interpreter for `uv venv` — pass a signed python.org 3.11 path "
                         "if Smart App Control blocks uv's CPython (WinError 4551)")
    ap.add_argument("--detach", action="store_true", help="re-spawn free of this shell and return")
    args = ap.parse_args(argv)

    root = Path(args.dir).expanduser().resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    pidf = logs / "pipeline.pid"
    if pidf.exists():
        try:
            other = int(pidf.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            other = 0
        if pidfile_is_other_worker(
            other, os.getpid(), os.getppid(),
            alive=_pid_alive(other), is_cmd=pid_is_cmd(other),
        ):
            print(f"a pipeline is already running here (pid {other}). Watch it, or "
                  f"{'taskkill /F /PID' if IS_WIN else 'kill'} {other}.")
            return 1

    if args.detach:
        return _respawn_detached(logs, argv)

    if not stdout_is_regular_file():
        _rotate(logs)
    pidf.write_text(str(os.getpid()), encoding="utf-8")
    spec = BASES[args.base]
    exp, hf_id, arch, base_name = (
        spec["experiment"], spec["hf_id"], spec["arch"], spec["base_model_row"]
    )
    py = sys.executable
    try:
        _log(logs, f"pipeline: base={args.base} experiment={exp} dir={root}")

        train = [py, str(REPO / "scripts" / "run_3090_experiment.py"),
                 "--dir", str(root), "--base", args.base, "--allow-contam"]
        if args.limit is not None:
            train += ["--limit", str(args.limit)]
        if args.venv_python:
            train += ["--venv-python", args.venv_python]
        if _run(logs, train, step="train"):
            _log(logs, "pipeline: STOP — training/eval did not finish (re-run to resume)")
            return 1

        if spec.get("ckpt_eval_after_train"):
            ckpt_eval = [
                py, str(REPO / "scripts" / "eval_ain_nakba_ckpts.py"),
                "--dir", str(root), "--base", args.base,
                "--official", "--no-wait",
            ]
            if _run(logs, ckpt_eval, step="ckpt_official"):
                _log(logs, "pipeline: STOP — official checkpoint walk did not finish")
                return 1

        bench = [py, str(REPO / "scripts" / "run_3090_benchmark.py"),
                 "--dir", str(root), "--experiment", exp,
                 "--base-hf-id", hf_id, "--base-name", base_name, "--arch", arch,
                 "--training-config", spec["config"]]
        extra = pipeline_bench_args(spec)
        if "--datasets-file" in extra:
            i = extra.index("--datasets-file")
            extra[i + 1] = str(REPO / extra[i + 1])
        bench += extra
        if _run(logs, bench, step="benchmark"):
            _log(logs, "pipeline: STOP — benchmark did not finish (re-run to resume / next full batch)")
            return 1

        _log(logs, f"pipeline: done. bundle -> {root / 'bench' / 'bundle.zip'}")
        return 0
    finally:
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
