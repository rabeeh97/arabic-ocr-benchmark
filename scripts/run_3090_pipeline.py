"""One detached command: train a base + benchmark it vs its untuned self.

Chains ``run_3090_experiment.py`` (bootstrap -> train -> merge -> in-domain eval)
and ``run_3090_benchmark.py`` (register -> smoke -> full batch -> leaderboard +
bundle.zip) for one base model, so a whole ~10-16 h experiment is a single
line and one file to fetch:

    python scripts/run_3090_pipeline.py --dir C:\\Users\\User\\a-experiment\\exp --base baseer --detach

``--base`` is ``qwen3vl4b`` | ``qwen3vl4b-visft`` | ``baseer`` | ``paligemma``
(``-visft`` = same recipe, ViT unfrozen). Re-run the same command
to resume (both sub-steps are individually resumable); the ``full`` benchmark
stage scores another 200 lines/dataset each run until every set is exhausted.

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

# base -> (experiment name, base hf_id, benchmark arch shim, base models.yaml row)
BASES = {
    "qwen3vl4b": ("exp3090-qwen3vl4b", "Qwen/Qwen3-VL-4B-Instruct", "qwen2_5_vl", "qwen3-vl-4b-bf16"),
    "qwen3vl4b-visft": ("exp3090-qwen3vl4b-visft", "Qwen/Qwen3-VL-4B-Instruct", "qwen2_5_vl", "qwen3-vl-4b-bf16"),
    "baseer":    ("exp3090-baseer-nakba", "Misraj/Baseer__Nakba", "qwen2_5_vl", "baseer-nakba-bf16"),
    "paligemma": ("exp3090-paligemma2-3b", "google/paligemma2-3b-pt-448", "paligemma", "paligemma2-3b-bf16"),
    "paligemma-visft": ("exp3090-paligemma2-3b-visft", "google/paligemma2-3b-pt-448", "paligemma", "paligemma2-3b-bf16"),
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(logs: Path, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with (logs / "pipeline.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _rotate(logs: Path) -> None:
    p = logs / "pipeline.log"
    if p.exists() and p.stat().st_size > 0:
        p.rename(logs / f"pipeline.{_now().replace(':', '').replace('-', '')}.log")


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


def _run(logs: Path, cmd: list[str], *, step: str) -> int:
    _log(logs, f"=== {step}: {' '.join(cmd)}")
    tee = not _stdout_is_regular_file()  # tty or pipe -> echo; detached log file -> don't dup
    with (logs / "pipeline.log").open("a", encoding="utf-8") as fh:
        # PYTHONUNBUFFERED: without it, the child's stdout is block-buffered
        # (it's a pipe, not a tty) and pipeline.log can lag far behind the
        # child's own run.log by a full buffer — losing the tail end (e.g. a
        # crash traceback) if the child dies before that buffer flushes.
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
            if tee:
                sys.stdout.write(ln)
                sys.stdout.flush()
            fh.write(ln)
        proc.wait()
    _log(logs, f"=== {step}: exit {proc.returncode}")
    return proc.returncode


def _respawn_detached(logs: Path, argv: list[str]) -> int:
    _rotate(logs)
    fh = (logs / "pipeline.log").open("a", encoding="utf-8")
    child = [sys.executable, str(Path(__file__).resolve()), *[a for a in argv if a != "--detach"]]
    kw: dict = {"cwd": str(REPO), "stdout": fh, "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL, "close_fds": True}
    if IS_WIN:
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP survives the launching shell
        # exiting, but NOT the Job Object Windows' OpenSSH server wraps an SSH
        # session in — that job kills every process in it (detached or not) when
        # the session's job winds down, with no crash logged anywhere.
        # CREATE_BREAKAWAY_FROM_JOB opts out; some job policies refuse breakaway
        # outright (CreateProcess then raises) -- fall back rather than fail.
        base_flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
        try:
            kw["creationflags"] = base_flags | 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
            proc = subprocess.Popen(child, **kw)
        except OSError:
            kw["creationflags"] = base_flags
            proc = subprocess.Popen(child, **kw)
    else:
        kw["start_new_session"] = True
        proc = subprocess.Popen(child, **kw)
    (logs / "pipeline.pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"detached pid {proc.pid}")
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
        if other and other != os.getpid() and _pid_alive(other):
            print(f"a pipeline is already running here (pid {other}). Watch it, or "
                  f"{'taskkill /F /PID' if IS_WIN else 'kill'} {other}.")
            return 1

    if args.detach:
        return _respawn_detached(logs, argv)

    _rotate(logs)
    pidf.write_text(str(os.getpid()), encoding="utf-8")
    exp, hf_id, arch, base_name = BASES[args.base]
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

        bench = [py, str(REPO / "scripts" / "run_3090_benchmark.py"),
                 "--dir", str(root), "--experiment", exp,
                 "--base-hf-id", hf_id, "--base-name", base_name, "--arch", arch]
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
