"""One detached command: train a base + benchmark it vs its untuned self.

Chains ``run_3090_experiment.py`` (bootstrap -> train -> merge -> in-domain eval)
and ``run_3090_benchmark.py`` (register -> smoke -> full batch -> leaderboard +
bundle.zip) for one base model, so a whole ~10-16 h experiment is a single
line and one file to fetch:

    python scripts/run_3090_pipeline.py --dir C:\\Users\\User\\a-experiment\\exp --base baseer --detach

``--base`` is ``qwen3vl4b`` | ``baseer`` | ``paligemma``. Re-run the same command
to resume (both sub-steps are individually resumable); the ``full`` benchmark
stage scores another 200 lines/dataset each run until every set is exhausted.

Watch it:  Get-Content <dir>\\logs\\pipeline.log -Wait -Tail 50
Fetch:     <dir>\\bench\\bundle.zip   (curated summaries + leaderboard, no cache junk)
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

# base -> (experiment name, base hf_id, benchmark arch shim, base models.yaml row)
BASES = {
    "qwen3vl4b": ("exp3090-qwen3vl4b", "Qwen/Qwen3-VL-4B-Instruct", "qwen2_5_vl", "qwen3-vl-4b-bf16"),
    "baseer":    ("exp3090-baseer-nakba", "Misraj/Baseer__Nakba", "qwen2_5_vl", "baseer-nakba-bf16"),
    "paligemma": ("exp3090-paligemma2-3b", "google/paligemma2-3b-pt-448", "paligemma", "paligemma2-3b-bf16"),
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


def _run(logs: Path, cmd: list[str], *, step: str) -> int:
    _log(logs, f"=== {step}: {' '.join(cmd)}")
    tee = sys.stdout.isatty()
    with (logs / "pipeline.log").open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env={**os.environ},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for ln in proc.stdout:
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
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
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
