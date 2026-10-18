"""After contest-prompt Baseer finishes, start ViT-unfreeze training.

Ckpt-400 and ckpt-800 official scoring are ditched. Watcher waits until
``nakba-misraj-baseer-contest`` lands in the official table, then launches
``ain7b_nakba_visft``.

    python scripts/handoff_ain_nakba_visft.py --detach
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

from _detach_spawn import emit_log_line, spawn_surviving_logout  # noqa: E402
from eval_ain_nakba_ckpts import (  # noqa: E402
    _gpu_used_mib,
    _pid_alive,
    _pid_is_cmd,
)

OLD_DIR = Path(r"C:\Users\User\a-experiment\suite-v1\ain7b_nakba")
VISFT_DIR = Path(r"C:\Users\User\a-experiment\suite-v1\ain7b_nakba_visft")
TABLE = OLD_DIR / "ckpt_eval" / "official" / "table.csv"
EVAL_PID = OLD_DIR / "logs" / "ckpt_eval.pid"
HANDOFF_LOG = OLD_DIR / "logs" / "handoff_visft.log"
POLL_S = 45
GPU_IDLE_MIB = 1500


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    emit_log_line(HANDOFF_LOG, f"[{_now()}] {msg}")


def _venv_py() -> Path:
    return OLD_DIR / "venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _table_n(model: str) -> int | None:
    if not TABLE.is_file():
        return None
    import csv
    with TABLE.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            if rec.get("model") == model:
                try:
                    return int(rec.get("n") or 0)
                except ValueError:
                    return None
    return None


def _eval_alive() -> bool:
    if not EVAL_PID.is_file():
        return False
    try:
        pid = int(EVAL_PID.read_text(encoding="utf-8").strip() or 0)
    except ValueError:
        return False
    return bool(pid) and _pid_alive(pid) and not _pid_is_cmd(pid)


def _prepare_visft_dir() -> None:
    VISFT_DIR.mkdir(parents=True, exist_ok=True)
    (VISFT_DIR / "logs").mkdir(exist_ok=True)
    for name in ("venv", "hf_cache", "uv_cache"):
        src = OLD_DIR / name
        dest = VISFT_DIR / name
        if dest.exists() or not src.exists():
            continue
        if IS_WIN:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dest), str(src)], check=False)
        else:
            dest.symlink_to(src, target_is_directory=True)
    env_src = OLD_DIR / ".env"
    env_dest = VISFT_DIR / ".env"
    if env_src.is_file() and not env_dest.exists():
        env_dest.write_text(env_src.read_text(encoding="utf-8"), encoding="utf-8")
    _log(f"visft dir ready: {VISFT_DIR}")


def _launch_visft() -> None:
    _prepare_visft_dir()
    py = _venv_py()
    cmd = [
        str(py), str(REPO / "scripts" / "run_3090_pipeline.py"),
        "--dir", str(VISFT_DIR), "--base", "ain7b_nakba_visft", "--detach",
    ]
    _log(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO), check=False)


def run() -> int:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    _log("handoff watching for contest-prompt Baseer, then visft")
    while True:
        n_contest = _table_n("nakba-misraj-baseer-contest")
        if n_contest and n_contest >= 2000:
            used = _gpu_used_mib()
            pipe = _eval_alive()
            _log(f"contest Baseer done n={n_contest} gpu_mib={used} eval_alive={pipe}")
            if not pipe and used is not None and used < GPU_IDLE_MIB:
                _launch_visft()
                _log("visft pipeline detached")
                return 0
        time.sleep(POLL_S)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args(argv)
    pidf = OLD_DIR / "logs" / "handoff_visft.pid"
    pidf.parent.mkdir(parents=True, exist_ok=True)
    if args.detach:
        child = [sys.executable, str(Path(__file__).resolve()),
                 *[a for a in argv if a != "--detach"]]
        pid = spawn_surviving_logout(
            child, cwd=REPO, log_path=HANDOFF_LOG, pid_path=pidf,
        )
        print(f"detached pid {pid}")
        print(f"log : {HANDOFF_LOG}")
        return 0
    pidf.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return run()
    finally:
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
