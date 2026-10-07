"""Staged ViT-unfreeze: train 400 → official score → continue only if CER improves.

Baseline is frozen AIN NAKBA ckpt-1600 corpus CER-raw (0.1078). Each chunk must
beat the best CER so far (strictly lower) or the loop stops.

    python scripts/stage_ain_nakba_visft.py --detach
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

from _detach_spawn import (  # noqa: E402
    emit_log_line, spawn_surviving_logout, stdout_is_regular_file,
)

VISFT_DIR = Path(r"C:\Users\User\a-experiment\suite-v1\ain7b_nakba_visft")
OLD_DIR = Path(r"C:\Users\User\a-experiment\suite-v1\ain7b_nakba")
# Frozen-ViT best on official nakba_blind (corpus CER-raw).
BASELINE_CER = 0.1078
CHUNK = 400
# Full-data epoch is ~1996 steps with val_fraction=0; last chunk ends there.
TARGETS = (400, 800, 1200, 1600, 1996)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    emit_log_line(VISFT_DIR / "logs" / "stage.log", f"[{_now()}] {msg}")


def _venv_py() -> Path:
    return VISFT_DIR / "venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def should_continue(new_cer: float, best_cer: float) -> bool:
    """Continue only when the new checkpoint strictly beats the best so far."""
    return new_cer < best_cer


def _read_cer(table: Path, model: str) -> float | None:
    if not table.is_file():
        return None
    with table.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            if rec.get("model") != model:
                continue
            try:
                return float(rec["cer_micro_raw"])
            except (TypeError, ValueError, KeyError):
                return None
    return None


def _set_max_steps(steps: int) -> Path:
    src = REPO / "configs" / "training" / "ain7b_nakba_visft.yaml"
    import yaml

    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(VISFT_DIR / "runs" / "ain7b-nakba-visft")
    cfg["base_model"]["hf_id"] = r"C:\Users\User\a-experiment\suite-v1\ain7b\merged\suite-v1-ain7b"
    cfg["train"]["max_steps"] = int(steps)
    cfg["train"]["save_steps"] = CHUNK
    cfg["train"]["load_best_model_at_end"] = False
    cfg["data"]["val_fraction"] = 0.0
    dest = VISFT_DIR / "effective_config.yaml"
    dest.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dest


def _run(cmd: list[str]) -> int:
    _log(f"$ {' '.join(cmd)}")
    env = {**os.environ}
    env.pop("PYTHONHOME", None)
    env["VIRTUAL_ENV"] = str(VISFT_DIR / "venv")
    env["HF_HOME"] = str(VISFT_DIR / "hf_cache")
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    log = VISFT_DIR / "logs" / "stage.log"
    # When detached, stdout is already >> stage.log — do not open it again.
    tee = not stdout_is_regular_file()
    fh = log.open("a", encoding="utf-8") if tee else None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for ln in proc.stdout:
            sys.stdout.write(ln)
            sys.stdout.flush()
            if fh is not None:
                fh.write(ln)
        return proc.wait()
    finally:
        if fh is not None:
            fh.close()


def _train_to(steps: int) -> int:
    cfg = _set_max_steps(steps)
    py = _venv_py()
    return _run([str(py), "-m", "arabic_ocr_bench.training.cli", "sft", "--config", str(cfg)])


def _score_step(step: int) -> float | None:
    py = _venv_py()
    rc = _run([
        str(py), str(REPO / "scripts" / "eval_ain_nakba_ckpts.py"),
        "--dir", str(VISFT_DIR), "--base", "ain7b_nakba_visft",
        "--official", "--no-wait", "--only-steps", str(step),
    ])
    if rc:
        _log(f"score step {step} failed rc={rc}")
        return None
    model = f"ain7b-nakba-visft-ckpt-{step}"
    return _read_cer(VISFT_DIR / "ckpt_eval" / "official" / "table.csv", model)


def run() -> int:
    (VISFT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    decision = VISFT_DIR / "logs" / "stage_decisions.jsonl"
    best = BASELINE_CER
    _log(f"staged visft start baseline_cer={best:.4f} targets={list(TARGETS)}")

    for target in TARGETS:
        _log(f"=== train to step {target} ===")
        rc = _train_to(target)
        if rc:
            _log(f"train to {target} failed rc={rc}")
            return rc
        _log(f"=== official score ckpt-{target} ===")
        cer = _score_step(target)
        if cer is None:
            _log(f"no CER for ckpt-{target}; stopping")
            return 1
        improved = should_continue(cer, best)
        rec = {
            "at": _now(),
            "step": target,
            "cer_micro_raw": cer,
            "best_before": best,
            "continue": improved,
        }
        with decision.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        _log(
            f"ckpt-{target} CER-raw={cer:.4f} best_before={best:.4f} "
            f"{'CONTINUE' if improved else 'STOP (no improvement)'}"
        )
        if not improved:
            return 0
        best = cer

    _log(f"staged visft complete best_cer={best:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args(argv)
    VISFT_DIR.mkdir(parents=True, exist_ok=True)
    (VISFT_DIR / "logs").mkdir(exist_ok=True)
    pidf = VISFT_DIR / "logs" / "stage.pid"
    if args.detach:
        child = [sys.executable, str(Path(__file__).resolve()),
                 *[a for a in argv if a != "--detach"]]
        pid = spawn_surviving_logout(
            child, cwd=REPO, log_path=VISFT_DIR / "logs" / "stage.log", pid_path=pidf,
        )
        print(f"detached pid {pid}")
        print(f"log : {VISFT_DIR / 'logs' / 'stage.log'}")
        return 0
    pidf.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return run()
    finally:
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
