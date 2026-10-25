"""Time the CRNN trainer on one Colab accelerator via the Colab CLI (colab-cli 0.7.x). One accelerator per invocation.

    ~/.venvs/kraken-rasam/bin/python scripts/colab_bench.py --gpu T4 [--epochs 3] [--width 0.5] [--bs 16]

Steps, each timed: usage before, `colab new`, upload 3 scripts, data download + prep on the VM, training, usage after, `colab stop`
(the session is ALWAYS stopped, also on error or Ctrl-C, so units are not burned idle). Writes results/rasam/colab_bench/<name>.json with
the wall time of every step, the per-epoch seconds the trainer printed, the GPU name and the raw `colab usage` text before and after.
The Mac CPU reference for the same flags is about 546 s per epoch (A1, half width, RASAM-1 train, 5,860 lines).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd: list[str], timeout: int = 3600) -> tuple[subprocess.CompletedProcess, float]:
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p, time.time() - t


def robust(cmd: list[str], timeout: int, tries: int = 4, ok=lambda p: p.returncode == 0):
    """Run a colab command, retrying on transient network errors (the Mac's link to Google was flaky on 2026-10-24)."""
    last, total = None, 0.0
    for i in range(tries):
        p, dt = sh(cmd, timeout)
        total += dt
        last = p
        if ok(p):
            return p, total
        time.sleep(10 * (i + 1))
    return last, total


def stop_and_verify(name: str) -> bool:
    """Stop the session, then confirm with `colab usage` that nothing is assigned. An idle VM keeps spending units, so keep trying."""
    for i in range(12):
        if i % 2 == 0:
            sh(["colab", "stop", "-s", name], 180)
        p, _ = sh(["colab", "usage"], 120)
        if "Active assignments: 0" in (p.stdout + p.stderr):
            return True
        time.sleep(15)
    return False


BUDGET_FILE = HERE.parent / "results/rasam/colab_bench/budget.json"


def balance(text: str) -> float | None:
    m = re.search(r"Current balance:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=10.0, help="total compute units all benchmark runs together may spend (from the baseline balance)")
    ap.add_argument("--max-run-units", type=float, default=3.0, help="refuse to start if less than this is left in the budget")
    ap.add_argument("--train-timeout-s", type=int, default=1500, help="kill the training step after this many seconds")
    ap.add_argument("--gpu", required=True, choices=["T4", "L4", "G4", "A100", "H100"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--width", default="0.5")
    ap.add_argument("--bs", default="16")
    ap.add_argument("--high-mem", action="store_true")
    a = ap.parse_args()
    name = f"bench-{a.gpu.lower()}-w{a.width}-bs{a.bs}-{time.strftime('%m%d%H%M')}"
    rec: dict = {"name": name, "gpu_requested": a.gpu, "epochs": a.epochs, "width": a.width, "bs": a.bs, "steps": {}}
    t_start = time.time()
    p, _ = sh(["colab", "usage"], 120)
    rec["usage_before"] = (p.stdout + p.stderr).strip()
    print("usage before:\n" + rec["usage_before"], flush=True)
    bal = balance(rec["usage_before"])
    p2, _ = sh(["colab", "usage"], 120)  # health check: two clean readings in a row before any session is created
    if bal is None or balance(p2.stdout + p2.stderr) is None:
        print("could not read the compute-unit balance twice from `colab usage` (network to the Colab API unhealthy); not starting", flush=True)
        raise SystemExit(75)  # EX_TEMPFAIL: the wrapper retries later
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if BUDGET_FILE.exists():
        base = json.loads(BUDGET_FILE.read_text())["baseline_balance"]
    else:
        base = bal
        BUDGET_FILE.write_text(json.dumps({"baseline_balance": base, "budget_units": a.budget}))
    spent, left = base - bal, a.budget - (base - bal)
    print(f"budget: baseline {base:.2f}, balance {bal:.2f}, spent {spent:.2f} of {a.budget:.2f}, left {left:.2f}", flush=True)
    if left < a.max_run_units:
        raise SystemExit(f"only {left:.2f} units left in the {a.budget:.0f}-unit budget (< {a.max_run_units}); not starting")
    created = False
    try:
        cmd = ["colab", "new", "-s", name, "--gpu", a.gpu] + (["--high-mem"] if a.high_mem else [])
        p, dt = sh(cmd, 900)
        rec["steps"]["new_s"] = round(dt, 1)
        print(f"colab new: rc={p.returncode} {dt:.0f}s\n{(p.stdout + p.stderr).strip()[-400:]}", flush=True)
        if p.returncode != 0:
            raise RuntimeError("could not create the session")
        created = True
        t = time.time()
        for f, dst in (("xai_train_aligner.py", "/content/xai_train_aligner.py"), ("rasam_prepare_crnn_data.py", "/content/rasam_prepare_crnn_data.py")):
            p, _ = robust(["colab", "upload", "-s", name, str(HERE / f), dst], 600)
            if p.returncode != 0:
                raise RuntimeError("upload failed: " + (p.stdout + p.stderr)[-300:])
        rec["steps"]["upload_s"] = round(time.time() - t, 1)
        p, dt = robust(["colab", "exec", "-s", name, "-f", str(HERE / "colab_remote_prepare.py")], 2400,
                       ok=lambda q: q.returncode == 0 and "TIMING" in (q.stdout + q.stderr))
        rec["steps"]["prepare_exec_s"] = round(dt, 1)
        rec["prepare_out"] = (p.stdout + p.stderr).strip()[-1500:]
        print("prepare:", rec["prepare_out"][-300:], flush=True)
        if p.returncode != 0 or "TIMING" not in rec["prepare_out"]:
            raise RuntimeError("data prep failed")
        script = (HERE / "colab_remote_train.py").read_text().replace("@EPOCHS@", str(a.epochs)).replace("@WIDTH@", a.width).replace("@BS@", a.bs)
        tmp = HERE.parent / "results/rasam/colab_bench" / f"{name}_train.py"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(script)
        p, dt = sh(["colab", "exec", "-s", name, "-f", str(tmp)], a.train_timeout_s)
        rec["steps"]["train_exec_s"] = round(dt, 1)
        out = (p.stdout + p.stderr).strip()
        rec["train_out"] = out[-3000:]
        print(out[-1200:], flush=True)
        rec["epoch_s"] = [int(m) for m in re.findall(r"^epoch \d+/\d+\s+loss\s+[\d.]+\s+(\d+)s", out, re.M)]
        m = re.search(r"^GPU (.+?) \|", out, re.M)
        rec["gpu_name"] = m.group(1) if m else None
        tm = re.search(r"download_s=([\d.]+) prep_s=([\d.]+)", rec["prepare_out"])
        if tm:
            rec["steps"]["hf_download_s"], rec["steps"]["prep_s"] = float(tm.group(1)), float(tm.group(2))
    finally:
        if created:
            t = time.time()
            rec["stopped_verified"] = stop_and_verify(name)
            rec["steps"]["stop_s"] = round(time.time() - t, 1)
            print(f"colab stop verified: {rec['stopped_verified']}", flush=True)
            if not rec["stopped_verified"]:
                print(f"!!! SESSION MAY STILL BE RUNNING AND SPENDING UNITS: run `colab stop -s {name}` and check `colab usage`", flush=True)
        p, _ = sh(["colab", "usage"], 120)
        rec["usage_after"] = (p.stdout + p.stderr).strip()
        print("usage after:\n" + rec["usage_after"], flush=True)
        b1 = balance(rec["usage_after"])
        rec["units_used"] = round(bal - b1, 2) if b1 is not None else None
        print(f"units used by this run: {rec['units_used']}", flush=True)
        rec["total_wall_s"] = round(time.time() - t_start, 1)
        out_dir = HERE.parent / "results/rasam/colab_bench"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1))
        if rec.get("epoch_s"):
            e = rec["epoch_s"]
            print(f"\n{a.gpu}: epoch seconds {e}; steady-state (epochs 2+) mean {sum(e[1:]) / max(1, len(e) - 1):.1f}s; Mac CPU ~546s -> speed-up x{546 / max(1, sum(e[1:]) / max(1, len(e) - 1)):.1f}")


if __name__ == "__main__":
    main()
