"""Run a job on a Colab GPU through the Colab CLI without losing results or overspending. For ALL our Colab work.

    python3 scripts/colab_job.py SPEC.json [--gpu T4] [--dry-run]       start a job from a spec
    python3 scripts/colab_job.py --adopt SESSION SPEC.json              re-attach to a running session: sync, keep watching, stop
    python3 scripts/colab_job.py --reap                                 stop every Colab session and verify

What it guarantees
  * The job runs DETACHED on the VM (nohup + `timeout`), so a dropped connection cannot kill it or hang the supervisor.
  * The supervisor polls with short, retried calls. Logs, checkpoints and results are synced to the SSD every `sync_every_s` seconds and at
    the end, so a dead session loses at most one interval of work (the trainers also checkpoint every epoch and every 200 steps).
  * It interrupts the job (final sync, then stop) when ANY of these hits: the job finishes; compute units spent by this job reach `max_units`;
    the global benchmark budget is used up (results/rasam/colab_bench/budget.json); `max_minutes` wall clock passes; Ctrl-C / SIGTERM.
  * The session is ALWAYS stopped at the end and the stop is verified with `colab usage` (0 active assignments). If it cannot be verified the
    last line tells you the exact command to run by hand.
  * One session runs a whole list of commands, so setup (download, prep) is paid once.

SPEC (JSON)
  name, gpu (T4|L4|G4|A100|H100), max_units, max_minutes, sync_every_s, poll_s,
  uploads: [[local path relative to the repo, remote path], ...],
  setup: [shell commands run first on the VM],
  runs: [{"label": ..., "cmd": ...}, ...]   run one after another; a failing run does not stop the next one,
  artifacts: [remote paths to sync (directories ok; *.pkl and hf caches are excluded)],
  local_out: directory under the repo that receives the synced files and job.log
Everything the VM prints between STAMP lines is timed on the VM clock, so network jitter does not pollute step timings.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The supervisor's own state (snippets, logs, synced results, budget) lives on the INTERNAL disk: the SSD dropped out twice on 2026-10-24/25 and a
# supervisor that cannot write cannot stop a billing session. The files are small (tens of MB); finished results are copied to the SSD (copy_to_repo).
STAGING = Path.home() / "colab_staging"
BUDGET_FILE = STAGING / "budget.json"
_REPO_BUDGET = REPO / "results/rasam/colab_bench/budget.json"
if not BUDGET_FILE.exists() and _REPO_BUDGET.exists():
    STAGING.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(_REPO_BUDGET.read_text())


def copy_to_repo(src: Path, dst: Path) -> bool:
    """Copy a finished job directory to the SSD results folder if the SSD is mounted; True when copied (staging copy is kept either way)."""
    import shutil

    try:
        if not REPO.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return True
    except OSError as e:
        print("copy to the SSD failed (staging copy kept):", e, flush=True)
        return False


def sh(cmd: list[str], timeout: int) -> tuple[subprocess.CompletedProcess, float]:
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:  # a hung CLI call must not hang the supervisor
        p = subprocess.CompletedProcess(cmd, 124, e.stdout or "", "timeout")
    return p, time.time() - t


def robust(cmd: list[str], timeout: int, tries: int = 3, ok=lambda p: p.returncode == 0):
    last, total = None, 0.0
    for i in range(tries):
        p, dt = sh(cmd, timeout)
        total += dt
        last = p
        if ok(p):
            return p, total
        time.sleep(8 * (i + 1))
    return last, total


def balance() -> tuple[float | None, str]:
    p, _ = sh(["colab", "usage"], 90)
    txt = p.stdout + p.stderr
    m = re.search(r"Current balance:\s*([\d.]+)", txt)
    return (float(m.group(1)) if m else None), txt


def remote_script(spec: dict) -> str:
    lines = ["#!/bin/bash", "cd /content", "mkdir -p /content/run", 'stamp(){ echo "STAMP $1 $(date +%s)"; }', "stamp start", "nvidia-smi -L"]
    lines += ["stamp setup_start"] + list(spec.get("setup", [])) + ["echo \"RC setup $?\"", "stamp setup_end"]
    for i, r in enumerate(spec["runs"]):
        lab = re.sub(r"\W+", "_", r.get("label", f"run{i}"))
        lines += [f"stamp {lab}_start", r["cmd"], f'echo "RC {lab} $?"', f"stamp {lab}_end"]
    lines += ["echo ALLDONE"]
    return "\n".join(lines) + "\n"


def start_snippet(script: str, max_s: int) -> str:
    return (
        "import subprocess, os\n"
        "os.makedirs('/content/run', exist_ok=True)\n"
        f"open('/content/run/job.sh', 'w').write({script!r})\n"
        "subprocess.Popen(['bash', '-c', 'nohup timeout %d bash -c \"bash /content/run/job.sh > /content/run/job.log 2>&1; echo $? > /content/run/exit_code\" "
        f"> /dev/null 2>&1 &' % {max_s}], start_new_session=True)\n"
        "print('STARTED')\n"
    )


STATUS = (
    "import os, subprocess\n"
    "ec = open('/content/run/exit_code').read().strip() if os.path.exists('/content/run/exit_code') else 'none'\n"
    "print('EXIT=' + ec)\n"
    "log = open('/content/run/job.log', errors='replace').read().splitlines() if os.path.exists('/content/run/job.log') else []\n"
    "keep = [l for l in log if l.startswith(('STAMP', 'RC ', 'epoch', 'params', 'train ', 'ALLDONE', 'TIMING', 'Traceback')) or 'Error' in l]\n"
    "print('\\n'.join(keep[-14:]))\n"
    "print(subprocess.run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip())\n"
)


def sync_snippet(artifacts: list[str]) -> str:
    paths = " ".join(a.lstrip("/") for a in (["/content/run"] + artifacts))
    return (
        "import subprocess\n"
        f"print(subprocess.run('tar czf /content/run_sync.tgz -C / --exclude=*.pkl --exclude=hfcache --exclude=*.parquet {paths} 2>/dev/null; ls -l /content/run_sync.tgz', "
        "shell=True, capture_output=True, text=True).stdout)\n"
    )


class Job:
    def __init__(self, spec: dict, name: str, out: Path, start_balance: float, floor: float) -> None:
        self.spec, self.name, self.out = spec, name, out
        self.start_balance, self.floor = start_balance, floor
        self.reason = "unknown"
        self.last_sync = 0.0
        self.syncs = 0
        self.t0 = time.time()
        out.mkdir(parents=True, exist_ok=True)

    def kexec(self, code: str, timeout: int = 120, tries: int = 3):
        f = self.out / "_snippet.py"
        f.write_text(code)
        return robust(["colab", "exec", "-s", self.name, "-f", str(f)], timeout, tries)

    def sync(self) -> bool:
        p, _ = self.kexec(sync_snippet(self.spec.get("artifacts", [])), 180)
        if p.returncode != 0:
            return False
        tgz = self.out / "sync.tgz"
        p, _ = robust(["colab", "download", "-s", self.name, "/content/run_sync.tgz", str(tgz)], 600, tries=2)
        if p.returncode != 0 or not tgz.exists():
            return False
        try:
            with tarfile.open(tgz) as t:
                t.extractall(self.out, filter="data")
        except Exception as e:  # a half-downloaded archive is skipped, the next sync replaces it
            print("sync extract failed:", e, flush=True)
            return False
        self.last_sync = time.time()
        self.syncs += 1
        return True

    def stop(self) -> bool:
        for i in range(12):
            if i % 2 == 0:
                sh(["colab", "stop", "-s", self.name], 180)
            _, txt = balance()
            if "Active assignments: 0" in txt:
                return True
            time.sleep(15)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--gpu")
    ap.add_argument("--adopt", metavar="SESSION")
    ap.add_argument("--reap", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.reap:
        p, _ = sh(["colab", "sessions"], 120)
        names = re.findall(r"(bench-[\w.\-]+|job-[\w.\-]+)", p.stdout + p.stderr)
        for n in sorted(set(names)):
            sh(["colab", "stop", "-s", n], 180)
        _, txt = balance()
        print(txt.strip())
        print("reaped" if "Active assignments: 0" in txt else "!!! still active assignments, check `colab sessions`")
        return
    if not a.spec:
        ap.error("spec required")
    spec = json.loads(Path(a.spec).read_text())
    if a.gpu:
        spec["gpu"] = a.gpu
    spec.setdefault("poll_s", 60)
    spec.setdefault("sync_every_s", 300)
    spec.setdefault("max_units", 2.0)
    spec.setdefault("max_minutes", 30)
    max_s = int(spec["max_minutes"] * 60)
    script = remote_script(spec)
    if a.dry_run:
        print(script)
        print(start_snippet(script, max_s)[:600])
        print(sync_snippet(spec.get("artifacts", [])))
        return

    bal, txt = balance()
    bal2, _ = balance()
    if bal is None or bal2 is None:
        print("Colab API unhealthy (no balance twice); not starting", flush=True)
        sys.exit(75)  # EX_TEMPFAIL
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if BUDGET_FILE.exists():
        meta = json.loads(BUDGET_FILE.read_text())
    else:
        meta = {"baseline_balance": bal, "budget_units": 10.0}
        BUDGET_FILE.write_text(json.dumps(meta))
    floor = meta["baseline_balance"] - meta["budget_units"]
    left = bal - floor
    print(f"balance {bal:.2f}; global floor {floor:.2f}; {left:.2f} units left in the overall budget; this job may spend {spec['max_units']:.2f}", flush=True)
    if left < min(spec["max_units"], 1.0):
        print("overall budget used up; not starting")
        sys.exit(1)
    name = a.adopt or f"job-{spec['name']}-{spec['gpu'].lower()}-{time.strftime('%m%d%H%M')}"
    out = STAGING / "jobs" / name
    job = Job(spec, name, out, bal, floor)

    def on_signal(signum, frame):  # Ctrl-C / SIGTERM: fall through to the finally block (sync + verified stop)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    created = bool(a.adopt)
    fails = 0
    try:
        if not a.adopt:
            p, dt = robust(["colab", "new", "-s", name] + ([] if spec["gpu"].upper() == "CPU" else ["--gpu", spec["gpu"]]), 600)
            print(f"colab new: rc={p.returncode} {dt:.0f}s", flush=True)
            if p.returncode != 0:
                job.reason = "could not create session"
                return
            created = True
            for src, dst in spec.get("uploads", []):
                p, _ = robust(["colab", "upload", "-s", name, str(REPO / src), dst], 600)
                if p.returncode != 0:
                    job.reason = f"upload failed: {src}"
                    return
            p, _ = job.kexec(start_snippet(script, max_s))
            if "STARTED" not in (p.stdout + p.stderr):
                job.reason = "job did not start"
                return
            print("job started on the VM (detached)", flush=True)
        while True:
            time.sleep(spec["poll_s"])
            b, _ = balance()
            p, _ = job.kexec(STATUS, 90, tries=1)
            out_txt = p.stdout + p.stderr
            if b is None or "EXIT=" not in out_txt:
                fails += 1
            else:
                fails = 0
            spent = (job.start_balance - b) if b is not None else float("nan")
            m = re.search(r"EXIT=(\S+)", out_txt)
            print(f"[{(time.time() - job.t0) / 60:5.1f} min] balance {b} spent {spent:.2f} | " + " | ".join(out_txt.strip().splitlines()[-3:])[:230], flush=True)
            if m and m.group(1) != "none":
                job.reason = f"job finished (exit code {m.group(1)})"
                break
            if b is not None and b <= job.floor:
                job.reason = "overall budget floor reached"
                break
            if b is not None and spent >= spec["max_units"]:
                job.reason = f"job budget of {spec['max_units']} units reached"
                break
            if time.time() - job.t0 > max_s:
                job.reason = f"max_minutes ({spec['max_minutes']}) reached"
                break
            if fails >= 8:
                job.reason = "lost contact 8 polls in a row"
                break
            if time.time() - job.last_sync > spec["sync_every_s"] and m and m.group(1) == "none":
                print("  sync:", "ok" if job.sync() else "failed", flush=True)
    except KeyboardInterrupt:
        job.reason = "interrupted"
    finally:
        print(f"ending: {job.reason}", flush=True)
        if created:
            ok = job.sync()
            if not ok:
                ok = job.sync()
            print("final sync:", "ok" if ok else "FAILED (last periodic sync is on disk)", flush=True)
            verified = job.stop()
            print("stop verified:", verified, flush=True)
            if not verified:
                print(f"!!! SESSION MAY STILL BE RUNNING AND SPENDING UNITS: run `colab stop -s {name}`, then `colab usage`", flush=True)
        b, _ = balance()
        used = round(job.start_balance - b, 2) if b is not None else None
        rec = {"name": name, "reason": job.reason, "units_used": used, "minutes": round((time.time() - job.t0) / 60, 1), "syncs": job.syncs,
               "start_balance": job.start_balance, "end_balance": b, "spec": spec}
        (out / "job_record.json").write_text(json.dumps(rec, indent=1))
        dst = REPO / spec.get("local_out", "results/rasam/colab_runs") / name
        copied = copy_to_repo(out, dst)
        print(f"units used by this job: {used}; balance now {b}; results in {out}" + (f" and copied to {dst}" if copied else " (SSD not available: run copy later)"), flush=True)


if __name__ == "__main__":
    main()
