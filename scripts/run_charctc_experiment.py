"""Character-CTC CPU experiment (Windows-first, also POSIX).

Tokenizer audit + closed-charset CRNN-CTC trained on the same 8k Muharaf
train lines as exp3090. Inference and training are pinned to CPU so the
suite_v1 VLM job keeps the 3090.

    python scripts/run_charctc_experiment.py --dir C:\\Users\\User\\a-experiment\\charctc --detach
    python scripts/run_charctc_experiment.py --dir C:\\Users\\User\\a-experiment\\charctc status

Phases (idempotent, tracked in ``<dir>/state.json``):

    install -> check -> prepare -> charset -> audit -> train -> smoke -> full -> report

Reuses the suite_v1 venv (torch + transformers already there) and its HF
cache. Does **not** pip-install into that venv. Everything written lives
under ``<dir>/``. Same detach / status / resume ergonomics as
``run_paddleocr_experiment.py``.
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
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

from _detach_spawn import emit_log_line, spawn_surviving_logout, stdout_is_regular_file

PHASES = [
    "install",
    "check",
    "prepare",
    "charset",
    "audit",
    "train",
    "smoke",
    "full",
    "report",
]
CONFIG = "configs/training/charctc/cpu.yaml"
EVAL_DATASETS = (
    "muharaf_lines",
    "kitab_khatt",
    "nakba_lines",
    "rasam1",
    "rasam2",
)
# Qwen3-VL-4B LoRA on the same 8k Muharaf lines (findings-and-notes.md, 2026-09-09).
# In-domain muharaf_lines was 36%; the four held-out sets are the headline.
QWEN_LORA_CER_LIGHT = {
    "muharaf_lines": 0.36,
    "kitab_khatt": 0.30,
    "nakba_lines": 0.29,
    "rasam1": 0.52,
    "rasam2": 0.49,
}
TROCR_MUHARAF_CER = 0.80

DEFAULT_VENVS_WIN = (
    Path(r"C:\Users\User\a-experiment\suite-v1\_shared\venv"),
    Path(r"C:\Users\User\a-experiment\suite-v1\qwen3vl4b_suite\venv"),
    Path(r"C:\Users\User\a-experiment\suite-v1\ain7b\venv"),
    Path(r"C:\Users\User\a-experiment\exp\venv"),
)
DEFAULT_HF_WIN = (
    Path(r"C:\Users\User\a-experiment\suite-v1\data_cache\hf"),
    Path(r"C:\Users\User\a-experiment\suite-v1\_shared\hf_cache"),
    Path(r"C:\Users\User\a-experiment\suite-v1\qwen3vl4b_suite\hf_cache"),
)
DEFAULT_ENV_WIN = Path(r"C:\Users\User\a-experiment\suite-v1\experiment.env")

_CHILD: subprocess.Popen | None = None
HEARTBEAT_S = 90
LOG_NAME = "charctc.log"
PID_NAME = "charctc.pid"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dirs(root: Path) -> dict:
    d = {
        "root": root,
        "logs": root / "logs",
        "cache": root / "cache",
        "runs": root / "runs",
        "audit": root / "audit",
        "bench": root / "bench",
        "hf_cache": root / "hf_cache",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _log(dirs: dict, msg: str) -> None:
    emit_log_line(dirs["logs"] / LOG_NAME, f"[{_now()}] {msg}")


def _rotate_log(dirs: dict) -> None:
    log = dirs["logs"] / LOG_NAME
    if log.exists() and log.stat().st_size > 0:
        stamp = _now().replace(":", "").replace("-", "")
        log.rename(dirs["logs"] / f"charctc.{stamp}.log")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def pin_cpu_env() -> None:
    """Keep torch off a GPU another process already owns."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["AOCR_DEVICE"] = "cpu"
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")


def _pin_env(dirs: dict, *, venv: Path, hf_home: Path) -> None:
    pin_cpu_env()
    src = str(REPO / "src")
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["VIRTUAL_ENV"] = str(venv)
    if IS_WIN and DEFAULT_ENV_WIN.exists():
        _load_env_file(DEFAULT_ENV_WIN)
    _load_env_file(dirs["root"] / "experiment.env")
    bin_dir = venv / ("Scripts" if IS_WIN else "bin")
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _state_path(dirs: dict) -> Path:
    return dirs["root"] / "state.json"


def _read_state(dirs: dict) -> dict:
    p = _state_path(dirs)
    if not p.exists():
        return {"phases": {}, "created": _now()}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"phases": {}, "created": _now()}


def _write_state(dirs: dict, state: dict) -> None:
    _state_path(dirs).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _mark(dirs: dict, state: dict, phase: str, **extra) -> None:
    state.setdefault("phases", {})[phase] = {"done": True, "at": _now(), **extra}
    _write_state(dirs, state)


def _done(state: dict, phase: str) -> bool:
    return bool(state.get("phases", {}).get(phase, {}).get("done"))


def train_needs_more_epochs(completed, wanted) -> bool:
    if completed is None:
        return True
    return int(completed) < int(wanted)


def _yaml_train_epochs(path: Path) -> int:
    in_train = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("train:"):
            in_train = True
            continue
        if in_train and stripped.endswith(":") and raw[:1] not in (" ", "\t"):
            in_train = False
        if in_train and stripped.startswith("epochs:"):
            return int(round(float(stripped.split(":", 1)[1].strip())))
    return 1


def smoke_still_collapsed(dirs: dict) -> bool:
    """True only when every smoke summary exists and CER-light is still ~1.0."""
    for ds in EVAL_DATASETS:
        p = dirs["bench"] / "smoke" / ds / "summary.json"
        if not p.exists():
            return False
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        cer = rec.get("cer_micro_light")
        if not isinstance(cer, (int, float)) or cer < 0.99:
            return False
    return True


def invalidate_eval_state(dirs: dict, state: dict) -> None:
    """Drop stale empty-hyp eval after a new train pass."""
    phases = state.setdefault("phases", {})
    drop = [
        k
        for k in list(phases)
        if k in ("smoke", "full", "report") or k.startswith("smoke:") or k.startswith("full:")
    ]
    for k in drop:
        phases.pop(k, None)
    _write_state(dirs, state)
    bench = dirs["bench"]
    if bench.exists():
        shutil.rmtree(bench, ignore_errors=True)
    bench.mkdir(parents=True, exist_ok=True)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _run(dirs: dict, cmd: list[str], *, phase: str, cwd: Path | None = None) -> int:
    global _CHILD
    _log(dirs, f"$ {' '.join(str(c) for c in cmd)}")
    tee = not stdout_is_regular_file()
    fh = (dirs["logs"] / LOG_NAME).open("a", encoding="utf-8") if tee else None
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd or REPO),
            env={**os.environ},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
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
                    _log(
                        dirs,
                        f"...  {phase}: still running (pid {proc.pid}, "
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
                    fh.flush()
            proc.wait()
        finally:
            _CHILD = None
    finally:
        if fh is not None:
            fh.close()
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


def _py(spec: dict) -> str:
    return str(_venv_python(spec["venv"]))


def _mod(spec: dict, module: str) -> list[str]:
    return [_py(spec), "-m", module]


def phase_install(dirs: dict, state: dict, spec: dict, args) -> None:
    py = _venv_python(spec["venv"])
    if not py.exists():
        raise SystemExit(
            f"install: venv python missing at {py}. Pass --venv pointing at "
            "the suite_v1 venv (torch+transformers). This experiment will not "
            "pip-install into a venv the GPU job is using."
        )
    _log(dirs, f"install: using venv python {py}")
    _log(dirs, f"install: PYTHONPATH={os.environ.get('PYTHONPATH')}")
    _log(dirs, f"install: HF_HOME={os.environ.get('HF_HOME')}")
    _log(dirs, "install: not pip-installing (reuse suite venv, overlay repo via PYTHONPATH)")
    _mark(dirs, state, "install", venv=str(spec["venv"]))


def phase_check(dirs: dict, state: dict, spec: dict, args) -> None:
    code = r"""
import os, sys, platform
print("python", sys.version.replace("\n"," "))
print("executable", sys.executable)
print("platform", platform.platform())
print("CUDA_VISIBLE_DEVICES", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("AOCR_DEVICE", os.environ.get("AOCR_DEVICE"))
print("HF_HOME", os.environ.get("HF_HOME"))
print("HF_TOKEN", "yes" if os.environ.get("HF_TOKEN") else "MISSING")
print("ncpus", os.cpu_count())
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", os.environ.get("CUDA_VISIBLE_DEVICES")
assert os.environ.get("AOCR_DEVICE") == "cpu"
import torch
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
assert not torch.cuda.is_available(), "CPU job must not see the 3090"
from arabic_ocr_bench.charctc.charset import build_charset
from arabic_ocr_bench.charctc.model import CharCRNN
from arabic_ocr_bench.charctc.decode import ctc_greedy
cs = build_charset(["بسم الله"])
m = CharCRNN(cs.n_classes)
n = sum(p.numel() for p in m.parameters())
print("charset_probe", cs.n_classes, "params", n)
print("check ok")
"""
    if _run(dirs, [_py(spec), "-c", code], phase="check"):
        raise SystemExit("check: venv sanity import failed")
    _mark(dirs, state, "check")


def phase_prepare(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "prepare") and not args.force:
        _log(dirs, "prepare: already done")
        return
    cmd = _mod(spec, "arabic_ocr_bench.charctc.prepare") + [
        "--config",
        str(REPO / CONFIG),
        "--out",
        str(dirs["cache"]),
    ]
    if args.limit:
        _log(dirs, f"prepare: note --limit {args.limit} is applied at train time; cache still uses yaml limit")
    if _run(dirs, cmd, phase="prepare"):
        raise SystemExit("prepare: failed")
    man = dirs["cache"] / "manifest.json"
    extra = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
    _mark(dirs, state, "prepare", **extra)


def phase_charset(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "charset") and not args.force:
        _log(dirs, "charset: already done")
        return
    dest = dirs["runs"] / "charset.json"
    code = (
        "from pathlib import Path\n"
        "from arabic_ocr_bench.charctc.prepare import write_charset\n"
        f"cs = write_charset(Path({str(dirs['cache'])!r}), Path({str(dest)!r}))\n"
        "print('n_classes', cs.n_classes)\n"
    )
    if _run(dirs, [_py(spec), "-c", code], phase="charset"):
        raise SystemExit("charset: failed")
    n = None
    if dest.exists():
        n = json.loads(dest.read_text(encoding="utf-8")).get("n_classes")
    _mark(dirs, state, "charset", n_classes=n, path=str(dest))


def phase_audit(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "audit") and not args.force:
        _log(dirs, "audit: already done")
        return
    cmd = _mod(spec, "arabic_ocr_bench.charctc.audit") + [
        "--cache",
        str(dirs["cache"]),
        "--out",
        str(dirs["audit"]),
    ]
    if args.audit_limit:
        cmd.extend(["--limit", str(args.audit_limit)])
    if _run(dirs, cmd, phase="audit"):
        raise SystemExit("audit: failed")
    _mark(dirs, state, "audit", path=str(dirs["audit"] / "audit.md"))


def phase_train(dirs: dict, state: dict, spec: dict, args) -> None:
    complete = dirs["runs"] / "train_complete.json"
    wanted = _yaml_train_epochs(REPO / CONFIG)
    completed_epochs = None
    if complete.exists():
        try:
            completed_epochs = json.loads(complete.read_text(encoding="utf-8")).get("epochs")
        except json.JSONDecodeError:
            completed_epochs = None
    more = train_needs_more_epochs(completed_epochs, wanted)
    if complete.exists() and _done(state, "train") and not args.force and not more:
        _log(dirs, "train: already complete")
        return
    if more or args.force:
        _log(
            dirs,
            f"train: resume from checkpoint (complete_epochs={completed_epochs} "
            f"wanted={wanted}); invalidating stale smoke/full",
        )
        invalidate_eval_state(dirs, state)
    cmd = _mod(spec, "arabic_ocr_bench.charctc.train") + [
        "--config",
        str(REPO / CONFIG),
        "--cache",
        str(dirs["cache"]),
        "--out",
        str(dirs["runs"]),
        "--device",
        "cpu",
    ]
    if _run(dirs, cmd, phase="train"):
        raise SystemExit("train: failed (re-run the same command to resume from checkpoint-*)")
    if not complete.exists():
        raise SystemExit("train: exited 0 but train_complete.json missing")
    info = json.loads(complete.read_text(encoding="utf-8"))
    _mark(dirs, state, "train", **info)


def _eval_stage(dirs: dict, state: dict, spec: dict, args, *, stage: str, limit: int) -> bool:
    ckpt = dirs["runs"]
    exhausted = []
    for ds in EVAL_DATASETS:
        key = f"{stage}:{ds}"
        if _done(state, key) and not args.force:
            _log(dirs, f"{stage}: {ds} already done")
            exhausted.append(ds)
            continue
        out = dirs["bench"] / stage / ds
        cmd = _mod(spec, "arabic_ocr_bench.charctc.eval") + [
            "--ckpt",
            str(ckpt),
            "--dataset",
            ds,
            "--out",
            str(out),
            "--limit",
            str(limit),
            "--device",
            "cpu",
        ]
        if _run(dirs, cmd, phase=stage):
            raise SystemExit(f"{stage}: eval failed on {ds} — fix and re-run (resumes).")
        summary_path = out / "summary.json"
        rec = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        n_new = int(rec.get("n_new") or 0)
        if stage == "smoke" or n_new < limit:
            _mark(dirs, state, key, **rec)
            exhausted.append(ds)
        else:
            _log(dirs, f"{stage}: {ds} scored {n_new} more lines; re-run to continue")
    return len(exhausted) == len(EVAL_DATASETS)


def phase_smoke(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "smoke") and not args.force:
        _log(dirs, "smoke: already done")
        return
    if _eval_stage(dirs, state, spec, args, stage="smoke", limit=args.smoke_n):
        _mark(dirs, state, "smoke")


def phase_full(dirs: dict, state: dict, spec: dict, args) -> None:
    if _done(state, "full") and not args.force:
        _log(dirs, "full: already done")
        return
    if smoke_still_collapsed(dirs) and not args.force:
        _log(
            dirs,
            "full: skipped — smoke still all-blank (CER-light >= 0.99). "
            "Not scoring 200-line batches of empty hyps.",
        )
        return
    if _eval_stage(dirs, state, spec, args, stage="full", limit=args.full_n):
        _mark(dirs, state, "full")
        _log(dirs, "full: every dataset exhausted")
    else:
        _log(dirs, "full: batch done — re-run this command to score the next batch")


def phase_report(dirs: dict, state: dict, spec: dict, args) -> None:
    lines = [
        "# charctc-cpu report",
        "",
        f"Generated {_now()}",
        "",
        "Trained a closed-charset CNN-BiLSTM-CTC on the same 8k Muharaf train "
        "lines as exp3090-qwen3vl4b, on CPU, while the VLM suite kept the GPU.",
        "",
        "## How to read this",
        "",
        f"- Stock TrOCR (English tokenizer) on Muharaf in this repo: **~{TROCR_MUHARAF_CER:.0%} CER**.",
        "- Qwen3-VL-4B LoRA on the same 8k lines (held-out CER-light):",
    ]
    for ds, cer in QWEN_LORA_CER_LIGHT.items():
        lines.append(f"  - `{ds}`: {cer:.0%}")
    lines += [
        "- If char-CTC lands around 15–20% on Muharaf/KHATT/NAKBA, output-space was a real lever.",
        "- If it lands around 30%, tokenizer was not the bottleneck.",
        "- If RASAM stays ~50%, Maghrebi is still a vision problem.",
        "",
        "## Tokenizer audit",
        "",
    ]
    audit_md = dirs["audit"] / "audit.md"
    if audit_md.exists():
        lines.append(audit_md.read_text(encoding="utf-8"))
    else:
        lines.append("_audit.md missing_")
    lines += ["", "## Eval", "", "| dataset | stage | n | CER-light | vs Qwen LoRA | diac-F1 | OOV lines |", "|---|---|---|---|---|---|---|"]
    for stage in ("smoke", "full"):
        for ds in EVAL_DATASETS:
            p = dirs["bench"] / stage / ds / "summary.json"
            if not p.exists():
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            cer = rec.get("cer_micro_light")
            ref = QWEN_LORA_CER_LIGHT.get(ds)
            delta = ""
            if cer is not None and ref is not None:
                delta = f"{cer - ref:+.3f}"
            lines.append(
                f"| `{ds}` | {stage} | {rec.get('n')} | {_fmt(cer)} | {delta} | "
                f"{_fmt(rec.get('diac_f1'))} | {rec.get('n_oov_lines')} |"
            )
    train_state = dirs["runs"] / "train_state.json"
    if train_state.exists():
        lines += ["", "## Train state", "", "```json", train_state.read_text(encoding="utf-8"), "```"]
    dest = dirs["root"] / "report.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(dirs, f"report: wrote {dest}")
    _mark(dirs, state, "report", path=str(dest))


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


PHASE_FN = {
    "install": phase_install,
    "check": phase_check,
    "prepare": phase_prepare,
    "charset": phase_charset,
    "audit": phase_audit,
    "train": phase_train,
    "smoke": phase_smoke,
    "full": phase_full,
    "report": phase_report,
}


def _respawn_detached(dirs: dict, argv: list[str]) -> int:
    _rotate_log(dirs)
    child_argv = [sys.executable, str(Path(__file__).resolve()), *[a for a in argv if a != "--detach"]]
    pid = spawn_surviving_logout(
        child_argv,
        cwd=REPO,
        log_path=dirs["logs"] / LOG_NAME,
        pid_path=dirs["logs"] / PID_NAME,
    )
    print(f"detached pid {pid}")
    print(f"log : {dirs['logs'] / LOG_NAME}")
    print(f"poll: python {Path(__file__).name} --dir {dirs['root']} status")
    print(f"tail: Get-Content {dirs['logs'] / LOG_NAME} -Wait -Tail 50")
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        if IS_WIN:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _active_worker(dirs: dict) -> int | None:
    pidf = dirs["logs"] / PID_NAME
    if not pidf.exists():
        return None
    try:
        pid = int(pidf.read_text(encoding="utf-8").strip() or 0)
    except ValueError:
        return None
    if pid and pid not in (os.getpid(), os.getppid()) and _pid_alive(pid):
        return pid
    return None


def print_log_tail(text: str, n: int = 40, stream=None) -> None:
    stream = stream if stream is not None else sys.stdout
    payload = "\n".join(text.splitlines()[-n:])
    try:
        stream.write(payload + "\n")
        stream.flush()
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "utf-8"
        buf = getattr(stream, "buffer", None)
        data = (payload + "\n").encode(enc, errors="replace")
        if buf is not None:
            buf.write(data)
        else:
            stream.write(data.decode(enc, errors="replace"))


def configure_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass


def cmd_status(dirs: dict) -> int:
    state = _read_state(dirs)
    print(f"experiment dir : {dirs['root']}")
    pidf = dirs["logs"] / PID_NAME
    if pidf.exists():
        pid = int(pidf.read_text(encoding="utf-8").strip() or 0)
        print(f"worker pid     : {pid}  ({'alive' if _pid_alive(pid) else 'not running'})")
    print("\nphases:")
    for ph in PHASES:
        rec = state.get("phases", {}).get(ph, {})
        print(f"  {ph:<9} {'done' if rec.get('done') else '--':<6} {rec.get('at', '')}")
    sub = {k: v for k, v in state.get("phases", {}).items() if ":" in k}
    if sub:
        print("\nper-dataset:")
        for k in sorted(sub):
            rec = sub[k]
            cer = rec.get("cer_micro_light", "")
            print(f"  {k:<28} done  cer_light={cer}  {rec.get('at', '')}")
    ts = dirs["runs"] / "train_state.json"
    if ts.exists():
        rec = json.loads(ts.read_text(encoding="utf-8"))
        print(
            f"\ntrain          : step={rec.get('step')} epoch={rec.get('epoch')} "
            f"val_cer={rec.get('val_cer')} best={rec.get('best_val_cer')}"
        )
    ckpts = sorted(dirs["runs"].glob("checkpoint-*"))
    if ckpts:
        print(f"newest ckpt    : {ckpts[-1].name}")
    log = dirs["logs"] / LOG_NAME
    if log.exists():
        print("\n--- log tail (40) ---")
        print_log_tail(log.read_text(encoding="utf-8", errors="replace"))
    return 0


def drive(dirs: dict, spec: dict, args, argv: list[str]) -> int:
    state = _read_state(dirs)
    state.setdefault("started", _now())
    _write_state(dirs, state)

    active = _active_worker(dirs)
    if active and not args.force:
        print(
            f"a charctc run is already active in this folder (pid {active}).\n"
            f"  watch it : python {Path(__file__).name} --dir {dirs['root']} status\n"
            f"  stop it  : {'taskkill /F /PID' if IS_WIN else 'kill'} {active}\n"
            f"  or pass --force to start anyway (only if you know it is dead)."
        )
        return 1

    if args.detach:
        return _respawn_detached(dirs, argv)

    if not stdout_is_regular_file():
        _rotate_log(dirs)
    (dirs["logs"] / PID_NAME).write_text(str(os.getpid()), encoding="utf-8")
    start = args.from_phase or ("install" if not args.only else args.only)
    plan = [args.only] if args.only else PHASES[PHASES.index(start):]
    _log(dirs, f"=== charctc-cpu: phases={plan} device=cpu ===")
    _log(dirs, f"venv={spec['venv']} hf_home={spec['hf_home']}")
    try:
        for ph in plan:
            _log(dirs, f"--- phase {ph} ---")
            try:
                PHASE_FN[ph](dirs, state, spec, args)
            except SystemExit as e:
                _log(dirs, f"STOP at phase {ph}: {e}")
                return 1
            state = _read_state(dirs)
        _log(dirs, "=== pass complete ===")
        _log(dirs, f"report : {dirs['root'] / 'report.md'}")
        _log(dirs, f"log    : {dirs['logs'] / LOG_NAME}")
        return 0
    finally:
        _kill_child()
        pidf = dirs["logs"] / PID_NAME
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


def _resolve_venv(args, dirs: dict) -> Path:
    if args.venv:
        return Path(args.venv).expanduser().resolve()
    saved = _read_state(dirs).get("phases", {}).get("install", {}).get("venv")
    if saved and _venv_python(Path(saved)).exists():
        return Path(saved)
    if IS_WIN:
        for candidate in DEFAULT_VENVS_WIN:
            if _venv_python(candidate).exists():
                return candidate
    raise SystemExit(
        "no --venv given and no suite_v1 venv was found. "
        "Pass --venv C:\\Users\\User\\a-experiment\\suite-v1\\qwen3vl4b_suite\\venv"
    )


def _resolve_hf_home(args, dirs: dict) -> Path:
    if args.hf_home:
        return Path(args.hf_home).expanduser().resolve()
    if IS_WIN:
        for candidate in DEFAULT_HF_WIN:
            if candidate.exists():
                return candidate
    return dirs["hf_cache"]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="run_charctc_experiment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", required=True, help="experiment folder (logs, cache, runs, bench)")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "status"])
    ap.add_argument(
        "--venv",
        default=None,
        help="suite_v1 venv with torch+transformers (NOT created, NOT pip-installed into)",
    )
    ap.add_argument("--hf-home", dest="hf_home", default=None)
    ap.add_argument("--limit", type=int, default=None, help="reserved; yaml train limit is the source of truth")
    ap.add_argument("--smoke-n", dest="smoke_n", type=int, default=20)
    ap.add_argument("--full-n", dest="full_n", type=int, default=200)
    ap.add_argument("--audit-limit", dest="audit_limit", type=int, default=500)
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--from", dest="from_phase", default=None, choices=PHASES)
    ap.add_argument("--only", default=None, choices=PHASES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    configure_stdio()
    dirs = _dirs(Path(args.dir).expanduser().resolve())
    venv = _resolve_venv(args, dirs)
    hf_home = _resolve_hf_home(args, dirs)
    _pin_env(dirs, venv=venv, hf_home=hf_home)
    if args.cmd == "status":
        return cmd_status(dirs)

    spec = {"venv": venv, "hf_home": hf_home}
    return drive(dirs, spec, args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
