"""Snapshot AIN NAKBA LoRA checkpoints and score them after the 3090 is free.

Training owns the GPU. This waiter copies adapter weights (not optimizer.pt)
so ``save_total_limit`` cannot drop them, then — once ``pipeline.pid`` is gone
and nvidia-smi shows the card idle — registers each adapter on the suite-v1
AIN merged base and runs the 200-line protocol on the five NAKBA-job datasets.

    python scripts/eval_ain_nakba_ckpts.py --dir C:\\Users\\User\\a-experiment\\suite-v1\\ain7b_nakba --detach

Results land in ``<dir>/ckpt_eval/`` (summaries + table.csv). Does not touch
the live Trainer output dir except to read it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"

from _detach_spawn import emit_log_line, spawn_surviving_logout, stdout_is_regular_file  # noqa: E402
from _envfile import load_experiment_env  # noqa: E402
from _experiment_bases import BASES  # noqa: E402

SPEC = BASES["ain7b_nakba"]
DATASETS = ["nakba_lines", "kitab_khatt", "muharaf_lines", "rasam1", "rasam2"]
LIMIT = 200
PROMPT_VERSION = "line-v2"
NORM = "raw, light, diac_insensitive"
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "trainer_state.json",
)
GPU_IDLE_MIB = 1500
IDLE_STREAK_NEEDED = 2
POLL_S = 60


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(logs: Path, msg: str) -> None:
    emit_log_line(logs / "ckpt_eval.log", f"[{_now()}] {msg}")


def _pid_alive(pid: int) -> bool:
    try:
        if IS_WIN:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, check=False,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _pid_is_cmd(pid: int) -> bool:
    """True when *pid* is a cmd.exe wrapper (WMI ``--detach`` parent), not the eval."""
    if not IS_WIN or not pid:
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
        capture_output=True, text=True, check=False,
    ).stdout.lower()
    return "cmd.exe" in out


def _gpu_used_mib() -> int | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return None
    try:
        return int(float(out.stdout.strip().splitlines()[0].strip()))
    except (ValueError, IndexError):
        return None


def _pidfile_alive(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip() or 0)
    except ValueError:
        return False
    return bool(pid) and _pid_alive(pid)


def _pipeline_alive(root: Path) -> bool:
    logs = root / "logs"
    return _pidfile_alive(logs / "pipeline.pid") or _pidfile_alive(logs / "run.pid")


def _pin_env(root: Path) -> None:
    os.environ.pop("PYTHONHOME", None)
    venv = root / "venv"
    os.environ["VIRTUAL_ENV"] = str(venv)
    scripts = venv / ("Scripts" if IS_WIN else "bin")
    os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
    src = str(REPO / "src")
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    os.environ["HF_HOME"] = str(root / "hf_cache")
    os.environ["UV_CACHE_DIR"] = str(root / "uv_cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    load_experiment_env(root)
    os.environ.pop("PYTHONHOME", None)


def _venv_python(root: Path) -> Path:
    return root / "venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _ckpt_ready(path: Path) -> bool:
    weights = path / "adapter_model.safetensors"
    cfg = path / "adapter_config.json"
    if not (weights.exists() and cfg.exists()):
        return False
    return weights.stat().st_size > 50 * 1024 * 1024


def snapshot_checkpoints(root: Path, logs: Path) -> list[Path]:
    src_root = root / "runs" / SPEC["experiment"]
    dest_root = root / "ckpt_snapshots"
    dest_root.mkdir(parents=True, exist_ok=True)
    snapped: list[Path] = []
    if not src_root.exists():
        return snapped
    for src in sorted(src_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        if not _ckpt_ready(src):
            continue
        dest = dest_root / src.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ADAPTER_FILES:
            s = src / name
            if not s.exists():
                continue
            d = dest / name
            if not d.exists() or s.stat().st_size != d.stat().st_size or s.stat().st_mtime > d.stat().st_mtime:
                shutil.copy2(s, d)
        snapped.append(dest)
    _log(logs, f"snapshot: {len(snapped)} checkpoints -> {dest_root}")
    return snapped


def _write_run_cfg(path: Path, *, model: str, dataset: str, results_dir: Path,
                   limit: int | None = LIMIT, prompt_version: str = PROMPT_VERSION) -> None:
    limit_line = "" if limit is None else f"limit: {limit}\n"
    path.write_text(
        "# generated by eval_ain_nakba_ckpts.py — do not edit\n"
        f"models: [{model}]\n"
        f"datasets: [{dataset}]\n"
        f"{limit_line}"
        "offset: 0\n"
        "accumulate: false\n"
        f"prompt_version: {prompt_version}\n"
        f"normalization_profiles: [{NORM}]\n"
        "concurrency: 1\n"
        "max_retries: 2\n"
        "cache: true\n"
        f"results_dir: {results_dir.as_posix()}\n",
        encoding="utf-8",
    )


def _run(logs: Path, cmd: list[str], *, cwd: Path) -> int:
    _log(logs, f"$ {' '.join(str(c) for c in cmd)}")
    tee = not stdout_is_regular_file()
    fh = (logs / "ckpt_eval.log").open("a", encoding="utf-8") if tee else None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env={**os.environ},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
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


def wait_for_gpu(root: Path, logs: Path) -> None:
    idle_streak = 0
    while True:
        snapshot_checkpoints(root, logs)
        used = _gpu_used_mib()
        pipe = _pipeline_alive(root)
        _log(logs, f"wait: pipeline={'alive' if pipe else 'dead'} gpu_mib={used}")
        if not pipe and used is not None and used < GPU_IDLE_MIB:
            idle_streak += 1
            if idle_streak >= IDLE_STREAK_NEEDED:
                _log(logs, "wait: GPU idle and pipeline finished")
                return
        else:
            idle_streak = 0
        time.sleep(POLL_S)


def _eval_loss(ckpt: Path) -> float | None:
    p = ckpt / "trainer_state.json"
    if not p.exists():
        return None
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    best = st.get("best_metric")
    if isinstance(best, (int, float)):
        return float(best)
    for row in reversed(st.get("log_history") or []):
        if "eval_loss" in row:
            return float(row["eval_loss"])
    return None


def register_and_score(root: Path, logs: Path, ckpts: list[Path]) -> None:
    py = _venv_python(root)
    out = root / "ckpt_eval"
    results = out / "results"
    cfg_dir = out / "_cfg"
    for p in (out, results, cfg_dir):
        p.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"done": {}}

    for ckpt in ckpts:
        step = ckpt.name.split("-")[1]
        name = f"ain7b-nakba-ckpt-{step}"
        card = f"AIN NAKBA LoRA {ckpt.name} (adapter on suite-v1 AIN, not merged)."
        rc = _run(logs, [
            str(py), "-m", "arabic_ocr_bench.training.cli", "register",
            "--name", name, "--path", SPEC["hf_id"], "--adapter-id", str(ckpt),
            "--arch", SPEC["arch"], "--max-new-tokens", "640", "--card", card,
        ], cwd=REPO)
        if rc:
            raise SystemExit(f"register {name} failed ({rc})")

    # Dataset-outer so NAKBA CER for every checkpoint lands before transfer sets.
    for ds in DATASETS:
        for ckpt in ckpts:
            step = ckpt.name.split("-")[1]
            name = f"ain7b-nakba-ckpt-{step}"
            key = f"{name}:{ds}"
            if state["done"].get(key):
                _log(logs, f"skip {key} (already done)")
                continue
            cfg = cfg_dir / f"{name}__{ds}.run.yaml"
            _write_run_cfg(cfg, model=name, dataset=ds, results_dir=results)
            rc = _run(logs, [str(py), "-m", "arabic_ocr_bench.cli", "run",
                             "--config", str(cfg), "--report"], cwd=REPO)
            if rc:
                raise SystemExit(f"bench {key} failed ({rc})")
            runs = sorted(results.glob("*/summary.csv"), key=lambda p: p.stat().st_mtime)
            if runs:
                src = runs[-1].parent
                dst = out / name / ds / src.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in ("summary.csv", "summary.md", "scores.jsonl",
                          "predictions.jsonl", "run_config.snapshot.yaml"):
                    if (src / f).exists():
                        shutil.copy2(src / f, dst / f)
            state["done"][key] = {"at": _now()}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            _log(logs, f"scored {key}")
            _write_table(out, ckpts)
    _write_table(out, ckpts)
    _log(logs, f"done. table -> {out / 'table.csv'}")


def _write_table(out: Path, ckpts: list[Path]) -> None:
    rows: list[dict] = []
    for ckpt in ckpts:
        step = ckpt.name.split("-")[1]
        name = f"ain7b-nakba-ckpt-{step}"
        eloss = _eval_loss(ckpt)
        for ds in DATASETS:
            matches = sorted((out / name / ds).glob("*/summary.csv"))
            cer = n = err = ""
            if matches:
                with matches[-1].open(encoding="utf-8", newline="") as fh:
                    rec = next(csv.DictReader(fh), None)
                if rec:
                    cer = rec.get("cer_micro_light") or ""
                    n = rec.get("n") or ""
                    err = rec.get("errors") or ""
            rows.append({
                "checkpoint": ckpt.name,
                "model": name,
                "dataset": ds,
                "n": n,
                "errors": err,
                "cer_micro_light": cer,
                "eval_loss": "" if eloss is None else f"{eloss:.6f}",
            })
    path = out / "table.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "checkpoint", "model", "dataset", "n", "errors", "cer_micro_light", "eval_loss",
        ])
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| checkpoint | dataset | n | CER-light | eval_loss |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        cer = r["cer_micro_light"]
        try:
            cer_s = f"{float(cer)*100:.1f}%"
        except (TypeError, ValueError):
            cer_s = cer or "—"
        lines.append(
            f"| {r['checkpoint']} | {r['dataset']} | {r['n'] or '—'} | {cer_s} | {r['eval_loss'] or '—'} |"
        )
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cer_by_ckpt_step(table_path: Path) -> dict[int, float]:
    """Map checkpoint step -> cer_micro_raw from an official table.csv."""
    out: dict[int, float] = {}
    if not table_path.is_file():
        return out
    with table_path.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            name = rec.get("model") or ""
            raw = rec.get("cer_micro_raw") or ""
            if not name.startswith("ain7b-nakba-ckpt-") or not raw:
                continue
            try:
                step = int(name.rsplit("-", 1)[-1])
                out[step] = float(raw)
            except ValueError:
                continue
    return out


def should_skip_ckpt400(cers: dict[int, float]) -> bool:
    """Skip official 400 unless 800 is strictly better than 1600 (lower CER)."""
    c800 = cers.get(800)
    c1600 = cers.get(1600)
    if c800 is None or c1600 is None:
        return True
    return c800 >= c1600


OFFICIAL_DATASET = "nakba_blind"
OFFICIAL_REF = "nakba-misraj-baseer"
OFFICIAL_REF_CONTEST = "nakba-misraj-baseer-contest"
OFFICIAL_STEPS = {1200, 1600, 1936}
# job = (model_name, adapter_or_none, prompt_version)
OfficialJob = tuple[str, str | None, str]


def _ckpt_model_prefix() -> str:
    if SPEC.get("official_all_ckpts"):
        return "ain7b-nakba-visft-ckpt"
    return "ain7b-nakba-ckpt"


def _official_jobs(ckpts: list[Path]) -> list[OfficialJob]:
    jobs: list[OfficialJob] = []
    for ckpt in sorted(ckpts, key=lambda p: int(p.name.split("-")[1]), reverse=True):
        step = ckpt.name.split("-")[1]
        jobs.append((f"{_ckpt_model_prefix()}-{step}", str(ckpt), PROMPT_VERSION))
    if not SPEC.get("official_all_ckpts"):
        jobs.append((OFFICIAL_REF, None, PROMPT_VERSION))
        jobs.append((OFFICIAL_REF_CONTEST, None, "nakba-contest"))
    return jobs


def _contest_from_preds(pred_path: Path) -> dict[str, str]:
    empty = {
        "cer_contest_micro": "", "cer_contest_macro": "",
        "wer_contest_micro": "", "wer_contest_macro": "",
    }
    if not pred_path.is_file():
        return empty
    pairs: list[tuple[str, str]] = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref = str(row.get("ground_truth") or "")
        if row.get("error"):
            hyp = ""
        else:
            hyp = str(row.get("cleaned_text") if row.get("cleaned_text") is not None
                      else (row.get("raw_text") or ""))
        pairs.append((ref, hyp))
    from arabic_ocr_bench.metrics import contest_aggregate
    agg = contest_aggregate(pairs)
    return {
        "cer_contest_micro": f"{agg['cer_micro']:.4f}",
        "cer_contest_macro": f"{agg['cer_macro']:.4f}",
        "wer_contest_micro": f"{agg['wer_micro']:.4f}",
        "wer_contest_macro": f"{agg['wer_macro']:.4f}",
    }


def score_official(
    root: Path, logs: Path, ckpts: list[Path], *,
    infer: bool = True, only_steps: set[int] | None = None,
) -> int:
    """Full 2,671-line Systems-Track split. Headline metric is corpus CER-raw."""
    py = _venv_python(root)
    use = list(ckpts)
    if only_steps is not None:
        use = [c for c in use if int(c.name.split("-")[1]) in only_steps]
    elif not SPEC.get("official_all_ckpts"):
        use = [c for c in ckpts if int(c.name.split("-")[1]) in OFFICIAL_STEPS]
    if infer and not use:
        want = sorted(only_steps) if only_steps is not None else sorted(OFFICIAL_STEPS)
        _log(logs, f"official: need checkpoint(s) for steps {want}")
        return 1
    out = root / "ckpt_eval" / "official"
    results = out / "results"
    cfg_dir = out / "_cfg"
    for p in (out, results, cfg_dir):
        p.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"done": {}}
    # Infer only the requested steps; keep the full official table of snapped ckpts.
    infer_jobs = _official_jobs(use)
    table_ckpts = list(ckpts) if SPEC.get("official_all_ckpts") or only_steps is not None else [
        c for c in ckpts if int(c.name.split("-")[1]) in OFFICIAL_STEPS
    ]
    table_jobs = _official_jobs(table_ckpts)

    if infer:
        for name, adapter, _prompt in infer_jobs:
            if adapter is None:
                continue
            card = f"AIN NAKBA LoRA adapter {name} on official nakba_blind."
            rc = _run(logs, [
                str(py), "-m", "arabic_ocr_bench.training.cli", "register",
                "--name", name, "--path", SPEC["hf_id"], "--adapter-id", adapter,
                "--arch", SPEC["arch"], "--max-new-tokens", "640", "--card", card,
            ], cwd=REPO)
            if rc:
                raise SystemExit(f"register {name} failed ({rc})")

    _write_official_table(out, table_jobs)
    if not infer:
        _log(logs, f"contest rescore only. table -> {out / 'table.csv'}")
        return 0

    for name, adapter, prompt_version in infer_jobs:
        key = f"{name}:{OFFICIAL_DATASET}"
        if state["done"].get(key):
            _log(logs, f"skip {key} (already done)")
            continue
        cfg = cfg_dir / f"{name}__{OFFICIAL_DATASET}.run.yaml"
        _write_run_cfg(cfg, model=name, dataset=OFFICIAL_DATASET,
                       results_dir=results, limit=None, prompt_version=prompt_version)
        rc = _run(logs, [str(py), "-m", "arabic_ocr_bench.cli", "run",
                         "--config", str(cfg), "--report"], cwd=REPO)
        if rc:
            raise SystemExit(f"bench {key} failed ({rc})")
        runs = sorted(results.glob("*/summary.csv"), key=lambda p: p.stat().st_mtime)
        if runs:
            src = runs[-1].parent
            dst = out / name / src.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("summary.csv", "summary.md", "scores.jsonl",
                      "predictions.jsonl", "run_config.snapshot.yaml"):
                if (src / f).exists():
                    shutil.copy2(src / f, dst / f)
        state["done"][key] = {"at": _now(), "adapter": adapter, "prompt": prompt_version}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        _write_official_table(out, table_jobs)
        _log(logs, f"scored {key}")
    _write_official_table(out, table_jobs)
    _log(logs, f"official done. table -> {out / 'table.csv'}")
    return 0


def _write_official_table(out: Path, jobs: list[OfficialJob]) -> None:
    rows: list[dict] = []
    for name, _adapter, _prompt in jobs:
        matches = sorted((out / name).glob("*/summary.csv"))
        rec = {}
        pred_path = Path()
        if matches:
            with matches[-1].open(encoding="utf-8", newline="") as fh:
                rec = next(csv.DictReader(fh), {}) or {}
            pred_path = matches[-1].parent / "predictions.jsonl"
        row = {
            "model": name,
            "dataset": OFFICIAL_DATASET,
            "n": rec.get("n", ""),
            "errors": rec.get("errors", ""),
            "cer_micro_raw": rec.get("cer_micro_raw", ""),
            "cer_macro_raw": rec.get("cer_macro_raw", ""),
            "wer_micro_raw": rec.get("wer_micro_raw", ""),
            "cer_micro_light": rec.get("cer_micro_light", ""),
        }
        row.update(_contest_from_preds(pred_path) if pred_path.is_file() else {
            "cer_contest_micro": "", "cer_contest_macro": "",
            "wer_contest_micro": "", "wer_contest_macro": "",
        })
        rows.append(row)
    path = out / "table.csv"
    fields = ["model", "dataset", "n", "errors", "cer_micro_raw", "cer_macro_raw",
              "wer_micro_raw", "cer_micro_light",
              "cer_contest_micro", "cer_contest_macro",
              "wer_contest_micro", "wer_contest_macro"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| model | n | corpus CER (raw) | contest CER (micro) | contest CER (macro) | "
        "per-line CER (raw) | WER (raw) | CER-light |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def pct(key: str) -> str:
            try:
                return f"{float(r[key])*100:.2f}%"
            except (TypeError, ValueError, KeyError):
                return r.get(key) or "—"
        lines.append(
            f"| {r['model']} | {r['n'] or '—'} | {pct('cer_micro_raw')} | "
            f"{pct('cer_contest_micro')} | {pct('cer_contest_macro')} | "
            f"{pct('cer_macro_raw')} | {pct('wer_micro_raw')} | {pct('cer_micro_light')} |"
        )
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base", default="ain7b_nakba", choices=list(BASES),
                    help="BASES key; visft scores every snapped checkpoint.")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--snapshot-only", action="store_true")
    ap.add_argument("--official", action="store_true",
                    help="Score epoch-1 checkpoints + Baseer on nakba_blind (2,671 lines, no limit).")
    ap.add_argument("--rescore-contest", action="store_true",
                    help="Rewrite official table with contest CER/WER from existing predictions.jsonl (CPU).")
    ap.add_argument("--no-wait", action="store_true",
                    help="Skip the GPU/pipeline idle wait (caller already freed the card).")
    ap.add_argument("--only-steps", type=int, nargs="+", default=None,
                    help="Score only these checkpoint steps (e.g. --only-steps 400).")
    args = ap.parse_args(argv)

    global SPEC
    SPEC = BASES[args.base]

    root = Path(args.dir).expanduser().resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pidf = logs / "ckpt_eval.pid"
    if pidf.exists():
        try:
            other = int(pidf.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            other = 0
        if other and other not in (os.getpid(), os.getppid()) and _pid_alive(other) and not _pid_is_cmd(other):
            print(f"ckpt eval already running (pid {other})")
            return 1

    if args.detach:
        child = [sys.executable, str(Path(__file__).resolve()),
                 *[a for a in argv if a != "--detach"]]
        pid = spawn_surviving_logout(
            child, cwd=REPO, log_path=logs / "ckpt_eval.log", pid_path=pidf,
        )
        print(f"detached pid {pid}")
        print(f"log : {logs / 'ckpt_eval.log'}")
        return 0

    pidf.write_text(str(os.getpid()), encoding="utf-8")
    _pin_env(root)
    try:
        ckpts = snapshot_checkpoints(root, logs)
        if args.snapshot_only:
            print(f"snapshotted {len(ckpts)} checkpoints")
            for c in ckpts:
                print(c)
            return 0
        if args.rescore_contest:
            return score_official(root, logs, ckpts, infer=False)
        if not args.no_wait:
            if not ckpts and not args.official:
                _log(logs, "no checkpoints yet; waiting anyway")
            wait_for_gpu(root, logs)
            ckpts = snapshot_checkpoints(root, logs)
        if args.official:
            only = set(args.only_steps) if args.only_steps else None
            return score_official(root, logs, ckpts, infer=True, only_steps=only)
        if not ckpts:
            _log(logs, "no checkpoints to score")
            return 1
        register_and_score(root, logs, ckpts)
        return 0
    finally:
        if pidf.exists() and pidf.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pidf.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
