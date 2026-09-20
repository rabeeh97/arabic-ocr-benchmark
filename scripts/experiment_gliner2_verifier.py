"""Text-only confidence gate on OCR output, using an Arabic GLiNER2 checkpoint.

For each already-scored prediction, ask GLiNER2 whether the text is well-formed Arabic
(confidence 0..1), then split CER into accepted / refused at several thresholds.
The gate never sees the image, so it can flag garbled text but not fluent-but-wrong text.

GLiNER2 needs torch, which is deliberately NOT a repo dependency. Run from a scratch venv:

    uv venv /tmp/gliner_venv --python 3.11
    uv pip install --python /tmp/gliner_venv/bin/python "gliner2[local]" jiwer regex
    PYTHONPATH=src /tmp/gliner_venv/bin/python scripts/experiment_gliner2_verifier.py \
        --run results/20260913T093935Z --models paligemma2-3b-bf16 exp3090-paligemma2-3b-ft-bf16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arabic_ocr_bench.metrics import aggregate, score_pair  # noqa: E402

CHECKPOINT = "Elafnawaf/gliner2-arabic-multitask"
LABELS = ["well_formed_arabic", "garbled_ocr_error"]
GOOD = LABELS[0]
THRESHOLDS = (0.5, 0.7, 0.9)
HIGH_CER = 0.3
MAX_CHARS = 1500  # long pages are truncated; the model has a bounded context

_STRIP = re.compile("[ً-ٰٟـ​-‏‪-‮﻿]")


def normalize_for_model(s: str) -> str:
    return re.sub(r"\s+", " ", _STRIP.sub("", s)).strip()[:MAX_CHARS]


def load_rows(run: Path, models: list[str], dataset: str | None) -> list[dict]:
    rows = []
    for line in (run / "predictions.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["model"] in models and not r.get("error") and dataset in (None, r["dataset"]):
            rows.append(r)
    return rows


def _confidence_of(out: dict) -> float:
    """Extract P(well-formed) from whatever shape classify_text returned."""
    v = out.get("ocr_quality", out) if isinstance(out, dict) else out
    if isinstance(v, dict):
        if "confidence" in v and v.get("label") == GOOD:
            return float(v["confidence"])
        if "confidence" in v:
            return 1.0 - float(v["confidence"])
        if GOOD in v:
            return float(v[GOOD])
    if isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], str):
        return float(v[1]) if v[0] == GOOD else 1.0 - float(v[1])
    raise ValueError(f"unrecognised classify_text output shape: {out!r}")


class Scorer:
    def __init__(self, cache_path: Path):
        from gliner2.auto import AutoExtractor

        self.model = AutoExtractor.from_pretrained(CHECKPOINT)
        self.cache_path = cache_path
        self.cache: dict[str, float] = {}
        if cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                d = json.loads(line)
                self.cache[d["key"]] = d["conf"]
        self.hits = self.misses = 0

    def __call__(self, text: str) -> float:
        text = normalize_for_model(text)
        key = hashlib.sha1(text.encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        if not text:
            conf = 0.0
        else:
            out = self.model.classify_text(
                text, {"ocr_quality": LABELS}, include_confidence=True
            )
            conf = _confidence_of(out)
        self.cache[key] = conf
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "conf": conf}) + "\n")
        return conf


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(score_pos > score_neg), ties count half. None when a class is empty."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def micro_cer(rows: list[dict]) -> float | None:
    if not rows:
        return None
    agg = aggregate([r["_ps"] for r in rows], profiles=("raw",))
    return agg["micro_cer"]["raw"]


def shuffled(s: str, seed: int) -> str:
    chars = list(s)
    random.Random(seed).shuffle(chars)
    return "".join(chars)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default=None, help="only rows from this dataset")
    ap.add_argument("--limit", type=int, default=None, help="max rows per model")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "gliner2_verifier")
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else REPO / args.run
    out_dir = args.out / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(args.out / "_cache.jsonl")

    rows = load_rows(run, args.models, args.dataset)
    if args.limit:
        per: dict[str, int] = {}
        kept = []
        for r in rows:
            per[r["model"]] = per.get(r["model"], 0) + 1
            if per[r["model"]] <= args.limit:
                kept.append(r)
        rows = kept

    for i, r in enumerate(rows):
        r["_ps"] = score_pair(r["ground_truth"], r.get("cleaned_text") or "", profiles=("raw",))
        r["conf"] = scorer(r.get("cleaned_text") or "")
        r["conf_gt"] = scorer(r["ground_truth"])
        r["conf_shuf"] = scorer(shuffled(r.get("cleaned_text") or "", seed=i))
        r["cer"] = r["_ps"]["cer"]["raw"]

    with (out_dir / "confidences.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "model": r["model"], "sample_id": r["sample_id"], "conf": r["conf"],
                "conf_ground_truth": r["conf_gt"], "conf_shuffled": r["conf_shuf"],
                "cer_raw": r["cer"],
            }) + "\n")

    fields = ["model", "n", "mean_conf", "mean_conf_ground_truth", "mean_conf_shuffled",
              "auroc_gt_vs_shuffled", "auroc_conf_predicts_low_cer", "threshold",
              "n_accepted", "n_refused", "cer_overall", "cer_accepted", "cer_refused"]
    table = []
    for model in args.models:
        mr = [r for r in rows if r["model"] == model]
        if not mr:
            continue
        mean = lambda k: sum(r[k] for r in mr) / len(mr)  # noqa: E731
        base = {
            "model": model, "n": len(mr),
            "mean_conf": round(mean("conf"), 4),
            "mean_conf_ground_truth": round(mean("conf_gt"), 4),
            "mean_conf_shuffled": round(mean("conf_shuf"), 4),
            "auroc_gt_vs_shuffled": auroc([r["conf_gt"] for r in mr], [r["conf_shuf"] for r in mr]),
            "auroc_conf_predicts_low_cer": auroc(
                [r["conf"] for r in mr if r["cer"] <= HIGH_CER],
                [r["conf"] for r in mr if r["cer"] > HIGH_CER],
            ),
            "cer_overall": micro_cer(mr),
        }
        for t in THRESHOLDS:
            acc = [r for r in mr if r["conf"] >= t]
            ref = [r for r in mr if r["conf"] < t]
            table.append({**base, "threshold": t, "n_accepted": len(acc), "n_refused": len(ref),
                          "cer_accepted": micro_cer(acc), "cer_refused": micro_cer(ref)})

    def fmt(v):
        return "" if v is None else (round(v, 4) if isinstance(v, float) else v)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in table:
            w.writerow({k: fmt(t.get(k)) for k in fields})

    print(f"cache: {scorer.hits} hits / {scorer.misses} model calls")
    print(f"wrote {out_dir}")
    for t in table:
        print({k: fmt(t.get(k)) for k in fields})


if __name__ == "__main__":
    main()
