"""Compare reference-free confidence signals for OCR lines, within each model.

Signals (higher = more likely correct; none of them use the ground truth):
  agree      mean similarity to the other models' outputs for the same line
  lm         negative per-character NLL under a small Arabic-capable LM
  gliner     GLiNER2-Arabic P(well-formed)  (reuses results/gliner2_verifier/_cache.jsonl)
  len_dev    negative |log(length / median length of the other models' outputs)|
  repeat     negative fraction of repeated character 4-grams
  nonarabic  negative fraction of letters that are not Arabic script
  random     control

Evaluated per model (models with >= --min-lines lines), never pooled across models, so a signal
that merely tells good models from bad ones gets no credit. Metrics: Spearman(signal, -CER) and the
CER improvement from keeping the top-coverage share of lines, with bootstrap CIs over lines.
All per-line signals are saved to signals.jsonl so nothing needs recomputing.

Run from a venv with torch/transformers/gliner2/jiwer (see docs/experiments/gliner2-verifier.md):
    PYTHONPATH=src python scripts/experiment_confidence_baselines.py --dataset kitab_khatt
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from arabic_ocr_bench.metrics import score_pair  # noqa: E402
from experiment_gliner2_verifier import Scorer, normalize_for_model  # noqa: E402

LM_NAME = "Qwen/Qwen2.5-0.5B"
COVERAGES = (0.5, 0.8)
SIGNALS = ["agree", "lm", "gliner", "len_dev", "repeat", "nonarabic", "random"]
_ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]


def load_latest(dataset: str) -> dict[tuple[str, str], dict]:
    """Latest non-errored prediction per (model, sample_id) across all run dirs."""
    out: dict[tuple[str, str], dict] = {}
    for p in sorted(glob.glob(str(REPO / "results" / "2*" / "predictions.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["dataset"] != dataset or r.get("error") or r["model"].startswith("mock"):
                continue
            out[(r["model"], r["sample_id"])] = r
    return out


def repeat_frac(s: str) -> float:
    grams = [s[i:i + 4] for i in range(len(s) - 3)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def nonarabic_frac(s: str) -> float:
    letters = [c for c in s if unicodedata.category(c).startswith("L")]
    if not letters:
        return 1.0
    bad = sum(not any(a <= ord(c) <= b for a, b in _ARABIC) for c in letters)
    return bad / len(letters)


class LM:
    def __init__(self, cache_path: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(LM_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(LM_NAME, torch_dtype=torch.float32).eval()
        self.cache_path = cache_path
        self.cache: dict[str, float] = {}
        if cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                d = json.loads(line)
                self.cache[d["key"]] = d["nll_per_char"]

    def nll_per_char(self, text: str) -> float:
        text = normalize_for_model(text)
        key = hashlib.sha1((LM_NAME + "|" + text).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if not text:
            val = 10.0
        else:
            ids = self.tok(text, return_tensors="pt").input_ids
            bos = self.torch.tensor([[self.tok.eos_token_id]])
            ids = self.torch.cat([bos, ids], dim=1)
            with self.torch.no_grad():
                logits = self.model(ids).logits[:, :-1]
            nll = self.torch.nn.functional.cross_entropy(
                logits[0], ids[0, 1:], reduction="sum"
            ).item()
            val = nll / len(text)
        self.cache[key] = val
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "nll_per_char": val}) + "\n")
        return val


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def gain_at(sig: np.ndarray, cer: np.ndarray, coverage: float) -> float:
    """Mean CER over all lines minus mean CER over the top-`coverage` share by signal."""
    k = max(1, int(round(coverage * len(sig))))
    keep = np.argsort(-sig, kind="stable")[:k]
    return float(cer.mean() - cer[keep].mean())


def metrics_for(sig: np.ndarray, cer: np.ndarray) -> dict[str, float]:
    out = {"rho": spearman(sig, -cer)}
    for c in COVERAGES:
        out[f"gain@{c}"] = gain_at(sig, cer, c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-lines", type=int, default=100)
    ap.add_argument("--min-others", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-gliner", action="store_true")
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--dry", action="store_true", help="only list models/lines")
    args = ap.parse_args()

    rows = load_latest(args.dataset)
    per_model: dict[str, list[str]] = defaultdict(list)
    per_line: dict[str, list[str]] = defaultdict(list)
    for (m, s) in rows:
        per_model[m].append(s)
        per_line[s].append(m)
    eval_models = sorted(m for m, ss in per_model.items() if len(ss) >= args.min_lines)
    print(f"{args.dataset}: {len(per_model)} models, {len(per_line)} lines; "
          f"evaluating {len(eval_models)} models with >= {args.min_lines} lines")
    for m in eval_models:
        print(f"  {m}: {len(per_model[m])} lines")
    if args.dry:
        return

    norm = {k: normalize_for_model(r.get("cleaned_text") or "") for k, r in rows.items()}
    gliner = None if args.no_gliner else Scorer(REPO / "results" / "gliner2_verifier" / "_cache.jsonl")
    out_dir = REPO / "results" / "confidence_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = None if args.no_lm else LM(out_dir / "_lm_cache.jsonl")

    rng = np.random.default_rng(0)
    table: list[dict] = []
    for m in eval_models:
        for s in per_model[m]:
            others = [o for o in per_line[s] if o != m]
            if len(others) < args.min_others:
                continue
            r = rows[(m, s)]
            text = r.get("cleaned_text") or ""
            t = norm[(m, s)]
            other_texts = [norm[(o, s)] for o in others]
            med_len = statistics.median(len(x) for x in other_texts)
            row = {
                "model": m, "sample_id": s, "n_others": len(others),
                "cer": score_pair(r["ground_truth"], text, profiles=("raw",))["cer"]["raw"],
                "agree": float(np.mean([Levenshtein.normalized_similarity(t, x) for x in other_texts])),
                "len_dev": -abs(math.log((len(t) + 1) / (med_len + 1))),
                "repeat": -repeat_frac(t),
                "nonarabic": -nonarabic_frac(t),
                "random": float(rng.random()),
            }
            if lm:
                row["lm"] = -lm.nll_per_char(text)
            if gliner:
                row["gliner"] = gliner(text)
            table.append(row)
    print(f"{len(table)} model-lines scored")

    run_dir = out_dir / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{args.dataset}"
    run_dir.mkdir()
    with (run_dir / "signals.jsonl").open("w", encoding="utf-8") as f:
        for row in table:
            f.write(json.dumps(row) + "\n")

    signals = [s for s in SIGNALS if s in table[0]]
    by_model = {m: [r for r in table if r["model"] == m] for m in eval_models}
    by_model = {m: v for m, v in by_model.items() if len(v) >= args.min_lines // 2}
    arrays = {
        m: ({s: np.array([r[s] for r in v]) for s in signals}, np.array([r["cer"] for r in v]))
        for m, v in by_model.items()
    }
    oracle = {m: metrics_for(-cer, cer) for m, (_, cer) in arrays.items()}

    def cell_means(idx_by_model: dict[str, np.ndarray] | None) -> dict[str, dict[str, float]]:
        res: dict[str, dict[str, float]] = {}
        for s in signals:
            per = []
            for m, (sig, cer) in arrays.items():
                ix = slice(None) if idx_by_model is None else idx_by_model[m]
                per.append(metrics_for(sig[s][ix], cer[ix]))
            res[s] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        res["oracle"] = {k: float(np.mean([o[k] for o in oracle.values()])) for k in next(iter(oracle.values()))}
        return res

    point = cell_means(None)
    boots = defaultdict(lambda: defaultdict(list))
    for _ in range(args.boot):
        idx = {m: rng.integers(0, len(cer), len(cer)) for m, (_, cer) in arrays.items()}
        for s, d in cell_means(idx).items():
            for k, v in d.items():
                boots[s][k].append(v)

    metric_names = list(next(iter(point.values())))
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal"] + [x for k in metric_names for x in (k, f"{k}_lo", f"{k}_hi")])
        for s in signals + ["oracle"]:
            cells = []
            for k in metric_names:
                lo, hi = np.percentile(boots[s][k], [2.5, 97.5])
                cells += [round(point[s][k], 4), round(float(lo), 4), round(float(hi), 4)]
            w.writerow([s] + cells)
    with (run_dir / "per_model.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "mean_cer"] + [f"{s}_gain@0.5" for s in signals] + ["oracle_gain@0.5"])
        for m, (sig, cer) in arrays.items():
            w.writerow([m, len(cer), round(float(cer.mean()), 4)]
                       + [round(gain_at(sig[s], cer, 0.5), 4) for s in signals]
                       + [round(oracle[m]["gain@0.5"], 4)])

    print(f"models evaluated: {len(arrays)}; wrote {run_dir}")
    print((run_dir / "summary.csv").read_text())


if __name__ == "__main__":
    main()
