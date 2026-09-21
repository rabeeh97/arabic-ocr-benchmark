"""Robustness of the cross-model-agreement confidence signal.

Reuses the loaders from experiment_confidence_baselines.py. For each evaluated model, re-computes
the agreement signal using only a chosen subset of the *other* models' outputs, then reports the
within-model Spearman(signal, -CER) and the CER gain from keeping the top 50% of lines.

Variants:
  all            every other model (the setting in the main comparison)
  no_family      drop other models from the same family (base/ft variants share a keyword)
  k=<n>          a random sample of n other models (averaged over --draws draws)
  no_family+k=<n>  both

    PYTHONPATH=src python scripts/experiment_agreement_ablation.py --dataset kitab_khatt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from arabic_ocr_bench.metrics import score_pair  # noqa: E402
from experiment_confidence_baselines import gain_at, load_latest, spearman  # noqa: E402
from experiment_gliner2_verifier import normalize_for_model  # noqa: E402

FAMILIES = ["paligemma", "qwen", "gemini", "pixtral", "minimax", "claude", "gpt", "trocr",
            "baseer", "qari", "mistral", "llama", "gemma", "nemotron", "glm", "kimi", "deepseek"]


def family(model: str) -> str:
    low = model.lower()
    for f in FAMILIES:
        if f in low:
            return f
    return low


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-lines", type=int, default=100)
    ap.add_argument("--min-others", type=int, default=3)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    args = ap.parse_args()

    rows = load_latest(args.dataset)
    per_model: dict[str, list[str]] = defaultdict(list)
    per_line: dict[str, list[str]] = defaultdict(list)
    for (m, s) in rows:
        per_model[m].append(s)
        per_line[s].append(m)
    eval_models = sorted(m for m, ss in per_model.items() if len(ss) >= args.min_lines)
    norm = {k: normalize_for_model(r.get("cleaned_text") or "") for k, r in rows.items()}
    sim_cache: dict[tuple[str, str, str], float] = {}

    def sim(m: str, o: str, s: str) -> float:
        key = (m, o, s)
        if key not in sim_cache:
            sim_cache[key] = Levenshtein.normalized_similarity(norm[(m, s)], norm[(o, s)])
        return sim_cache[key]

    rng = np.random.default_rng(0)
    variants = ["all", "no_family"] + [f"k={k}" for k in args.ks] + [f"no_family+k={k}" for k in args.ks]
    # per variant, per model: (signal array averaged over draws, cer array, keep-mask of lines)
    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {v: {} for v in variants}
    for m in eval_models:
        lines = per_model[m]
        cer = np.array([score_pair(rows[(m, s)]["ground_truth"], rows[(m, s)].get("cleaned_text") or "",
                                   profiles=("raw",))["cer"]["raw"] for s in lines])
        for v in variants:
            no_fam = v.startswith("no_family")
            k = int(v.split("k=")[1]) if "k=" in v else None
            draws = args.draws if k else 1
            sig = np.zeros(len(lines))
            valid = np.ones(len(lines), dtype=bool)
            for i, s in enumerate(lines):
                pool = [o for o in per_line[s] if o != m and not (no_fam and family(o) == family(m))]
                if len(pool) < max(args.min_others if k is None else k, 1):
                    valid[i] = False
                    continue
                if k is None:
                    sig[i] = np.mean([sim(m, o, s) for o in pool])
                else:
                    sig[i] = np.mean([
                        np.mean([sim(m, o, s) for o in rng.choice(pool, size=k, replace=False)])
                        for _ in range(draws)
                    ])
            results[v][m] = (sig[valid], cer[valid])

    def summarize(v: str, idx: dict[str, np.ndarray] | None) -> tuple[float, float]:
        rhos, gains = [], []
        for m, (sig, cer) in results[v].items():
            if len(cer) < args.min_lines // 2:
                continue
            ix = slice(None) if idx is None else idx[m]
            rhos.append(spearman(sig[ix], -cer[ix]))
            gains.append(gain_at(sig[ix], cer[ix], 0.5))
        return float(np.mean(rhos)), float(np.mean(gains))

    lines_out = []
    print(f"{'variant':18} {'models':>6} {'rho':>7} {'rho 95% CI':>16} {'gain@0.5':>9} {'95% CI':>16}")
    for v in variants:
        used = [m for m, (_, cer) in results[v].items() if len(cer) >= args.min_lines // 2]
        if not used:
            print(f"{v:18} {0:>6}  (not enough other models)")
            continue
        point = summarize(v, None)
        boots = []
        for _ in range(args.boot):
            idx = {m: rng.integers(0, len(results[v][m][1]), len(results[v][m][1])) for m in used}
            boots.append(summarize(v, idx))
        b = np.array(boots)
        r_lo, r_hi = np.percentile(b[:, 0], [2.5, 97.5])
        g_lo, g_hi = np.percentile(b[:, 1], [2.5, 97.5])
        print(f"{v:18} {len(used):>6} {point[0]:7.3f} [{r_lo:6.3f},{r_hi:6.3f}] "
              f"{point[1]:9.4f} [{g_lo:6.4f},{g_hi:6.4f}]")
        lines_out.append({"variant": v, "models": len(used), "rho": point[0], "rho_lo": float(r_lo),
                          "rho_hi": float(r_hi), "gain@0.5": point[1], "gain_lo": float(g_lo),
                          "gain_hi": float(g_hi)})
    out = REPO / "results" / "confidence_baselines" / f"agreement_ablation_{args.dataset}.json"
    out.write_text(json.dumps(lines_out, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
