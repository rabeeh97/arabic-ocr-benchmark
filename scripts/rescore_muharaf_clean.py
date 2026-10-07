"""Re-score `muharaf_lines` (Muharaf test) without the lines CER-HV flags as
label errors (task X2; ``alazzawi2026labelerr``).

CER-HV (github.com/SanaNGU/CER-HV, pinned below) publishes human-verified
review logs per split: ``error-analysis/muharaf/hitl_labels_log_<split>.csv``.
Every reviewed line carries a label; everything except "Valid but hard" is a
label error the paper removes before retraining (wrong transcription,
segmentation, orientation, script mismatch, signature / non-text). Their
``test_NNNNNN`` IDs are our ``muharaf_lines/test/NNNNNN`` sample IDs (all 218
reviewed test lines match our GT exactly).

The repository has no licence file, so nothing is vendored: the logs are
downloaded to ``~/.cache/arabic_ocr/cerhv/<commit>/`` on first use.

    PYTHONPATH=src .venv/bin/python scripts/rescore_muharaf_clean.py

Writes ``results/characterize/muharaf_clean_rescore.json`` and prints a table.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from arabic_ocr_bench.characterize.runs import load_canonical
from arabic_ocr_bench.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a39544d7794ea4481060c3a2699b7d57343260cb"
URL = ("https://raw.githubusercontent.com/SanaNGU/CER-HV/" + COMMIT
       + "/error-analysis/muharaf/hitl_labels_log_{split}.csv")
CACHE = Path.home() / ".cache/arabic_ocr/cerhv" / COMMIT
VALID = "valid but hard"
MIN_N = 100


def review_log(split: str) -> list[dict]:
    path = CACHE / f"{split}.csv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL.format(split=split), timeout=60) as r:
            path.write_bytes(r.read())
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def flagged(split: str) -> dict[str, str]:
    """{our sample_id -> CER-HV label} for every line judged a label error."""
    out = {}
    for row in review_log(split):
        label = (row.get("LabelNorm") or row["Label"]).strip()
        if label.lower() != VALID:
            out[f"muharaf_lines/{split}/{row['ImageID'].split('_')[-1]}"] = label
    return out


def micro_cer(pairs) -> float:
    e = n = 0
    for gt, pred in pairs:
        g, p = normalize(gt, "light"), normalize(pred, "light")
        e += Levenshtein.distance(g, p)
        n += len(g)
    return e / n if n else 0.0


def main() -> None:
    bad = flagged("test")
    train_bad = flagged("train")
    canon = load_canonical(ROOT)
    rows = []
    for (model, ds), c in sorted(canon.items()):
        if ds != "muharaf_lines":
            continue
        rd = [r for r in c["readings"].values() if not r.error]
        if len(rd) < MIN_N:
            continue
        clean = [r for r in rd if r.sample_id not in bad]
        rows.append({
            "model": model, "n": len(rd), "n_clean": len(clean),
            "cer_light": micro_cer((r.gt, r.pred) for r in rd),
            "cer_light_clean": micro_cer((r.gt, r.pred) for r in clean),
        })
    rows.sort(key=lambda r: r["cer_light_clean"])
    for i, r in enumerate(sorted(rows, key=lambda r: r["cer_light"])):
        r["rank"] = i + 1
    for i, r in enumerate(rows):
        r["rank_clean"] = i + 1
    labels: dict[str, int] = {}
    for v in bad.values():
        labels[v] = labels.get(v, 0) + 1
    doc = {"source": f"github.com/SanaNGU/CER-HV@{COMMIT}",
           "flagged_test": len(bad), "flagged_train": len(train_bad),
           "flagged_test_labels": labels, "rows": rows}
    out = ROOT / "results/characterize/muharaf_clean_rescore.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"flagged: test {len(bad)}, train {len(train_bad)}  ->  {out.relative_to(ROOT)}")
    print(f"{'model':42s} {'n':>5s} {'CER':>7s} {'clean':>7s} {'Δ':>6s} rank")
    for r in rows:
        print(f"{r['model']:42s} {r['n']:5d} {100 * r['cer_light']:6.1f}% {100 * r['cer_light_clean']:6.1f}% "
              f"{100 * (r['cer_light_clean'] - r['cer_light']):+5.1f} {r['rank']:>3d}→{r['rank_clean']}")


if __name__ == "__main__":
    main()
