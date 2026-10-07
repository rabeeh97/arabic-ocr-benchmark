"""Error decomposition over every canonical prediction (characterisation P2).

1. Picks one canonical reading per (model, sample) (R0.3, see
   ``characterize/runs.py``) and writes ``results/characterize/canonical_runs.json``.
2. Per (model, dataset): the normalisation ladder (R2.1), rasm-CER and the dot
   tax (R2.2), the diacritic-mismatch counts (R2.3) and the aligned error
   taxonomy (R2.4). Errored readings are counted, not scored.

    PYTHONPATH=src .venv/bin/python scripts/decompose_errors.py

Writes ``results/characterize/<UTC stamp>/errors.json``; ``scripts/build_atlas.py``
picks up the newest one.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from arabic_ocr_bench.characterize import errors as E
from arabic_ocr_bench.characterize.runs import load_canonical, manifest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_leaderboard import fine_tuned_models, in_train_datasets  # noqa: E402

MIN_N = 20  # pairs below this are kept but flagged
COMPETENT_CER = 0.4  # diac-insensitive CER at or below which a pair counts as competent


def score(readings) -> dict:
    ladders, tax, diac = [], Counter(), Counter()
    inside = paw = base_in = base_paw = 0
    n_err = 0
    for r in readings:
        if r.error:
            n_err += 1
            continue
        ladders.append(E.ladder_edits(r.gt, r.pred))
        g, p = E.ladder_strings(r.gt), E.ladder_strings(r.pred)
        canon = E.RUNGS.index("canonical")
        tax.update(E.classify_ops(g[canon], p[canon]))
        o = E.RUNGS.index("orthographic")
        diac.update(E.diacritic_mismatch(g[o], p[o]))
        a, b = E.is_paw_gap_space(g[canon], p[canon])
        inside += a
        paw += b
        a, b = E.paw_gap_base_rate(g[canon])
        base_in += a
        base_paw += b
    out = {"n_scored": len(ladders), "n_error": n_err}
    if not ladders:
        return out
    out["ladder"] = E.summarize_ladder(ladders)
    total = sum(tax.values()) or 1
    out["taxonomy"] = {c: tax.get(c, 0) for c in E.CATEGORIES}
    out["taxonomy_share"] = {c: tax.get(c, 0) / total for c in E.CATEGORIES}
    gl = diac.get("aligned_letters", 0) or 1
    out["diacritics"] = {
        **dict(diac),
        "inserted_per_100": 100 * diac.get("inserted", 0) / gl,
        "dropped_share": diac.get("dropped", 0) / (diac.get("gt_marks", 0) or 1),
    }
    out["word_internal_spaces"] = {
        "inserted": inside, "at_paw_gap": paw,
        "paw_gap_share": paw / inside if inside else None,
        # chance level: share of GT word-internal letter boundaries that are PAW gaps
        "paw_gap_base_rate": base_paw / base_in if base_in else None,
    }
    return out


def main() -> None:
    canon = load_canonical(ROOT)
    (ROOT / "results/characterize").mkdir(parents=True, exist_ok=True)
    runs_path = ROOT / "results/characterize/canonical_runs.json"
    runs_path.write_text(json.dumps(manifest(canon), ensure_ascii=False, indent=1), encoding="utf-8")

    in_train = in_train_datasets()
    fts = fine_tuned_models({m for m, _ in canon})
    pairs = []
    for (model, ds), c in sorted(canon.items()):
        row = {"model": model, "dataset": ds, "prompt_version": c["prompt_version"],
               "fine_tuned": model in fts,
               "in_train": model in fts and ds in in_train,
               **score(c["readings"].values())}
        row["small"] = row["n_scored"] < MIN_N
        pairs.append(row)
        print(f"{model:42s} {ds:24s} n={row['n_scored']:5d}", file=sys.stderr)

    # pooled per dataset and overall, over pairs that are quotable
    def pool(rows):
        lad = [p["ladder"] for p in rows if "ladder" in p]
        edits = Counter()
        for x in lad:
            edits.update(x["edits"])
        e0 = edits[E.RUNGS[0]] or 1
        share = {E.RUNGS[i]: (edits[E.RUNGS[i - 1]] - edits[E.RUNGS[i]]) / e0
                 for i in range(1, len(E.RUNGS))}
        share["residual"] = edits[E.RUNGS[-1]] / e0
        tax = Counter()
        for p in rows:
            tax.update(p.get("taxonomy", {}))
        t = sum(tax.values()) or 1
        return {"pairs": len(lad), "share": share,
                "taxonomy_share": {c: tax.get(c, 0) / t for c in E.CATEGORIES}}

    usable = [p for p in pairs if not p["in_train"] and not p["small"] and "ladder" in p]
    # pooled numbers are dominated by models that emit garbage (whole-line
    # failures), so the per-dataset view pools only competent pairs
    competent = [p for p in usable if p["ladder"]["cer"]["no_diacritics"] <= COMPETENT_CER]
    by_ds = {ds: pool([p for p in competent if p["dataset"] == ds])
             for ds in sorted({p["dataset"] for p in competent})}

    ins = sum(p["word_internal_spaces"]["inserted"] for p in competent)
    gap = sum(p["word_internal_spaces"]["at_paw_gap"] for p in competent)
    base = [p["word_internal_spaces"]["paw_gap_base_rate"] for p in competent
            if p["word_internal_spaces"]["paw_gap_base_rate"] is not None]
    dot = [p["ladder"]["dot_tax"] / p["ladder"]["cer"]["no_diacritics"]
           for p in competent if p["ladder"]["cer"]["no_diacritics"] > 0]
    headline = {
        "competent_pairs": len(competent), "usable_pairs": len(usable),
        "paw_gap_share": gap / ins if ins else None, "paw_spaces_inserted": ins,
        "paw_gap_base_rate_median": statistics.median(base) if base else None,
        "dot_tax_share_median": statistics.median(dot) if dot else None,
        "dot_tax_median": statistics.median(p["ladder"]["dot_tax"] for p in competent) if competent else None,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results/characterize" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"generated": stamp, "rungs": list(E.RUNGS), "categories": list(E.CATEGORIES),
           "min_n": MIN_N, "competent_cer": COMPETENT_CER, "headline": headline, "pairs": pairs,
           "by_dataset": by_ds, "overall": pool(competent), "overall_all": pool(usable)}
    (out_dir / "errors.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_dir.relative_to(ROOT))


if __name__ == "__main__":
    main()
