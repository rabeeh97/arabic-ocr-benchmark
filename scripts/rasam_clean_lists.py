"""Decide which RASAM-1/2 / TariMa lines to drop, with the same automatic rules on every split (no manual verdicts).

    PYTHONPATH=src .venv/bin/python scripts/rasam_clean_lists.py --pack rasam_tm_h96_rep2.pkl --reads A.json B.json ... \
        [--folds results/rasam/clean/folds.json] [--out results/rasam/clean]

Inputs
  pack    a rasam_prepare_crnn_data.py pack built with the crop repairs (rasam_recut_broken.py --crop-check): broken crops are already
          FIXED there (re-cut from the page images), so what is left to decide is what to DROP.
  reads   cross-fitted readings (colab_remote_readcf.py) {key: {"hyp", "fold_model"?}}: every line is read by a model that never saw
          a line of its page (train/validation: the other fold's model; test: the fold model that was not trained on the page).
  external  results/rasam/clean/external/khiro45_rassam_1_data_validation_full.csv: a manual check of 4,649 RASAM-1 lines
          (calfa-co/rasam-dataset PR #7); lines marked invalid are dropped.
Rules (a line is dropped if any fires; each is recorded)
  vertical   crop at least 1.8 x taller than wide (vertical margin notes; a horizontal recogniser cannot read them)
  blank      almost no ink (< 1 % dark pixels) while the reference has text the reading misses (CER >= 50)
  garbled    reference >= 15 chars, >= 1.0 bit/char less plausible than the reading under the OpenITI char LM, and >= 25 % CER apart
             (not MS 1936, whose Berber text is legitimately implausible for an Arabic LM)
  garbled_page  on a page (>= 5 lines, not MS 1936) where the median line reference is >= 0.6 bit/char less plausible than its reading
             (uncorrected HTR pages), a line >= 0.3 bit/char less plausible and >= 15 % CER apart
  truncated  the reading covers the reference (sub + del <= 15 %) and adds >= 15 % + 5 characters
  doubled    a >= 15-character span of the reference (spaces ignored) recurs right after itself, the reading does not repeat it and
             is shorter by about that span (formulaic repeats in legal texts are read twice and do not fire)
  shifted    reference >= 15 chars, >= 30 % CER from its own reading but < 30 % and < half that from the reading of another line on the page
  external   marked invalid in the manual check above
Writes OUT/drop.json (sorted keys) and OUT/clean_report.json (counts per rule / set / split, agreement with the manual check).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from big_char_lm import BigCharLM  # noqa: E402
from rasam_data_audit import BIG, bpc, repeated_span  # noqa: E402
from rasam_recut_broken import norm  # noqa: E402
from arabic_ocr_bench.metrics import char_edit_counts  # noqa: E402

EXT = Path("results/rasam/clean/external/khiro45_rassam_1_data_validation_full.csv")


def cer(ref: str, hyp: str) -> float:
    k = char_edit_counts(ref, hyp)
    return 100 * (k["sub"] + k["del"] + k["ins"]) / max(1, k["ref_len"])


def ink(img: np.ndarray) -> float:
    g = img.astype(np.float32)
    return float((g < 0.6 * np.median(g)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--reads", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/rasam/clean"))
    a = ap.parse_args()
    rows = pickle.load(open(a.pack, "rb"))["rows"]
    reads = {}
    for p in a.reads:
        reads.update(json.loads(p.read_text()))
    lm = BigCharLM(Path(BIG))
    by_page = defaultdict(list)
    for r in rows:
        if r["key"] in reads:
            by_page[(r["manuscript"], r["page"])].append(r["key"])
    # the manual check: rows are '<ms>_<page>_<xml line>_flat.png' + text; match to our lines by page and text
    ext_bad, ext_seen = set(), set()
    if EXT.exists():
        idx = defaultdict(list)
        for r in rows:
            idx[(f"{r['manuscript']}_{r['page']}", norm(r["label"]))].append(r["key"])
        for e in csv.DictReader(open(EXT, encoding="utf-8")):
            m = re.match(r"(.+_\d+)_l_", e["image_path"])
            keys = idx.get((m.group(1), norm(e["text"]))) if m else None
            if keys:
                ext_seen.update(keys)
                if e["valid"] != "True":
                    ext_bad.update(keys)
    # page-level: an uncorrected-HTR page has references that are less plausible than the readings on most of its lines
    gap = {}
    for r in rows:
        rd = reads.get(r["key"])
        if rd is not None and r["manuscript"] != "BULAC_MS_ARA_1936" and len(r["label"]) >= 10:
            gap[r["key"]] = bpc(lm, r["label"]) - bpc(lm, rd["hyp"])
    bad_pages = set()
    for pg, ks in by_page.items():
        g = [gap[k] for k in ks if k in gap]
        if len(g) >= 5 and float(np.median(g)) >= 0.6:
            bad_pages.add(pg)
    print("garbled pages:", sorted(f"{m}|{p}" for m, p in bad_pages))
    reasons = {}
    for r in rows:
        k, ref = r["key"], r["label"]
        why = []
        h, w = r["img"].shape
        if h >= 1.8 * w:
            why.append("vertical")
        if k in ext_bad:
            why.append("external")
        rd = reads.get(k)
        if rd is not None:
            hyp = rd["hyp"]
            c = cer(ref, hyp)
            if ink(r["img"]) < 0.01 and c >= 50:
                why.append("blank")
            if r["manuscript"] != "BULAC_MS_ARA_1936" and len(ref) >= 15 and c >= 25 and bpc(lm, ref) - bpc(lm, hyp) >= 1.0:
                why.append("garbled")
            elif (r["manuscript"], r["page"]) in bad_pages and gap.get(k, 0) >= 0.3 and c >= 15:
                why.append("garbled_page")
            e = char_edit_counts(ref, hyp)
            if (e["sub"] + e["del"]) <= 0.15 * max(1, e["ref_len"]) and e["ins"] >= 0.15 * len(ref) + 5:
                why.append("truncated")
            rs, hs = ref.replace(" ", ""), hyp.replace(" ", "")
            rp = repeated_span(rs, n=15)
            if rp and hs.count(rp) < 2 and len(hs) <= len(rs) - 0.6 * len(rp):  # the reading is shorter by about the repeat
                why.append("doubled")
            if c >= 30 and len(ref) >= 15:
                others = [cer(ref, reads[o]["hyp"]) for o in by_page[(r["manuscript"], r["page"])] if o != k]
                if others and min(others) < min(30.0, 0.5 * c):
                    why.append("shifted")
        if why:
            reasons[k] = why
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "drop.json").write_text(json.dumps(sorted(reasons), indent=0))
    tab = defaultdict(Counter)
    tot = Counter()
    for r in rows:
        grp = f"{r['dataset']}/{r['orig_split']}"
        tot[grp] += 1
        for x in reasons.get(r["key"], []):
            tab[grp][x] += 1
        if r["key"] in reasons:
            tab[grp]["ANY"] += 1
    # agreement of the automatic rules with the manual check, on the lines it covers
    auto = {k for k, v in reasons.items() if set(v) - {"external"}}
    tp = len(auto & ext_bad & ext_seen)
    agree = {"manual_lines_matched": len(ext_seen), "manual_invalid": len(ext_bad), "auto_flagged_in_matched": len(auto & ext_seen),
             "both": tp, "auto_precision_vs_manual": round(tp / max(1, len(auto & ext_seen)), 3),
             "auto_recall_vs_manual": round(tp / max(1, len(ext_bad)), 3)}
    rep = {"rules": "see scripts/rasam_clean_lists.py", "lines": dict(tot), "dropped": {g: dict(c) for g, c in sorted(tab.items())},
           "agreement_with_manual_check": agree, "reasons": reasons}
    (a.out / "clean_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for g in sorted(tot):
        print(g, tot[g], dict(tab[g]))
    print("agreement with the manual check:", agree)
    print("->", a.out / "drop.json", len(reasons), "lines")


if __name__ == "__main__":
    main()
