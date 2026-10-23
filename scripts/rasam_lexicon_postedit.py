"""Offline test of two things on a RASAM prediction file, using TRAIN labels only (no test text is used to build anything):

1. Lexicon snapping: a hypothesis word that is not in the training lexicon is replaced by the most frequent training word within
   MAX_DIST edits (unique best only), as an approximation of what a word-class model gets for free on a closed lexicon.
2. Misaligned crops: for lines with CER > 0.5, does the reading match the reference of ANOTHER test line (the crop contained a
   neighbouring line) better than its own?

    PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/rasam_lexicon_postedit.py results/rasam/crnn/A1_w05_s0_best__r1__test.json RASAM-1
"""
from __future__ import annotations

import glob
import json
import sys
import unicodedata
from collections import Counter

import pyarrow.parquet as pq
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"


def micro(pairs) -> float:
    e = n = 0
    for ref, hyp in pairs:
        c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp, "raw").strip())
        e += c["sub"] + c["del"] + c["ins"]
        n += c["ref_len"]
    return e / n


def main() -> None:
    pred, dataset = sys.argv[1], sys.argv[2]
    d = json.load(open(pred))
    lex = Counter()
    for split in ("train", "validation"):
        for f in sorted(glob.glob(f"{ROOT}{dataset}/snapshots/*/data/{split}-*.parquet")):
            for r in pq.read_table(f, columns=["transcription"]).to_pylist():
                lex.update(normalize(unicodedata.normalize("NFC", r["transcription"] or ""), "raw").split())
    words = list(lex)
    print(f"lexicon from {dataset} train+validation: {len(words)} word types, {sum(lex.values())} tokens")
    base = micro([(l["ref"], l["hyp"]) for l in d["lines"]])
    test_words = [w for l in d["lines"] for w in normalize(l["ref"], "raw").split()]
    oov = sum(w not in lex for w in test_words) / len(test_words)
    print(f"test reference words not in the training lexicon (OOV): {oov:.1%}; baseline micro CER {base:.4f}")
    cache = {}
    for max_dist in (1, 2):
        for min_freq in (1, 2):
            cache.clear()
            changed = good = bad = 0
            out = []
            for l in d["lines"]:
                new = []
                for w in normalize(l["hyp"], "raw").split():
                    if w in lex or len(w) < 3:
                        new.append(w)
                        continue
                    if w not in cache:
                        cands = [(x, s) for x, s, _ in process.extract(w, words, scorer=Levenshtein.distance, score_cutoff=max_dist, limit=8)
                                 if lex[x] >= min_freq]
                        if cands:
                            best = min(s for _, s in cands)
                            top = sorted((-lex[x], x) for x, s in cands if s == best)
                            cache[w] = top[0][1] if len(top) == 1 or top[0][0] != top[1][0] else None
                        else:
                            cache[w] = None
                    new.append(cache[w] or w)
                    if cache[w] and cache[w] != w:
                        changed += 1
                out.append((l["ref"], " ".join(new)))
            res = micro(out)
            # words changed that turned a wrong word into a right one, and vice versa (word-level against the reference line)
            for l, (_, h2) in zip(d["lines"], out):
                rset = set(normalize(l["ref"], "raw").split())
                h1 = normalize(l["hyp"], "raw").split()
                h2w = h2.split()
                if len(h1) == len(h2w):
                    for a, b in zip(h1, h2w):
                        if a != b:
                            good += (b in rset) and (a not in rset)
                            bad += (a in rset) and (b not in rset)
            print(f"  snap within {max_dist} edit(s), min lexicon frequency {min_freq}: micro CER {res:.4f} ({res - base:+.4f}); "
                  f"{changed} words changed ({good} wrong->right, {bad} right->wrong)")
    # misaligned crops
    refs = [normalize(l["ref"], "raw").strip() for l in d["lines"]]
    hard = misaligned = 0
    ex = []
    for i, l in enumerate(d["lines"]):
        own = Levenshtein.normalized_distance(refs[i], normalize(l["hyp"], "raw").strip())
        if own <= 0.5:
            continue
        hard += 1
        same_page = [j for j, m in enumerate(d["lines"]) if m["page"] == l["page"] and j != i]
        best = min(((Levenshtein.normalized_distance(refs[j], normalize(l["hyp"], "raw").strip()), j) for j in same_page), default=(1, None))
        if best[0] < own - 0.15 and best[0] < 0.35:
            misaligned += 1
            ex.append((l["id"], round(own, 2), d["lines"][best[1]]["id"], round(best[0], 2)))
    print(f"lines with normalised edit distance > 0.5: {hard}; of those the reading matches another line on the same page much better "
          f"(crop shows a neighbouring line): {misaligned}")
    print("  examples (line, own distance, matching line, its distance):", ex[:6])


if __name__ == "__main__":
    main()
