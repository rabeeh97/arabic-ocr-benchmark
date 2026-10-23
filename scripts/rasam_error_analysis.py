"""Character-level error analysis of a RASAM prediction file (results/rasam/{crnn,replication}/*.json).

    PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/rasam_error_analysis.py results/rasam/crnn/A1_w05_s0_best__r1__test.json [--out results/rasam/analysis/NAME.json]

Aligns reference and hypothesis with Levenshtein (code points, raw profile), then reports: operation split, errors by character class,
top substitution / deletion / insertion pairs, the share of substitutions that are one-dot confusions, per-character error rates,
error rate by position in the line, space merges and splits, and the worst lines. Writes a JSON summary and a list of worst lines to inspect.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from arabic_ocr_bench.normalize import normalize

DOT_GROUPS = ["بتثني", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق"]
DOT_OF = {c: i for i, g in enumerate(DOT_GROUPS) for c in g}
HARAKAT = set("ًٌٍَُِّْٰ")
ALEF = set("اأإآٱ")


def cls(c: str) -> str:
    if c == " ":
        return "space"
    if c in HARAKAT:
        return "diacritic"
    if c.isdigit() or c in "٠١٢٣٤٥٦٧٨٩":
        return "digit"
    if "ARABIC" in unicodedata.name(c, "") and c.isalpha():
        return "letter"
    return "other"


def align(ref: str, hyp: str) -> list[tuple[str, str | None, str | None, int]]:
    """Levenshtein ops as (op, ref_char, hyp_char, ref_index); op in ok/sub/del/ins (ins carries the ref index it precedes)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None, i - 1))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1], i))
            j -= 1
    return ops[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--out")
    ap.add_argument("--worst", type=int, default=30)
    a = ap.parse_args()
    d = json.load(open(a.pred))
    ops_total, by_class, ref_count, ref_err = Counter(), defaultdict(Counter), Counter(), Counter()
    subs, dels, inss = Counter(), Counter(), Counter()
    pos_err, pos_n = [0] * 10, [0] * 10
    dot_sub = alef_sub = 0
    merges = splits = 0
    line_cer, words_single, words_wrong = [], 0, 0
    for l in d["lines"]:
        # the benchmark metric (jiwer) strips edge whitespace; the CRNN often emits a leading/trailing space, so do the same here
        ref, hyp = normalize(l["ref"], "raw").strip(), normalize(l["hyp"], "raw").strip()
        ops = align(ref, hyp)
        nerr = 0
        for op, rc, hc, idx in ops:
            if rc is not None:
                ref_count[rc] += 1
                pos_n[min(9, idx * 10 // max(1, len(ref)))] += 1
            if op == "ok":
                continue
            nerr += 1
            ops_total[op] += 1
            c = rc if rc is not None else hc
            by_class[cls(c)][op] += 1
            if rc is not None:
                ref_err[rc] += 1
                pos_err[min(9, idx * 10 // max(1, len(ref)))] += 1
            if op == "sub":
                subs[(rc, hc)] += 1
                if rc in DOT_OF and hc in DOT_OF and DOT_OF[rc] == DOT_OF[hc]:
                    dot_sub += 1
                if rc in ALEF and hc in ALEF:
                    alef_sub += 1
            elif op == "del":
                dels[rc] += 1
                merges += rc == " "
            else:
                inss[hc] += 1
                splits += hc == " "
        rw, hw = ref.split(), hyp.split()
        line_cer.append((nerr / max(1, len(ref)), l["id"], l["manuscript"], l["region"], ref, hyp))
        # crude word view: words in the reference that never appear in the hypothesis, and how many of those are one edit away
        hset = set(hw)
        for w in rw:
            if w not in hset:
                words_wrong += 1
                if any(abs(len(w) - len(h)) <= 1 and sum(x != y for x, y in zip(w, h)) + abs(len(w) - len(h)) <= 1 for h in hw):
                    words_single += 1
    tot_ref = sum(ref_count.values())
    tot_err = sum(ops_total.values())
    nsub = ops_total["sub"]
    print(f"{Path(a.pred).name}: {tot_err} edit ops over {tot_ref} reference chars = {tot_err / tot_ref:.4f}")
    print("  ops:", {k: f"{v} ({v / tot_err:.0%})" for k, v in ops_total.items()})
    print("  by class (errors / share of all errors):", {c: f"{sum(v.values())} ({sum(v.values()) / tot_err:.0%})" for c, v in by_class.items()})
    print("  class base rates in reference:", {c: f"{sum(n for ch, n in ref_count.items() if cls(ch) == c) / tot_ref:.1%}" for c in by_class})
    print(f"  substitutions that are one-dot confusions: {dot_sub} ({dot_sub / max(1, nsub):.0%} of substitutions); alef-variant swaps: {alef_sub} ({alef_sub / max(1, nsub):.0%})")
    print(f"  space errors: {merges} merged words (space deleted), {splits} split words (space inserted)")
    print("  top substitutions:", ", ".join(f"{r}->{h} {n}" for (r, h), n in subs.most_common(20)))
    print("  top deletions:", ", ".join(f"{c!r} {n}" for c, n in dels.most_common(10)))
    print("  top insertions:", ", ".join(f"{c!r} {n}" for c, n in inss.most_common(10)))
    rates = sorted(((ref_err[c] / n, c, n) for c, n in ref_count.items() if n >= 40), reverse=True)
    print("  highest per-character error rates (n>=40):", ", ".join(f"{c} {r:.0%} (n={n})" for r, c, n in rates[:14]))
    print("  lowest:", ", ".join(f"{c} {r:.0%}" for r, c, n in rates[-8:]))
    print("  error rate by position-in-line decile (start..end of the reference string):", " ".join(f"{e / max(1, n):.3f}" for e, n in zip(pos_err, pos_n)))
    print(f"  reference words missing from the reading: {words_wrong}; of those within one edit of some reading word: {words_single} ({words_single / max(1, words_wrong):.0%})")
    line_cer.sort(reverse=True)
    n_hi = sum(1 for x in line_cer if x[0] > 0.5)
    print(f"  lines with CER > 0.5: {n_hi}/{len(line_cer)}; lines with CER = 0: {sum(1 for x in line_cer if x[0] == 0)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "file": a.pred, "ops": dict(ops_total), "by_class": {c: dict(v) for c, v in by_class.items()},
            "dot_substitution_share": dot_sub / max(1, nsub), "space_merges": merges, "space_splits": splits,
            "top_substitutions": [[r, h, n] for (r, h), n in subs.most_common(60)],
            "top_deletions": dels.most_common(30), "top_insertions": inss.most_common(30),
            "char_error_rates": [[c, r, n] for r, c, n in rates[:60]], "position_decile_rates": [e / max(1, n) for e, n in zip(pos_err, pos_n)],
            "worst_lines": [{"cer": c, "id": i, "manuscript": m, "region": rg, "ref": r, "hyp": h} for c, i, m, rg, r, h in line_cer[: a.worst]],
        }, ensure_ascii=False, indent=1))
        print("  wrote", a.out)


if __name__ == "__main__":
    main()
