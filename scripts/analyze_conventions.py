"""Transcription conventions, writers' dialect spellings, and NAKBA's train/test text leak (themes T0, T4; 2026-10-15).

    .geomvenv/bin/python scripts/analyze_conventions.py [--out results/analysis/B/conventions.json]

1. Convention fingerprint per label set (the six test sets and the two training pools): hamza on alif, tāʾ marbūṭa among
   word-final ه/ة, alif maqṣūra among word-final ى/ي, digit systems, harakat per 100 letters, tatweel, double spaces, a
   space before punctuation.
2. Dialect-merger spellings: for each merger (ث→ت, ذ→د, ذ→ز, ظ→ض, ظ→ز, ق→ء), word pairs that differ only by that letter
   and both occur in the pooled vocabulary; per set, how many tokens use the merged spelling of such a pair (e.g. متل for
   مثل), with the top pairs. These are writers spelling as they speak; a model that reads the page faithfully is then
   "wrong" against a standard-spelling label, and vice versa.
3. NAKBA leak: NAKBA test / blind lines whose cleaned text occurs in NAKBA train, and every system's CER on leaked vs
   clean lines; NAKBA-trained systems benefit if they memorised.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from analyze_letters import words_view  # noqa: E402
from geom_inkcrop_export import clean  # noqa: E402

MERGERS = [("ث", "ت"), ("ذ", "د"), ("ذ", "ز"), ("ظ", "ض"), ("ظ", "ز"), ("ق", "ء")]
HARAKAT = re.compile("[ً-ْٰ]")
LETTER = re.compile("[ء-ي]")


def fingerprint(lines: list[str]) -> dict:
    words = [w for ln in lines for w in ln.split()]
    alif = [c for ln in lines for c in ln if c in "اأإآ"]
    fin_h = [w[-1] for w in words if w and w[-1] in "هة"]
    fin_y = [w[-1] for w in words if w and w[-1] in "يى"]
    digits = Counter("western" if c in "0123456789" else "arabic-indic" if "٠" <= c <= "٩" else "persian" for ln in lines for c in ln
                     if c.isdigit() or "٠" <= c <= "٩" or "۰" <= c <= "۹")
    letters = sum(len(LETTER.findall(ln)) for ln in lines)
    nd = sum(digits.values())
    return {"lines": len(lines), "hamza_on_alif": sum(c != "ا" for c in alif) / max(1, len(alif)), "ta_marbuta_share": fin_h.count("ة") / max(1, len(fin_h)),
            "alif_maqsura_share": fin_y.count("ى") / max(1, len(fin_y)), "digits": {k: v / nd for k, v in digits.items()} if nd else {},
            "harakat_per_100_letters": 100 * sum(len(HARAKAT.findall(ln)) for ln in lines) / max(1, letters),
            "tatweel_lines": sum("ـ" in ln for ln in lines) / len(lines), "double_space_lines": sum("  " in ln for ln in lines) / len(lines),
            "space_before_punct": sum(len(re.findall(r" [.,،:؛!؟?]", ln)) for ln in lines) / max(1, sum(len(re.findall(r"[.,،:؛!؟?]", ln)) for ln in lines))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=W / "results/analysis/B/conventions.json")
    a = ap.parse_args()
    S = pq.read_table(W / "results/store/samples.parquet").to_pylist()
    sets = {ds: [s["gt_raw"] for s in S if s["dataset"] == ds] for ds in ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2")}
    pools = {}
    for name, f in (("muharaf_train", "aligner_data_standard.pkl"), ("nakba_train", "aligner_data_nakba.pkl")):
        pools[name] = [r["label"] for r in pickle.load(open(W / "results/panel" / f, "rb"))["rows"] if r["role"] == "train"]
    fp = {k: fingerprint(v) for k, v in {**sets, **pools}.items()}

    vocab = Counter(w for v in {**sets, **pools}.values() for ln in v for w in words_view(ln))
    merged = {}
    for k, v in {**sets, **pools}.items():
        toks = Counter(w for ln in v for w in words_view(ln))
        hits, pairs = 0, Counter()
        for std, dia in MERGERS:
            for w, n in toks.items():
                if dia not in w:
                    continue
                for i, ch in enumerate(w):
                    if ch == dia:
                        s = w[:i] + std + w[i + 1:]
                        if vocab[s] >= 3 and vocab[s] >= vocab[w]:  # the standard spelling is a real, at least as common word
                            hits += n
                            pairs[f"{w}←{s}"] += n
        merged[k] = {"tokens": sum(toks.values()), "merged_spelling_tokens": hits, "per_1000_tokens": 1000 * hits / max(1, sum(toks.values())),
                     "top_pairs": pairs.most_common(8)}

    train_text = {clean(t) for t in pools["nakba_train"]}
    leak = {}
    keyset = {}
    for s in S:
        if s["dataset"] in ("nakba_lines", "nakba_blind"):
            g = clean(s["gt_raw"])
            keyset[s["sample_key"]] = "leaked, ≥ 20 chars" if g in train_text and len(g) >= 20 else "leaked, short" if g in train_text else "clean"
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for r in pq.read_table(W / "results/store/lines.parquet").to_pylist():
        if r["sample_key"] not in keyset or r["transform"] != "none" or r["pack"] not in ("panel-2026-10-13", "bench") or "paddleocr" in r["run_id"]:
            continue
        c = acc[(r["run_id"], r["dataset"])][keyset[r["sample_key"]]]
        c[0] += r["e_no_diacritics"]; c[1] += max(1, r["n_no_diacritics"]); c[2] += 1
    for (run, ds), d in acc.items():
        if d["clean"][2] < 50 or (d["leaked, ≥ 20 chars"][2] + d["leaked, short"][2]) < 10:
            continue
        leak[f"{run}|{ds}"] = {k: {"lines": v[2], "cer": v[0] / v[1]} for k, v in d.items()}
    share = Counter(keyset.values())
    res = {"fingerprint": fp, "merged_spellings": merged, "nakba_leak_lines": dict(share), "nakba_leak_by_run": leak}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'set':14s} {'hamza':>6s} {'ة':>6s} {'ى':>6s} {'harakat':>8s} {'tatweel':>8s} {'2-space':>8s} {'sp.punct':>8s}  digits")
    for k, v in fp.items():
        print(f"{k:14s} {100 * v['hamza_on_alif']:5.1f}% {100 * v['ta_marbuta_share']:5.1f}% {100 * v['alif_maqsura_share']:5.1f}% {v['harakat_per_100_letters']:8.2f} "
              f"{100 * v['tatweel_lines']:7.1f}% {100 * v['double_space_lines']:7.1f}% {100 * v['space_before_punct']:7.1f}%  " + ", ".join(f"{d} {100 * x:.0f}%" for d, x in v["digits"].items()))
    print("\nmerged (dialect) spellings per 1,000 tokens, top pairs")
    for k, v in merged.items():
        print(f"  {k:14s} {v['per_1000_tokens']:5.2f}  " + "  ".join(f"{p}:{n}" for p, n in v["top_pairs"][:5]))
    print("\nNAKBA leak lines:", dict(share))
    for k, v in sorted(leak.items(), key=lambda t: t[0]):
        if any(x in k for x in ("nakba", "baseer", "crnn/base|", "crnn/std_s0", "qwen3-vl-8b", "suite-v1-qwen3vl8b")):
            print(f"  {k[:58]:58s} " + "  ".join(f"{kk}: {100 * vv['cer']:.1f}% ({vv['lines']})" for kk, vv in v.items()))


if __name__ == "__main__":
    main()
