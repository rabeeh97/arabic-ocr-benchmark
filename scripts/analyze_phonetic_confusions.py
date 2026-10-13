"""Do systems confuse letters that sound alike, or letters that look alike? (makhārij question, 2026-10-13)

    .geomvenv/bin/python scripts/analyze_phonetic_confusions.py [--store results/store] [--out results/analysis/B/phonetic.json]

Every letter substitution in the store (letters view, analyze_letters.outcomes) is put in one class:
  look-only   same dotless skeleton in that position (rasm), not a sound pair: e.g. ب/ن/ي, ج/خ, ر/ز, س/ش, ع/غ
  sound-only  a pair that shares a place of articulation (makhraj) or that dialects merge, with different skeletons:
              e.g. ث/س, ذ/ز, ظ/ض, ق/ء, ق/ك, ص/س, ط/ت, ض/د
  both        same skeleton and a sound pair: ت/ث, د/ذ, ص/ض, ط/ظ, ح/خ
  other       everything else
Chance baseline: the share of each class among all (GT letter, other letter) pairs weighted by the GT letter
frequency, so a system with no preference would land on it. A sound-only share well above chance means the errors
carry pronunciation (a language prior or the writer's own spelling), not only vision.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from analyze_letters import outcomes  # noqa: E402
from arabic_ocr_bench.characterize.text import ALPHABET, rasm_of  # noqa: E402

# places of articulation, simplified from the classical makhārij (labial, interdental, dental-alveolar stops, sibilants,
# sonorants, palatal, velar-uvular, pharyngeal, glottal)
PLACE = {**dict.fromkeys("بمفو", "labial"), **dict.fromkeys("ثذظ", "interdental"), **dict.fromkeys("تدطض", "dental"),
         **dict.fromkeys("سزص", "sibilant"), **dict.fromkeys("نلر", "sonorant"), **dict.fromkeys("جشي", "palatal"),
         **dict.fromkeys("كقخغ", "velar"), **dict.fromkeys("عح", "pharyngeal"), **dict.fromkeys("ءهأإآ", "glottal")}
MERGERS = {frozenset(p) for p in ("ثس", "ثت", "ذز", "ذد", "ظض", "ظز", "قء", "قأ", "قك", "صس", "طت", "ضد", "هح", "عء")}


def sound_pair(a: str, b: str) -> bool:
    return frozenset((a, b)) in MERGERS or (a in PLACE and PLACE.get(a) == PLACE.get(b))


def look_pair(a: str, b: str, form: str) -> bool:
    ra, rb = rasm_of(a, form), rasm_of(b, form)
    return ra is not None and ra == rb


def klass(a: str, b: str, form: str) -> str:
    s, v = sound_pair(a, b), look_pair(a, b, form)
    return "both" if s and v else "sound-only" if s else "look-only" if v else "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/B/phonetic.json")
    a = ap.parse_args()
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    runs = defaultdict(list)
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        if r.get("in_train") == "yes" or r["transform"] != "none" or r["dataset"] not in ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines"):
            continue
        if any(x in r["run_id"] for x in ("ain", "paddleocr", "trocr")):
            continue
        if r["pack"] not in ("panel-2026-10-13", "bench", "eval_data+heldout"):
            continue
        runs[(r["run_id"], r["dataset"])].append(r)
    res = {}
    chance_n = Counter()
    for (run, ds), rows in sorted(runs.items()):
        if len(rows) < 150:
            continue
        c, pairs = Counter(), Counter()
        for r in rows:
            for o in outcomes(S[r["sample_key"]]["gt_raw"], r["hyp"])[0]:
                if run == "crnn/base" and o["ch"] in ALPHABET:
                    for b in ALPHABET:
                        if b != o["ch"]:
                            chance_n[klass(o["ch"], b, o["form"])] += 1
                if o["oc"] == "sub" and o["read"] and o["read"] != o["ch"]:
                    k = klass(o["ch"], o["read"], o["form"])
                    c[k] += 1
                    if k == "sound-only":
                        pairs[o["ch"] + "→" + o["read"]] += 1
        n = sum(c.values())
        if n < 100:
            continue
        res[f"{run}|{ds}"] = {"subs": n, **{k: c[k] / n for k in ("look-only", "sound-only", "both", "other")}, "top_sound_only": pairs.most_common(6)}
    tot = sum(chance_n.values())
    chance = {k: chance_n[k] / tot for k in ("look-only", "sound-only", "both", "other")}
    a.out.write_text(json.dumps({"chance": chance, "runs": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("chance (uniform confusion):", {k: f"{100 * v:.1f}%" for k, v in chance.items()})
    print(f"{'run|dataset':58s} {'subs':>6s} {'look':>6s} {'sound':>6s} {'both':>6s} {'other':>6s}  top sound-only")
    for k, v in res.items():
        print(f"{k[:58]:58s} {v['subs']:6d} {100 * v['look-only']:6.1f} {100 * v['sound-only']:6.1f} {100 * v['both']:6.1f} {100 * v['other']:6.1f}  "
              + " ".join(f"{p}:{n}" for p, n in v["top_sound_only"][:4]))


if __name__ == "__main__":
    main()
