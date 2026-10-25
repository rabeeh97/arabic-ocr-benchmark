"""Rescore N-best lists with a pretrained Arabic LM: tune the weight on VALIDATION N-best lists, apply it to the test N-best lists.

    PYTHONPATH=src python scripts/rasam_nbest_rescore.py --nbest-dir results/rasam/lm --scores-dir SCORES_DIR --name NB_TM_r1 --tag r1

Score of a candidate = beam score (CTC acoustic + alpha * char 8-gram + beta * length, from rasam_lm_decode.py --nbest)
                       + w * (pretrained-LM log-prob) + v * (number of LM tokens)
w, v are chosen on the validation lines only (grid); the test result is reported for the chosen pair and, for reference, the 1-best and the oracle.
The pretrained LM can only choose among the beam's readings, so it cannot invent text that the image does not support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize


def cer(pairs) -> float:
    e = n = 0
    for ref, hyp in pairs:
        c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp, "raw").strip())
        e += c["sub"] + c["del"] + c["ins"]
        n += c["ref_len"]
    return e / max(1, n) * 100


def pick(lines, scores, w, v):
    out = []
    for line, sc in zip(lines, scores):
        cands = line.get("nbest") or [[line.get("hyp", ""), 0.0]]
        best = max(range(len(cands)), key=lambda k: cands[k][1] + (w * sc[k][0] + v * sc[k][1] if k < len(sc) else -1e9))
        out.append((line["ref"], cands[best][0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest-dir", type=Path, default=Path("results/rasam/lm"))
    ap.add_argument("--scores-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    val = json.loads((a.nbest_dir / f"{a.name}__{a.tag}__val_nbest.json").read_text())["lines"]
    test = json.loads((a.nbest_dir / f"{a.name}__{a.tag}__test.json").read_text())["lines"]
    sv = json.loads((a.scores_dir / f"preds_lmscore_{a.name}__{a.tag}__val_nbest.json").read_text())["scores"]
    st = json.loads((a.scores_dir / f"preds_lmscore_{a.name}__{a.tag}__test.json").read_text())["scores"]
    grid = [(w, v) for w in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8) for v in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    res = sorted((cer(pick(val, sv, w, v)), w, v) for w, v in grid)
    vbest, w, v = res[0]
    base_val = cer(pick(val, sv, 0.0, 0.0))
    one_best = cer([(l["ref"], l["hyp"]) for l in test])
    resc = cer(pick(test, st, w, v))
    oracle = cer([(l["ref"], min((c[0] for c in l.get("nbest", [[l["hyp"], 0]])), key=lambda h: char_edit_counts(normalize(l["ref"], "raw"), normalize(h, "raw"))["sub"]
                                  + char_edit_counts(normalize(l["ref"], "raw"), normalize(h, "raw"))["del"] + char_edit_counts(normalize(l["ref"], "raw"), normalize(h, "raw"))["ins"]))
                  for l in test])
    print(f"{a.name} {a.tag}: validation {base_val:.2f} -> {vbest:.2f} % with w={w}, v={v};  TEST 1-best {one_best:.2f} %  rescored {resc:.2f} %  (oracle in the 10-best {oracle:.2f} %)")
    Path("results/rasam/lm").joinpath(f"{a.name}__{a.tag}__rescored.json").write_text(json.dumps(
        {"name": a.name, "tag": a.tag, "w": w, "v": v, "val_before": base_val, "val_after": vbest, "test_1best": one_best, "test_rescored": resc, "test_oracle_10best": oracle}, indent=1))


if __name__ == "__main__":
    main()
