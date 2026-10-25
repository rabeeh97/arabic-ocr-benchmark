"""Contamination check before training on extra data: does anything in the extra training data appear in the test set we score?

    python scripts/rasam_overlap_check.py

Compares manuscript names and normalised line texts (NFC, bidi marks removed, whitespace collapsed; also a diacritic-stripped variant) between:
  * TariMa (all splits)            vs  RASAM-1 test, RASAM-2 test
  * RASAM-1 (all splits, incl. its test)  vs  RASAM-2 test      (for training on all of RASAM-1 and scoring RASAM-2)
Prints counts and a few examples; writes results/rasam/overlap_check.json.
"""
from __future__ import annotations

import glob
import json
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
BIDI = re.compile("[​-‏‪-‮⁦-⁩﻿؜]")
HARAKAT = re.compile("[ً-ْٰـ]")


def norm(t: str, strip: bool = False) -> str:
    t = BIDI.sub("", unicodedata.normalize("NFC", t or ""))
    if strip:
        t = HARAKAT.sub("", t)
    return " ".join(t.split())


def load(name: str, splits: tuple[str, ...]) -> list[dict]:
    out = []
    for sp in splits:
        for f in sorted(glob.glob(f"{ROOT}{name}/snapshots/*/data/{sp}-*.parquet")):
            out += [dict(r, split=sp) for r in pq.read_table(f, columns=["transcription", "manuscript_name"]).to_pylist()]
    return out


def compare(a_name: str, a: list[dict], b_name: str, b: list[dict]) -> dict:
    ms = sorted({r["manuscript_name"] for r in a} & {r["manuscript_name"] for r in b})
    res = {"pair": f"{a_name} vs {b_name}", "shared_manuscripts": ms}
    for strip in (False, True):
        bt = {norm(r["transcription"], strip) for r in b if len(norm(r["transcription"], strip)) >= 12}
        hits = [norm(r["transcription"], strip) for r in a if norm(r["transcription"], strip) in bt]
        res["exact_line_overlap" + ("_no_diacritics" if strip else "")] = len(set(hits))
        if hits and not strip:
            res["examples"] = sorted(set(hits))[:3]
    res["test_lines_compared"] = len(b)
    print(json.dumps(res, ensure_ascii=False))
    return res


def main() -> None:
    allsp = ("train", "validation", "test")
    tm, r1all, r1t, r2t = load("tarima", allsp), load("RASAM-1", allsp), load("RASAM-1", ("test",)), load("RASAM-2", ("test",))
    out = [compare("TariMa (all splits)", tm, "RASAM-1 test", r1t), compare("TariMa (all splits)", tm, "RASAM-2 test", r2t),
           compare("RASAM-1 (all splits)", r1all, "RASAM-2 test", r2t)]
    Path("results/rasam").mkdir(parents=True, exist_ok=True)
    Path("results/rasam/overlap_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
