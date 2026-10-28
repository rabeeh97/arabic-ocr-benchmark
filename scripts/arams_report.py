"""Score the AraMS-28k Maghrebi books under AraMS's transcription conventions and check text overlap with our training text (2026-10-28).

    PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/arams_report.py [--name AM_E60]

AraMS ground truth has letters and spaces only (no harakat, tatweel, punctuation or digits), while our models were trained on Calfa's
transcriptions, which keep some of them. So readings are scored after removing everything AraMS never transcribes:
  - "letters": drop harakat, tatweel, punctuation and digits from the reading, collapse whitespace (hamza / alef forms kept);
  - "arams_norm": also merge alef forms (أ إ آ ٱ -> ا) and alef maqsura (ى -> ي), the paper's gt_normalized (its published
    Kraken 32.71 % / HATFormer 37.88 % on book_03 use that reference).
Micro CER (edits / reference characters) per book, greedy and with LM, plus per-page spread.

Text overlap: share of each book's lines that contain a word 5-gram from the RASAM-1 / RASAM-2 / TariMa training transcriptions
(letters profile). A high share means our character LM has seen the text (e.g. a hadith collection copied in both), which would
help it beyond reading the hand.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

import pyarrow.parquet as pq

from arabic_ocr_bench.metrics import char_edit_counts

HF = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
TAGS = {"am03": "book_03 (AraMS test)", "am19": "book_19", "am20": "book_20", "am24": "book_24"}
DROP = re.compile(r"[ً-ٰٟـؐ-ؚۖ-ۭ0-9٠-٩۰-۹]|[^\w\s]")
ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})


def letters(t: str) -> str:
    return " ".join(DROP.sub("", t).split())


def arams_norm(t: str) -> str:
    return letters(t).translate(ALEF)


PROFILES = {"letters": letters, "arams_norm": arams_norm}


def micro(pairs, f) -> tuple[float, int]:
    e = n = 0
    for ref, hyp in pairs:
        c = char_edit_counts(f(ref), f(hyp))
        e += c["sub"] + c["del"] + c["ins"]
        n += c["ref_len"]
    return 100 * e / max(1, n), n


def train_ngrams(k: int = 5) -> set:
    grams = set()
    for name in ("RASAM-1", "RASAM-2", "tarima"):
        for f in glob.glob(f"{HF}{name}/snapshots/*/data/train-*.parquet"):
            for t in pq.read_table(f, columns=["transcription"]).column("transcription").to_pylist():
                w = letters(t or "").translate(ALEF).split()
                grams.update(tuple(w[i:i + k]) for i in range(len(w) - k + 1))
    return grams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="AM_E60")
    a = ap.parse_args()
    grams = train_ngrams()
    out = {"name": a.name, "books": {}}
    for tag, label in TAGS.items():
        d = json.loads(Path(f"results/rasam/lm/{a.name}__{tag}__test.json").read_text())
        L = d["lines"]
        row = {"label": label, "n_lines": len(L), "alpha": d.get("alpha"), "beta": d.get("beta")}
        for p, f in PROFILES.items():
            row[f"greedy_{p}"], row["ref_chars"] = micro([(x["ref"], x["greedy"]) for x in L], f)
            row[f"lm_{p}"], _ = micro([(x["ref"], x["hyp"]) for x in L], f)
        pages = {}
        for x in L:
            pages.setdefault(x["page"], []).append((x["ref"], x["hyp"]))
        pc = sorted(micro(v, arams_norm)[0] for v in pages.values())
        row["page_cer_median"], row["page_cer_p10"], row["page_cer_p90"] = (round(statistics.median(pc), 2), round(pc[len(pc) // 10], 2),
                                                                           round(pc[9 * len(pc) // 10], 2))
        seen = 0
        for x in L:
            w = arams_norm(x["ref"]).split()
            seen += any(tuple(w[i:i + 5]) in grams for i in range(len(w) - 4))
        row["lines_with_seen_5gram_pct"] = round(100 * seen / len(L), 1)
        out["books"][tag] = row
    allL = [x for tag in TAGS for x in json.loads(Path(f"results/rasam/lm/{a.name}__{tag}__test.json").read_text())["lines"]]
    out["all"] = {p: {"greedy": micro([(x["ref"], x["greedy"]) for x in allL], f)[0], "lm": micro([(x["ref"], x["hyp"]) for x in allL], f)[0]}
                  for p, f in PROFILES.items()}
    Path("results/rasam/arams_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{'book':22} {'lines':>6} | {'greedy':>7} {'LM':>6} (letters) | {'greedy':>7} {'LM':>6} (arams_norm) | page CER p10/med/p90 | seen 5-gram")
    for tag, r in out["books"].items():
        print(f"{r['label']:22} {r['n_lines']:6} | {r['greedy_letters']:7.2f} {r['lm_letters']:6.2f}           | {r['greedy_arams_norm']:7.2f} "
              f"{r['lm_arams_norm']:6.2f}              | {r['page_cer_p10']:5.1f} {r['page_cer_median']:5.1f} {r['page_cer_p90']:5.1f}   | "
              f"{r['lines_with_seen_5gram_pct']:5.1f}%")
    print("ALL", {p: {k: round(v, 2) for k, v in s.items()} for p, s in out["all"].items()})


if __name__ == "__main__":
    main()
