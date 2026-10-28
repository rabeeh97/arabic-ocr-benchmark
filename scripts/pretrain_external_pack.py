"""Idea 1 (2026-10-28): add other Arabic handwriting line datasets to a RASAM / TariMa CRNN pack, for pretraining our CRNN before the
RASAM / TariMa fine-tune.

    python scripts/pretrain_external_pack.py BASE.pkl OUT.pkl --charset RUN/charset.json [--hf-root DIR] [--max-per-set N] [--workers N]

External sets (train splits only, tags in brackets): Muharaf (mu, aamijar/muharaf-public), KHATT (kh, Teklia/KHATT_v1.0), Baybars
(bb, calfa-ai/baybars), Iskandar (is, calfa-ai/iskandar). Their lines get the same preprocessing and labels as the RASAM pack
(rasam_prepare_crnn_data.prep / clean, raw labels, the base pack's height) and role "train". A line with any character outside the
recogniser's charset (Latin, digits, punctuation the RASAM charset lacks) is dropped, so the classes stay those of our models and the
char LMs. Base rows (RASAM-1/2, TariMa train / validation / test) are kept as they are, so the same pack serves pretraining (all tags)
and the fine-tune (--train-sets r1 r2 tm). AraMS-28k is left out: its Maghrebi books are our unseen test books and the zip is too
large for a Colab session.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

import rasam_prepare_crnn_data as P

SETS = {"mu": ("aamijar/muharaf-public", "text"), "kh": ("Teklia/KHATT_v1.0", "text"),
        "bb": ("calfa-ai/baybars", "transcription"), "is": ("calfa-ai/iskandar", "transcription")}
CS: set = set()


def _one(args):
    tag, i, label, img_bytes, ms = args
    arr = P.prep(Image.open(io.BytesIO(img_bytes)))
    if not label or arr.shape[1] // 4 < len(label) + 2:  # CTC needs a frame per character
        return None
    return {"id": f"{tag}/train/{i:06d}", "dataset": tag, "role": "train", "label": label, "img": arr, "manuscript": ms,
            "page": None, "region": None, "orig_split": "train"}


def _init(h: int, cs: list) -> None:
    P.H, P.MAX_W = h, 24 * h
    CS.update(cs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--charset", type=Path, required=True)
    ap.add_argument("--hf-root", default="/content/hfcache", help="HF cache dir (datasets are downloaded into it when missing)")
    ap.add_argument("--max-per-set", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    tok = open("/content/.hf_token").read().strip() if os.path.exists("/content/.hf_token") else None
    blob = pickle.load(open(a.base, "rb"))
    h = int(blob["H"])
    cs = json.loads(a.charset.read_text(encoding="utf-8"))["chars"]
    _init(h, cs)
    base_ms = {r.get("manuscript") for r in blob["rows"]}
    stats = {}
    with Pool(a.workers, initializer=_init, initargs=(h, cs)) as pool:
        for tag, (repo, col) in SETS.items():
            d = snapshot_download(repo, repo_type="dataset", cache_dir=a.hf_root, allow_patterns=["*train*.parquet", "**/train*"], token=tok)
            fs = sorted(f for f in glob.glob(d + "/**/*.parquet", recursive=True) if os.path.basename(f).startswith("train"))
            jobs, oov, i = [], 0, 0
            for f in fs:
                t = pq.read_table(f)
                cols = t.column_names
                for rec in t.to_pylist():
                    label = P.clean(rec[col], "raw")
                    if any(c not in CS for c in label):
                        oov += 1
                        continue
                    ms = rec.get("manuscript_name") if "manuscript_name" in cols else None
                    jobs.append((tag, i, label, rec["image"]["bytes"], ms))
                    i += 1
                    if a.max_per_set and i >= a.max_per_set:
                        break
                if a.max_per_set and i >= a.max_per_set:
                    break
            rows = [r for r in pool.imap(_one, jobs, chunksize=64) if r is not None]
            overlap = sorted({r["manuscript"] for r in rows if r["manuscript"]} & base_ms)
            stats[tag] = {"files": len(fs), "kept": len(rows), "dropped_oov": oov, "dropped_ctc": len(jobs) - len(rows),
                          "mb": round(sum(r["img"].nbytes for r in rows) / 1e6), "manuscripts_shared_with_base": overlap}
            print(tag, stats[tag], flush=True)
            blob["rows"].extend(rows)
            del jobs, rows
    blob["split"] = blob.get("split", "") + "+ext-" + "-".join(SETS)
    pickle.dump(blob, open(a.out, "wb"), protocol=4)
    print("STATS", json.dumps(stats), "->", a.out, f"{a.out.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
