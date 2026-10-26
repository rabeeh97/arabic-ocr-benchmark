"""Does Calfa's AMIDDA reuse the Hugging Face RASAM-1 / RASAM-2 / TariMa splits? Streams the parquet files in small batches (8.6 GB Mac).

    python scripts/amidda_split_check.py

Keys per line: md5 of the image bytes, and (image_name, transcription). For each official split: how many of its lines are in AMIDDA
train / validation / test. If the official test lines are all in AMIDDA test and none in AMIDDA train, the per-corpus numbers on the
AMIDDA model card are on the official test lines.
"""
from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter

import pyarrow.parquet as pq

HUB = "/Volumes/PortableSSD/huggingface/hub"


def files(repo, split):
    return sorted(f for f in glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/{split}-*.parquet") if "/._" not in f)


def keys(repo, split, extra=()):
    img, txt, meta = set(), set(), Counter()
    for f in files(repo, split):
        for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "transcription", "image_name", *extra]):
            d = b.to_pydict()
            for i in range(len(d["transcription"])):
                im = d["image"][i]
                if im and im.get("bytes"):
                    img.add(hashlib.md5(im["bytes"]).hexdigest())
                txt.add((d["image_name"][i], " ".join((d["transcription"][i] or "").split())))
                if extra:
                    meta[d[extra[0]][i]] += 1
    return img, txt, meta


def main() -> None:
    am = {}
    for s in ("train", "validation", "test"):
        am[s] = keys("calfa-ai--AMIDDA", s, ("dataset",))
        print(f"AMIDDA {s}: {sum(am[s][2].values())} lines", dict(am[s][2]), flush=True)
    res = {}
    for repo, name in (("calfa-ai--RASAM-1", "RASAM-1"), ("calfa-ai--RASAM-2", "RASAM-2"), ("calfa-ai--tarima", "TariMa")):
        for split in ("test", "validation", "train"):
            img, txt, _ = keys(repo, split)
            r = {"n_img": len(img), "n_txt": len(txt)}
            for s in am:
                r[f"img_in_{s}"] = len(img & am[s][0])
                r[f"txt_in_{s}"] = len(txt & am[s][1])
            res[f"{name} {split}"] = r
            print(name, split, r, flush=True)
    json.dump(res, open("results/rasam/amidda_split_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
