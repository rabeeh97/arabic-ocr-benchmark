"""Training data for the laptop replica of the CRNN-CTC aligner (same format as xai_prepare_aligner_data.py).

    .geomvenv/bin/python scripts/geom_prepare_replica_data.py OUT.pkl [--n-muharaf 9000]

The 3090 model trained on about 24 k lines (Muharaf train, RASAM-1 and RASAM-2, the small RASAM sets repeated).
This replica uses a random subset of the Muharaf train lines plus the cached RASAM test lines, because RASAM train
is not in the local cache. Held out: 300 Muharaf test lines and 100 RASAM-1 lines. NAKBA and KHATT are never used.
"""
from __future__ import annotations

import argparse
import glob
import io
import pickle
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import HF, _table  # noqa: E402
from geom_transfer import to64  # noqa: E402

STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")


def clean(t: str) -> str:
    return " ".join(STRIP.sub("", unicodedata.normalize("NFC", t)).split())


def rows_from(paths, col, tag, role, n, rng, skip_ids=None):
    pool = []
    for pi, path in enumerate(paths):
        t = _table(path)
        pool += [(pi, i) for i in range(t.num_rows)]
    rng.shuffle(pool)
    tables = {}
    out = []
    for pi, i in pool:
        if len(out) >= n:
            break
        if pi not in tables:
            tables[pi] = _table(paths[pi])
        row = tables[pi].slice(i, 1).to_pylist()[0]
        label = clean(row[col] or "")
        if not 2 <= len(label) <= 120:
            continue
        img = to64(np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")))
        if img.shape[1] // 4 < len(label) + 4:
            continue
        out.append({"id": f"{tag}/{pi}/{i:06d}", "dataset": tag, "role": role, "label": label, "img": img})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-muharaf", type=int, default=9000)
    a = ap.parse_args()
    rng = random.Random(0)
    g = lambda pat: sorted(glob.glob(str(HF / pat)))  # noqa: E731
    rows = []
    rows += rows_from(g("aamijar___muharaf-public/default/*/*/muharaf-public-train-*.arrow"), "text", "muharaf_lines", "train", a.n_muharaf, rng)
    rows += rows_from(g("aamijar___muharaf-public/default/*/*/muharaf-public-test.arrow"), "text", "muharaf_lines", "heldout", 300, rng)
    r1 = rows_from(g("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow"), "transcription", "rasam1", "train", 100000, rng)
    for r in r1[:100]:
        r["role"] = "heldout"
    rows += r1
    rows += rows_from(g("calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow"), "transcription", "rasam2", "train", 100000, rng)
    from collections import Counter
    print(Counter((r["dataset"], r["role"]) for r in rows))
    pickle.dump({"H": 64, "rows": rows}, open(a.out, "wb"), protocol=4)
    print(f"{a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
