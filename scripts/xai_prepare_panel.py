"""The standard evaluation panel for every CRNN run (plan 2026-10-13, §3.9): every line of every test set, as stored and
ink-cropped, with canonical ids and no CTC-length filter.

    .geomvenv/bin/python scripts/xai_prepare_panel.py OUT.pkl [--sets khatt nakba ...]

Rows: id = "{dataset}/{split}/{index:06d}" (the store key; KITAB stores its 200 KHATT lines as split "train"), legacy_id =
the id the earlier per-line files use, dataset = tag or tag__inkcrop, label (harakat, tatweel and bidi marks stripped, as
for training), label_raw, img (64 px high, as geom_inkcrop_export.py), meta (page / manuscript fields where the set has
them), ink_extent (share of the image height holding the central 98 % of ink rows) and ctc_ok (whether the old
eval packs would have kept the line: width / 4 >= label length + 4). Earlier packs dropped lines failing ctc_ok, which
on KHATT removed the 17 worst loose crops; here they are scored and only flagged.
"""
from __future__ import annotations

import argparse
import glob
import io
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import HF, _table  # noqa: E402
from geom_inkcrop_export import clean, inkcrop  # noqa: E402
from geom_transfer import to64  # noqa: E402

SETS = {  # name: (arrow glob under HF, tag, split, text column, legacy id format, meta columns)
    "khatt": ("ahmedheakl___arocrbench_khatt/default/*/*/*.arrow", "kitab_khatt", "train", "text", "{tag}/{i:06d}", ["source"]),
    "nakba": ("U4RASD___omar-al-saleh-manuscripts-segments/default-2645abedcb4f16f8/*/*/*-test.arrow", "nakba_lines", "test", "text", "{tag}/{i:06d}", ["filename"]),
    "nakba_blind": ("U4RASD___omar-al-saleh-manuscripts-segments/default-0150fd3dc57fdfd3/*/*/*-blind_test.arrow", "nakba_blind", "blind_test", "text", "{tag}/{i:06d}", ["filename"]),
    "muharaf": ("aamijar___muharaf-public/default/*/*/muharaf-public-test.arrow", "muharaf_lines", "test", "text", "{tag}/test/{i:06d}", []),
    "rasam1": ("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow", "rasam1", "test", "transcription", "{tag}/test/{i:06d}", ["manuscript_name", "image_name"]),
    "rasam2": ("calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow", "rasam2", "test", "transcription", "{tag}/test/{i:06d}", ["manuscript_name", "image_name"]),
}


def ink_extent(g: np.ndarray) -> float:
    c = inkcrop(g)
    return round(c.shape[0] / max(1, g.shape[0]), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    a = ap.parse_args()
    rows = []
    for name in a.sets:
        pat, tag, split, col, legacy, meta_cols = SETS[name]
        (path,) = sorted(glob.glob(str(HF / pat)))[:1]
        t = _table(path)
        n_bad = 0
        for i in range(t.num_rows):
            row = t.slice(i, 1).to_pylist()[0]
            raw = row[col] or ""
            label = clean(raw)
            g = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("L"), dtype=np.uint8)
            img, crop = to64(g), to64(inkcrop(g))
            base = {"id": f"{tag}/{split}/{i:06d}", "legacy_id": legacy.format(tag=tag, i=i), "role": "eval", "label": label, "label_raw": raw,
                    "meta": {c: row.get(c) for c in meta_cols}, "ink_extent": ink_extent(g), "ctc_ok": img.shape[1] // 4 >= len(label) + 4}
            n_bad += not base["ctc_ok"]
            rows.append({**base, "dataset": tag, "img": img})
            rows.append({**base, "dataset": f"{tag}__inkcrop", "img": crop, "ctc_ok": crop.shape[1] // 4 >= len(label) + 4})
        print(f"{tag:14s} {t.num_rows:5d} lines  {n_bad} would have been dropped by the old CTC-length filter", flush=True)
    pickle.dump({"H": 64, "split": "panel-2026-10-13", "rows": rows}, open(a.out, "wb"), protocol=4)
    print(len(rows), "rows ->", a.out, f"{a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
