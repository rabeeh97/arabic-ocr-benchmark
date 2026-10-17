"""Dotless (skeleton) labels for the skeleton-only CRNN (checklist 4.2, decoding-and-restoration §4).

Laptop, builds the label map from the store's position-aware rasm mapping (arabic_ocr_bench.characterize.errors.to_rasm,
19 skeleton letters; everything that is not an Arabic letter is kept):
    .geomvenv/bin/python scripts/xai_skeleton_labels.py build skeleton_map.json results/panel/aligner_data_standard.pkl \
        results/panel/panel.pkl results/panel/khatt_full_panel.pkl

Box (no arabic_ocr_bench needed), applies the map to a training pickle; every label must be in the map:
    python xai_skeleton_labels.py apply skeleton_map.json aligner_data_standard.pkl aligner_data_standard_skel.pkl

The same map scores the panel: crnn_eval_panel.py --label-map skeleton_map.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

MAIN_SRC = "/Volumes/PortableSSD/repos/arabic_ocr/src"  # read-only import of the store's rasm mapping


def build(out: Path, pickles: list[Path]) -> None:
    sys.path.insert(0, MAIN_SRC)
    from arabic_ocr_bench.characterize.errors import to_rasm
    from arabic_ocr_bench.characterize.text import prepare  # presentation forms (e.g. the lam-alef ligature) expand first

    labels = set()
    for p in pickles:
        labels |= {r["label"] for r in pickle.load(open(p, "rb"))["rows"]}
    m = {lab: to_rasm(prepare(lab)) for lab in sorted(labels)}
    out.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    before = {c for lab in m for c in lab}
    after = {c for v in m.values() for c in v}
    print(f"{len(m)} labels; characters {len(before)} -> {len(after)}; changed labels {sum(k != v for k, v in m.items())} -> {out}")


def apply(map_path: Path, src: Path, dst: Path) -> None:
    m = json.loads(map_path.read_text(encoding="utf-8"))
    blob = pickle.load(open(src, "rb"))
    missing = {r["label"] for r in blob["rows"]} - set(m)
    if missing:
        raise SystemExit(f"{len(missing)} labels missing from the map, e.g. {sorted(missing)[:3]}")
    rows = [{**r, "label": m[r["label"]], "label_dotted": r["label"]} for r in blob["rows"]]
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({**blob, "split": f"{blob.get('split', 'aligner')}-skeleton", "rows": rows}, f, protocol=4)
    tmp.replace(dst)  # a crash mid-write leaves no half-written pickle under the final name
    print(len(rows), "rows ->", dst, "classes", len({c for r in rows for c in r["label"]}) + 1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
    elif sys.argv[1] == "apply":
        apply(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
