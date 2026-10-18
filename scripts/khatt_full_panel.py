"""Full KHATT v1.0 (Teklia copy) for the evaluation panel and for an in-domain KHATT CRNN (checklist 1.6, map `khatt-full`).

    .geomvenv/bin/python scripts/khatt_full_panel.py [--out-dir results/panel] [--skip-train]

Source: the Teklia/KHATT_v1.0 parquet snapshot in the HF hub cache on the SSD (4,672 train, 963 validation, 1,038 test lines;
no writer IDs, no page IDs). Writes four files to OUT_DIR:

  khatt_full_panel.pkl     test lines in the xai_prepare_panel.py format, as stored (dataset `kitab_khatt_full`) and
                           ink-cropped (`kitab_khatt_full__inkcrop`); canonical id `kitab_khatt_full/test/{index:06d}`.
                           Scored by crnn_eval_panel.py exactly like panel.pkl.
  khatt_full_train.pkl     train lines (role "train") and validation lines (role "heldout") as stored, in the
                           xai_train_aligner.py format, for the in-domain KHATT CRNN (checklist 4.6). No test lines.
  khatt_full_flags.parquet every line of all three splits (no images): id, split, label, flags below.
  khatt_full_manifest.json counts per split and flag, source snapshot, sha256 of the three files above.

Flags (also in each panel row's `meta`):
  text_norm      letters-only, whitespace-collapsed text (clean() then drop every non-Arabic-letter character).
  text_count     lines in KHATT v1.0 (all splits) with the same text_norm; text_repeat = text_count > 1 (the design's
                 "text occurs more than once"). text_cluster groups identical texts (same text, several writers: item 2.16).
  ngram_share    share of the line's word 3-grams that occur in another KHATT v1.0 line (line breaks differ by writer, so
                 exact repeats undercount fixed text). fixed_text = ngram_share >= 0.5 (lines under 3 words: text_repeat).
  text_in_train  text_norm occurs in the v1.0 train split (another train line, for train lines): text memorisation.
  kitab_text_ids KITAB-KHATT (arocrbench_khatt, 200 lines) ids with the same text_norm; kitab_text = non-empty.
  kitab_image_id the KITAB-KHATT line that is the same image (same size, thumbnail correlation > 0.99), else "".
                 The aligner-recipe CRNNs trained on the KITAB lines ("panel" role), so these lines are in_train for them.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import io
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import HF, _table  # noqa: E402
from geom_inkcrop_export import clean, inkcrop  # noqa: E402
from geom_transfer import to64  # noqa: E402

HUB = Path("/Volumes/PortableSSD/huggingface/hub")
SNAP = "datasets--Teklia--KHATT_v1.0/snapshots/*/data/{split}-*.parquet"
KITAB = "ahmedheakl___arocrbench_khatt/default/*/*/*.arrow"
TAG = "kitab_khatt_full"
SPLITS = ("train", "validation", "test")
NON_LETTER = re.compile(r"[^ء-ي\s]")


def norm_text(t: str) -> str:
    return " ".join(NON_LETTER.sub(" ", clean(t or "")).split())


def trigrams(t: str) -> list[tuple[str, ...]]:
    w = t.split()
    return [tuple(w[k : k + 3]) for k in range(len(w) - 2)]


def text_flags(texts: list[str], splits: list[str], kitab_texts: list[str]) -> list[dict]:
    """Text-only flags for every line (texts already normalised). Pure, so it is unit-tested."""
    count = collections.Counter(texts)
    cluster = {t: i for i, t in enumerate(sorted(t for t, n in count.items() if n > 1))}
    train_count = collections.Counter(t for t, s in zip(texts, splits) if s == "train")
    owners: dict[tuple, set[int]] = collections.defaultdict(set)
    for j, t in enumerate(texts):
        for g in trigrams(t):
            owners[g].add(j)
    kit = collections.defaultdict(list)
    for j, t in enumerate(kitab_texts):
        kit[t].append(f"kitab_khatt/train/{j:06d}")
    out = []
    for j, t in enumerate(texts):
        gs = trigrams(t)
        share = sum(len(owners[g] - {j}) > 0 for g in gs) / len(gs) if gs else 0.0
        rep = count[t] > 1
        out.append({"text_norm": t, "text_count": count[t], "text_repeat": rep, "text_cluster": cluster.get(t, -1),
                    "ngram_share": round(share, 4), "fixed_text": share >= 0.5 if gs else rep,
                    "text_in_train": train_count[t] > (splits[j] == "train"),
                    "kitab_text_ids": kit.get(t, []), "kitab_text": t in kit})
    return out


def thumb(g: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(g).resize((128, 16), Image.BILINEAR), dtype=np.float32).ravel()


def same_image(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    ta, tb = thumb(a), thumb(b)
    if ta.std() == 0 or tb.std() == 0:
        return bool(np.array_equal(a, b))
    return float(np.corrcoef(ta, tb)[0, 1]) > 0.99


def gray(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("L"), dtype=np.uint8)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results/panel")
    ap.add_argument("--skip-train", action="store_true", help="do not write khatt_full_train.pkl")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    src = {s: sorted(glob.glob(str(HUB / SNAP.format(split=s)))) for s in SPLITS}
    assert all(len(v) == 1 for v in src.values()), f"expected one parquet per split under {HUB}: {src}"
    tables = {s: pq.read_table(src[s][0]) for s in SPLITS}
    (kpath,) = sorted(glob.glob(str(HF / KITAB)))[:1]
    kitab = _table(kpath).to_pylist()
    kitab_norm = [norm_text(r["text"]) for r in kitab]

    lines = [(s, i, r["text"] or "") for s in SPLITS for i, r in enumerate(tables[s].select(["text"]).to_pylist())]
    flags = text_flags([norm_text(t) for _, _, t in lines], [s for s, _, _ in lines], kitab_norm)
    kitab_img = {}
    for (s, i, _), f in zip(lines, flags):
        f["kitab_image_id"] = ""
        if f["kitab_text"]:
            g = gray(tables[s].slice(i, 1).column("image")[0].as_py()["bytes"])
            for kid in f["kitab_text_ids"]:
                j = int(kid.rsplit("/", 1)[1])
                if j not in kitab_img:
                    kitab_img[j] = gray(kitab[j]["image"]["bytes"])
                if same_image(g, kitab_img[j]):
                    f["kitab_image_id"] = kid
                    break

    panel, train = [], []
    for (s, i, raw), f in zip(lines, flags):
        if s != "test" and a.skip_train:
            continue
        g = gray(tables[s].slice(i, 1).column("image")[0].as_py()["bytes"])
        label = clean(raw)
        img = to64(g)
        sid = f"{TAG}/{s}/{i:06d}"
        if s == "test":
            crop = to64(inkcrop(g))
            base = {"id": sid, "legacy_id": sid, "role": "eval", "label": label, "label_raw": raw, "meta": {"split": s, **f},
                    "ink_extent": round(inkcrop(g).shape[0] / max(1, g.shape[0]), 4), "ctc_ok": img.shape[1] // 4 >= len(label) + 4}
            panel.append({**base, "dataset": TAG, "img": img})
            panel.append({**base, "dataset": f"{TAG}__inkcrop", "img": crop, "ctc_ok": crop.shape[1] // 4 >= len(label) + 4})
        else:
            train.append({"id": sid, "dataset": TAG, "role": "train" if s == "train" else "heldout", "label": label, "img": img})

    files = {}
    p = a.out_dir / "khatt_full_panel.pkl"
    # 2026-10-18: the __inkcrop rows were uncropped copies before the inkcrop() fix for binarised images (KHATT v1.0 is
    # {0, 255}); the new pack name lets store_build.py drop __inkcrop scores made on the old pack (sources.json stale_packs).
    pickle.dump({"H": 64, "split": "panel-khatt-full-2026-10-18", "rows": panel}, open(p, "wb"), protocol=4)
    files[p.name] = p
    if not a.skip_train:
        p = a.out_dir / "khatt_full_train.pkl"
        pickle.dump({"H": 64, "split": "khatt-full-v1.0", "rows": train}, open(p, "wb"), protocol=4)
        files[p.name] = p
    tab = pa.Table.from_pylist([{"id": f"{TAG}/{s}/{i:06d}", "split": s, "index": i, "label_raw": raw, "label": clean(raw), **f}
                                for (s, i, raw), f in zip(lines, flags)])
    p = a.out_dir / "khatt_full_flags.parquet"
    pq.write_table(tab, p)
    files[p.name] = p

    counts = {}
    for s in SPLITS:
        fs = [f for (sp, _, _), f in zip(lines, flags) if sp == s]
        counts[s] = {"lines": len(fs), **{k: sum(bool(f[k]) for f in fs) for k in
                                         ("text_repeat", "fixed_text", "text_in_train", "kitab_text", "kitab_image_id")},
                     "unique_text": sum(not f["fixed_text"] for f in fs)}
    kit_found = {k for f in flags for k in f["kitab_text_ids"]}
    manifest = {"item": "MASTER-CHECKLIST 1.6 (khatt-full)", "source": {s: src[s][0] for s in SPLITS}, "kitab_source": kpath,
                "tags": [TAG, f"{TAG}__inkcrop"], "id_format": f"{TAG}/{{split}}/{{index:06d}}", "counts": counts,
                "kitab_texts_found_in_v1": len(kit_found), "kitab_images_found_in_v1": sum(bool(f["kitab_image_id"]) for f in flags),
                "panel_rows": len(panel), "train_rows": len(train),
                "files": {n: {"bytes": q.stat().st_size, "sha256": sha256(q)} for n, q in files.items()}}
    (a.out_dir / "khatt_full_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "kitab_texts_found_in_v1", "kitab_images_found_in_v1", "panel_rows", "train_rows")}, indent=1))
    for n, q in files.items():
        print(f"{n:28s} {q.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
