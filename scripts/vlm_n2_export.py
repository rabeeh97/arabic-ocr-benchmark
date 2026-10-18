"""GPU night 2 (checklist 7.2-7.6): the line images the VLM jobs read, built on the laptop from the cached Arrow files.

    .geomvenv/bin/python scripts/vlm_n2_export.py results/vlm_n2/vlm_n2_lines.pkl

Images are kept at native resolution (not the 64 px CRNN panel): RGB, longest side capped at 1600 like the benchmark's
KitabBench adapter (the client then caps at 1024, as the benchmark runner does), stored as PNG (greyscale when R = G = B).
Ids are the benchmark sample ids ("{dataset}/{split}/{index:06d}"), so readings join the earlier suite-v1 bench rows.

Rows: key = "{id}|{variant}", id, dataset, variant, png, gt (raw label, as the benchmark scores it), meta.
  kitab_khatt (all 200): stored; inkcrop (geom_inkcrop_export.inkcrop rows, applied to the RGB image); shear10 (X-T2's
    KHATT->Muharaf slant, +10.37 degrees, geom_transfer.op_shear per channel); h60 (height 60 px, Muharaf's storage height,
    LANCZOS); h60_n4 / h60_n8 / h60_n16 (h60 plus Gaussian noise of that SD in grey levels, seeded per line).   [7.3, 7.6]
  nakba_lines test 0-399, rasam1 test 0-399, rasam2 test (all 372): stored.                                   [7.5, 7.2]
  nakba_blind 0-149: stored.                                                                                  [7.2]
  muharaf_lines test 0-199: stored.                                                                           [7.4]
Also writes icl_trials.json (checklist 7.2, design writer-adaptation.md B): one trial per (test line, condition):
  cov    the unit's best letter-coverage line (greedy set cover over letter x positional form) with its label, then the test line
  label  the same reference's label only, no image (text-priming control)
  other  the coverage-best line of a different writer (RASAM-2: the next manuscript; NAKBA: Muharaf test), with label
  rand   a random line of the same unit
  cov3   the three best coverage lines of the unit
Units: RASAM-2 manuscripts (references and tests from the same manuscript; 10 test lines each); NAKBA, one writer
(references from NAKBA test lines 0-399, tests = blind lines 0-149). Design (d) is approximated by "cov" (no own
calibration-sentence handwriting exists yet; the 2.7 candidates in results/calibration/ are copy-texts for the future set).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import HF, _table  # noqa: E402
from geom_inkcrop_export import clean  # noqa: E402
from geom_transfer import op_shear  # noqa: E402
from xai_adapt_crnn import coverage_order  # noqa: E402

KHATT_SHEAR = 10.366268436188712  # transfer_params.pkl["khatt:muharaf"]["P"]["shear"] (X-T2)
NOISE = (4, 8, 16)
SETS = {  # tag: (arrow glob, split, text column, rows, meta columns)
    "kitab_khatt": ("ahmedheakl___arocrbench_khatt/default/*/*/*.arrow", "train", "text", None, ["source"]),
    "nakba_lines": ("U4RASD___omar-al-saleh-manuscripts-segments/default-2645abedcb4f16f8/*/*/*-test.arrow", "test", "text", 400, ["filename"]),
    "nakba_blind": ("U4RASD___omar-al-saleh-manuscripts-segments/default-0150fd3dc57fdfd3/*/*/*-blind_test.arrow", "blind_test", "text", 150, ["filename"]),
    "rasam1": ("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow", "test", "transcription", 400, ["manuscript_name", "image_name"]),
    "rasam2": ("calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow", "test", "transcription", None, ["manuscript_name", "image_name"]),
    "muharaf_lines": ("aamijar___muharaf-public/default/*/*/muharaf-public-test.arrow", "test", "text", 200, []),
}


def cap_image(img: Image.Image, max_dim: int) -> Image.Image:
    """arabic_ocr_bench.imaging.cap_image without the aspect-ratio pad (the client applies the full version at 1024)."""
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def png(img: Image.Image) -> bytes:
    a = np.asarray(img)
    if a.ndim == 3 and (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all():
        img = Image.fromarray(a[..., 0])
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def ink_rows(g: np.ndarray) -> tuple[int, int]:
    """Row range kept by geom_inkcrop_export.inkcrop (same thresholds), so the crop can be applied to the RGB image."""
    from skimage.filters import threshold_otsu

    try:
        t = threshold_otsu(g)
    except ValueError:
        return 0, g.shape[0]
    ink = g <= t if t <= g.min() else g < t
    prof = ink.sum(1).astype(float)
    if prof.sum() == 0:
        return 0, g.shape[0]
    c = np.cumsum(prof) / prof.sum()
    top, bot = int(np.searchsorted(c, 0.01)), int(np.searchsorted(c, 0.99))
    m = int(0.25 * (bot - top + 1)) + 2
    return max(0, top - m), min(g.shape[0], bot + m + 1)


def shear_rgb(img: Image.Image, deg: float) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    out = np.stack([op_shear(a[..., c], deg) for c in range(3)], -1)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def to_h60(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, round(w * 60 / h)), 60), Image.LANCZOS)


def noisy(img: Image.Image, sd: float, seed: int) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    n = np.random.default_rng(seed).normal(0, sd, a.shape[:2])[..., None]
    return Image.fromarray(np.clip(a + n + 0.5, 0, 255).astype(np.uint8))


def khatt_variants(img: Image.Image, i: int):
    yield "stored", img
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    t, b = ink_rows(g)
    yield "inkcrop", img.crop((0, t, img.size[0], b))
    yield "shear10", shear_rgb(img, KHATT_SHEAR)
    h60 = to_h60(img)
    yield "h60", h60
    for sd in NOISE:
        yield f"h60_n{sd}", noisy(h60, sd, 1000 * sd + i)


def icl_trials(rows: list[dict], n_test: int = 10, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    lab = {r["id"]: {**r, "label": clean(r["gt"])} for r in rows if r["variant"] == "stored"}
    units = defaultdict(list)
    for r in lab.values():
        if r["dataset"] == "rasam2":
            units["rasam2:" + str(r["meta"]["manuscript_name"])].append(r)
    names = sorted(u for u in units if len(units[u]) >= 6)  # 3 coverage refs + 1 random + tests
    plan = []  # (unit, refs pool rows, test rows, other-writer ref id)
    cov_of = {u: coverage_order(units[u], 3) for u in names}
    for k, u in enumerate(names):
        rs = sorted(units[u], key=lambda r: r["id"])
        plan.append((u, rs, None, cov_of[names[(k + 1) % len(names)]][0]))
    nak_pool = sorted((r for r in lab.values() if r["dataset"] == "nakba_lines"), key=lambda r: r["id"])
    muh = sorted((r for r in lab.values() if r["dataset"] == "muharaf_lines"), key=lambda r: r["id"])
    blind = sorted((r for r in lab.values() if r["dataset"] == "nakba_blind"), key=lambda r: r["id"])
    plan.append(("nakba", nak_pool, blind, coverage_order(muh, 1)[0]))
    trials = []
    for u, pool, test, other in plan:
        cov3 = cov_of[u] if u in cov_of else coverage_order(pool, 3)
        rand = rng.choice([r["id"] for r in pool if r["id"] not in cov3])
        if test is None:
            left = [r["id"] for r in pool if r["id"] not in cov3 and r["id"] != rand]
            test_ids = sorted(rng.sample(left, min(n_test, len(left))))
        else:
            test_ids = [r["id"] for r in test]
        conds = {"cov": ([cov3[0]], False), "label": ([cov3[0]], True), "other": ([other], False), "rand": ([rand], False),
                 "cov3": (cov3, False)}
        for c, (refs, label_only) in conds.items():
            for t in test_ids:
                trials.append({"key": f"{t}|{c}", "unit": u, "test": t, "cond": c, "refs": refs, "label_only": label_only})
    order = {c: i for i, c in enumerate(["cov", "label", "other", "rand", "cov3"])}  # the keep rule needs cov, label, other
    trials.sort(key=lambda r: (order[r["cond"]], r["unit"] == "nakba", r["test"]))
    return trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="at most this many lines per set (smoke tests)")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, (pat, split, col, n, meta_cols) in SETS.items():
        (path,) = sorted(glob.glob(str(HF / pat)))[:1]
        t = _table(path)
        n = t.num_rows if n is None else min(n, t.num_rows)
        if a.limit:
            n = min(n, a.limit)
        for i in range(n):
            row = t.slice(i, 1).to_pylist()[0]
            img = cap_image(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"), 1600)
            base = {"id": f"{tag}/{split}/{i:06d}", "dataset": tag, "gt": row[col] or "", "meta": {c: row.get(c) for c in meta_cols}}
            variants = khatt_variants(img, i) if tag == "kitab_khatt" else [("stored", img)]
            for v, im in variants:
                rows.append({**base, "key": f"{base['id']}|{v}", "variant": v, "png": png(im)})
        print(f"{tag:14s} {n:4d} lines", flush=True)
    pickle.dump({"version": "vlm_n2-2026-10-18", "rows": rows}, open(a.out, "wb"), protocol=4)
    trials = icl_trials(rows)
    (a.out.parent / "icl_trials.json").write_text(json.dumps(trials, ensure_ascii=False, indent=0), encoding="utf-8")
    mb = a.out.stat().st_size / 1e6
    print(f"{len(rows)} rows -> {a.out} ({mb:.0f} MB); {len(trials)} ICL trials", flush=True)


if __name__ == "__main__":
    main()
