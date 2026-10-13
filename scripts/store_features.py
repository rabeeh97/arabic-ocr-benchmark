"""Image measurements for every panel line, plus a Muharaf-train reference (plan 2026-10-13, §3.6 item 2).

    .geomvenv/bin/python scripts/store_features.py [--out results/store] [--n-ref 3000] [--procs 6]

Runs geom_measure.measure() on the native-resolution image of every line of the six panel sets and on n-ref Muharaf
training lines, and writes OUT/features.parquet (one row per sample_key; the reference rows have dataset
"muharaf_train") with the scalar measures plus quantities at the CRNN's 64 px input:
  slant_deg            skeleton slant (positive leans right), slant_iqr
  stroke_px64          stroke width in pixels after resizing to 64 px height
  pen_contrast         horizontal / vertical stroke width
  rule_share           share of ink in ruled-line components
  dot_px64             dot diameter at 64 px height
  ink_px_per_char      line width at 64 px height / label length
  ink_extent           share of the height holding the central 98 % of ink rows (as xai_prepare_panel.py)
and, against the reference distribution, *_pct (percentile rank among training lines) and flags
slant_outlier (outside training p2.5–p97.5), stroke_outlier, pen_outlier, ruled (rule_share > 0.02),
low_res (ink_px_per_char < training p5, or dot_px64 < 2).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from geom_data import HF, _table  # noqa: E402
from geom_inkcrop_export import clean, inkcrop  # noqa: E402
from geom_measure import measure  # noqa: E402
from xai_prepare_panel import SETS  # noqa: E402

REF = ("aamijar___muharaf-public/default/*/*/muharaf-public-train-00000-of-00003.arrow", "text")
KEEP = ("h0", "w0", "sw_native", "sw", "sw_horizontal_stroke", "sw_vertical_stroke", "sw_cv", "rule_share", "n_dots", "dot_diam", "norm_scale",
        "slant_skel_deg", "slant_skel_iqr", "slant_proj_deg", "slope_proj_deg", "ink_density", "asc_over_sw", "desc_over_sw", "xheight_over_sw", "turn_mean_rad")


def rows_of(pat: str, col: str, tag: str, split: str, limit: int | None = None):
    (path,) = sorted(glob.glob(str(HF / pat)))[:1]
    t = _table(path)
    idx = range(t.num_rows) if limit is None else np.random.default_rng(0).choice(t.num_rows, min(limit, t.num_rows), replace=False)
    for i in idx:
        row = t.slice(int(i), 1).to_pylist()[0]
        g = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("L"), dtype=np.uint8)
        yield (f"{tag}/{split}/{int(i):06d}", int(i), g, clean(row[col] or ""))


def job(args):
    key, i, g, label = args
    r = measure((key.split("/")[0], i, g, label))
    out = {"sample_key": key, "dataset": key.split("/")[0], "ok": bool(r.get("ok")), "label_len": len(label)}
    for k in KEEP:
        v = r.get(k)
        out[k] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None
    c = inkcrop(g)
    out["ink_extent"] = round(c.shape[0] / max(1, g.shape[0]), 4)
    h0, w0 = g.shape
    out["stroke_px64"] = out["sw_native"] * 64 / h0 if out["sw_native"] else None
    out["dot_px64"] = (out["dot_diam"] / out["norm_scale"]) * 64 / h0 if out["dot_diam"] and out["norm_scale"] else None
    out["pen_contrast"] = out["sw_horizontal_stroke"] / out["sw_vertical_stroke"] if out["sw_horizontal_stroke"] and out["sw_vertical_stroke"] else None
    out["slant_deg"], out["slant_iqr"] = out["slant_skel_deg"], out["slant_skel_iqr"]
    out["ink_px_per_char"] = (w0 * 64 / h0) / max(1, len(label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=W / "results/store")
    ap.add_argument("--n-ref", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    res = []
    with Pool(a.procs) as p:
        res += p.map(job, rows_of(REF[0], REF[1], "muharaf_train", "train00000", a.n_ref), chunksize=8)
        print("reference", sum(r["ok"] for r in res), "/", len(res), flush=True)
        for name, (pat, tag, split, col, *_rest) in SETS.items():
            rs = p.map(job, rows_of(pat, col, tag, split), chunksize=8)
            print(tag, sum(r["ok"] for r in rs), "/", len(rs), flush=True)
            res += rs
    ref = [r for r in res if r["dataset"] == "muharaf_train" and r["ok"]]

    def pct(name):
        v = np.sort([r[name] for r in ref if r[name] is not None])
        for r in res:
            r[f"{name}_pct"] = None if r[name] is None else float(np.searchsorted(v, r[name]) / len(v))
        return v

    bands = {}
    for name in ("slant_deg", "stroke_px64", "pen_contrast", "ink_px_per_char", "dot_px64"):
        v = pct(name)
        bands[name] = {"p2.5": float(np.percentile(v, 2.5)), "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "p97.5": float(np.percentile(v, 97.5))}
    for r in res:
        out = lambda n: r[f"{n}_pct"] is not None and (r[f"{n}_pct"] < 0.025 or r[f"{n}_pct"] > 0.975)  # noqa: E731
        r["slant_outlier"], r["stroke_outlier"], r["pen_outlier"] = out("slant_deg"), out("stroke_px64"), out("pen_contrast")
        r["ruled"] = (r["rule_share"] or 0) > 0.02
        r["low_res"] = (r["ink_px_per_char"] < bands["ink_px_per_char"]["p5"]) or (r["dot_px64"] is not None and r["dot_px64"] < 2)
    pq.write_table(pa.Table.from_pylist(res), a.out / "features.parquet")
    (a.out / "features_reference.json").write_text(json.dumps(bands, indent=1), encoding="utf-8")
    print(json.dumps(bands, indent=1))
    for ds in ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2"):
        rs = [r for r in res if r["dataset"] == ds and r["ok"]]
        print(f"{ds:14s} measured {len(rs):5d}  " + "  ".join(f"{k} {100 * np.mean([r[k] for r in rs]):.1f}%" for k in ("slant_outlier", "stroke_outlier", "pen_outlier", "ruled", "low_res"))
              + f"  median slant {np.nanmedian([r['slant_deg'] if r['slant_deg'] is not None else np.nan for r in rs]):+.1f}°")


if __name__ == "__main__":
    main()
