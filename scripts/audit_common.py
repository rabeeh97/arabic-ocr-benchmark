"""Shared loaders for the data-and-label audit scripts (checklist 2.8-2.12; docs/research/designs/data-and-label-audit.md).

    from audit_common import load_samples, load_systems, norm_text, panel_images, OUT

load_systems(rung_cols)  per dataset, {run_id: {sample_key: row}} at transform "none", one pack per run (the pack with the
                         most lines), lines the run trained on dropped (in_train == "yes", CORRECTIONS C4), AIN-7B rows
                         (C6), reversed-output engines (C5: paddleocr, trocr) and runs with < 50 lines left out, as in the
                         A3 matrix of analyze_lines.py.
norm_text(s)             the leakage / spelling text key: ladder rung no_diacritics (canonical, tatweel and digits folded,
                         orthographic folds, harakat stripped), punctuation removed, whitespace collapsed.
"""
from __future__ import annotations

import base64
import io
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from arabic_ocr_bench.characterize import errors  # noqa: E402

OUT = W / "results" / "audit"
STORE = W / "results" / "store"
PANEL = W / "results" / "panel" / "panel.pkl"
KHATT_FULL_PANEL = W / "results" / "panel" / "khatt_full_panel.pkl"
TEST_SETS = ("kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2")
EXCLUDE = ("ain", "paddleocr", "trocr")
_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def norm_text(s: str) -> str:
    s = errors.ladder_strings(s or "")[4]
    return " ".join(_PUNCT.sub(" ", s).split())


def load_samples() -> dict[str, dict]:
    return {r["sample_key"]: r for r in pq.read_table(STORE / "samples.parquet").to_pylist()}


def load_systems(extra_cols: tuple[str, ...] = (), min_lines: int = 50) -> dict[str, dict[str, dict[str, dict]]]:
    cols = ["run_id", "family", "transform", "sample_key", "dataset", "pack", "in_train", "hyp", *extra_cols]
    t = pq.read_table(STORE / "lines.parquet", columns=list(dict.fromkeys(cols))).to_pylist()
    by = defaultdict(lambda: defaultdict(dict))  # (ds, run) -> pack -> key -> row
    for r in t:
        if r["transform"] != "none" or r["in_train"] == "yes" or any(x in r["run_id"].lower() for x in EXCLUDE):
            continue
        by[(r["dataset"], r["run_id"])][r["pack"]][r["sample_key"]] = r
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (ds, run), packs in by.items():
        best = max(packs.values(), key=len)
        if len(best) >= min_lines:
            out[ds][run] = best
    return out


def panel_images(keys: set[str] | None = None) -> dict[str, np.ndarray]:
    """sample_key -> the as-stored 64 px line image (panel.pkl and the full KHATT test panel)."""
    imgs = {}
    for p in (PANEL, KHATT_FULL_PANEL):
        if not p.exists():
            continue
        for r in pickle.load(open(p, "rb"))["rows"]:
            if r["dataset"].endswith("__inkcrop"):
                continue
            if keys is None or r["id"] in keys:
                imgs[r["id"]] = r["img"]
    return imgs


def png_b64(img: np.ndarray, max_w: int = 1400) -> str:
    from PIL import Image
    im = Image.fromarray(np.asarray(img).astype(np.uint8))
    if im.width > max_w:
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))))
    b = io.BytesIO()
    im.save(b, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def boot_ci(x: np.ndarray, stat=np.mean, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    bs = [stat(x[rng.integers(0, n, n)]) for _ in range(reps)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--acc:#2b6cb0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--acc:#7fb0ff}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1500px;padding:16px}}
table{{border-collapse:collapse;margin:8px 0 18px;font-size:13px}}td,th{{border:1px solid var(--line);padding:3px 7px;text-align:left;vertical-align:top}}
.ar{{direction:rtl;font-size:17px;font-family:'Noto Naskh Arabic','Geeza Pro',serif}}img{{max-width:100%;display:block;background:#fff}}
.card{{border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:8px 0}}.mut{{color:var(--mut)}}h2{{margin-top:28px}}
</style></head><body>"""
