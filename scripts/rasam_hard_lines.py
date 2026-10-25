"""Which official RASAM test lines are hardest, what they have in common, what the LM fixes and breaks, and a free ensemble test.

    PYTHONPATH=src python scripts/rasam_hard_lines.py --tag r1|r2 --best LM_JSON --others JSON [JSON ...] [--top 40]

--best    the best system's result file (a results/rasam/lm/*.json: lines carry "hyp" with LM and "greedy" without)
--others  other systems on the same test set (results/rasam/{crnn,lm}/*.json), used for "hard for everyone" and the ensemble
Outputs (results/rasam/hard_lines/): <tag>_features.json (per-line features and CERs), <tag>_report.txt (tables), and an HTML gallery of the hardest
lines with their images (also copied to ~/colab_staging/dashboard/hard_lines_<tag>.html).
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
HARAKAT = re.compile("[ً-ْٰ]")
DOTTED = set("بتثنيجخذزشضظغفقة")
HAMZA = set("ءأإؤئآ")


def counts(ref: str, hyp: str) -> tuple[int, int]:
    c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp, "raw").strip())
    return c["sub"] + c["del"] + c["ins"], c["ref_len"]


def cer(pairs) -> float:
    e = n = 0
    for a, b in pairs:
        x, y = counts(a, b)
        e += x
        n += y
    return e / max(1, n)


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def image_features(img: Image.Image) -> dict:
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    g = rgb.mean(axis=2)
    mask = g < 12
    paper = g[~mask]
    r, gg, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    colored = ((r > gg + 40) & (r > b + 40)) | ((b > r + 30) & (b > gg + 10))  # red or blue ink (rubrics)
    dark = (~mask) & (g < (np.median(paper) - 60 if paper.size else 100))
    return {"h": img.size[1], "w": img.size[0], "mask_frac": float(mask.mean()), "colored_frac": float(colored[~mask].mean()) if (~mask).any() else 0.0,
            "ink_frac": float(dark.mean()), "contrast": float(paper.std()) if paper.size else 0.0}


def text_features(t: str) -> dict:
    letters = [c for c in t if c.isalpha()]
    return {"n_chars": len(t), "n_words": len(t.split()), "n_diacritics": len(HARAKAT.findall(t)), "has_hash": "#" in t,
            "has_digit": any(c.isdigit() for c in t), "dotted_frac": sum(c in DOTTED for c in letters) / max(1, len(letters)),
            "hamza": sum(c in HAMZA for c in t)}


def quintile_table(rows: list[dict], key: str, label: str) -> str:
    vals = np.array([r[key] for r in rows], dtype=float)
    if np.allclose(vals, vals[0]):
        return ""
    qs = np.unique(np.quantile(vals, [0, .2, .4, .6, .8, 1]))
    out = f"  {label}:"
    for k in range(len(qs) - 1):
        sel = [r for r in rows if qs[k] <= r[key] <= qs[k + 1]] if k == len(qs) - 2 else [r for r in rows if qs[k] <= r[key] < qs[k + 1]]
        if sel:
            out += f"  [{qs[k]:.3g}-{qs[k + 1]:.3g}] {cer([(r['ref'], r['hyp']) for r in sel]) * 100:.1f}% (n={len(sel)})"
    return out + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["r1", "r2"])
    ap.add_argument("--best", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    ds = {"r1": "RASAM-1", "r2": "RASAM-2"}[a.tag]
    recs = []
    for f in sorted(glob.glob(f"{ROOT}{ds}/snapshots/*/data/test-*.parquet")):
        recs += pq.read_table(f).to_pylist()
    best = json.load(open(a.best))["lines"]
    others = {Path(f).stem.replace("__test", ""): json.load(open(f))["lines"] for f in a.others}
    assert len(best) == len(recs), (len(best), len(recs))
    rows = []
    for i, (r, l) in enumerate(zip(recs, best)):
        e, n = counts(l["ref"], l["hyp"])
        eg, _ = counts(l["ref"], l.get("greedy", l["hyp"]))
        row = {"i": i, "ref": l["ref"], "hyp": l["hyp"], "greedy": l.get("greedy", l["hyp"]), "cer": e / max(1, n), "cer_greedy": eg / max(1, n),
               "manuscript": r["manuscript_name"], "region": r["region_type"], "page": r["image_name"], **text_features(l["ref"]),
               **image_features(Image.open(io.BytesIO(r["image"]["bytes"])))}
        row["px_per_char"] = row["w"] * 96 / max(1, row["h"]) / max(1, row["n_chars"])
        row["other_cer"] = {k: counts(l["ref"], v[i]["hyp"])[0] / max(1, n) for k, v in others.items() if len(v) == len(best)}
        rows.append(row)
    out_dir = Path("results/rasam/hard_lines")
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = [f"{ds} test, best system {Path(a.best).stem}: CER with LM {cer([(r['ref'], r['hyp']) for r in rows]) * 100:.2f} %, greedy {cer([(r['ref'], r['greedy']) for r in rows]) * 100:.2f} % (n={len(rows)})\n"]
    # what the LM does
    better = sum(r["cer"] < r["cer_greedy"] for r in rows)
    worse = sum(r["cer"] > r["cer_greedy"] for r in rows)
    rep.append(f"LM vs greedy: {better} lines better, {worse} worse, {len(rows) - better - worse} unchanged\n")
    # concentration of errors
    errs = sorted((counts(r["ref"], r["hyp"])[0] for r in rows), reverse=True)
    tot = max(1, sum(errs))
    for frac in (0.05, 0.10, 0.25):
        k = max(1, int(len(errs) * frac))
        rep.append(f"the worst {frac:.0%} of lines hold {sum(errs[:k]) / tot:.0%} of all character errors\n")
    rep.append("\nCER by feature (quintiles of the feature; micro CER of the best system):\n")
    for key, label in (("n_chars", "reference length (chars)"), ("h", "crop height px"), ("px_per_char", "horizontal px per char at 96 px height"),
                       ("ink_frac", "dark-ink fraction"), ("contrast", "paper contrast (std)"), ("colored_frac", "red/blue ink fraction"),
                       ("dotted_frac", "share of dotted letters"), ("n_diacritics", "diacritics in reference")):
        rep.append(quintile_table(rows, key, label))
    for key in ("region", "manuscript"):
        rep.append(f"  by {key}: " + ", ".join(f"{k}: {cer([(r['ref'], r['hyp']) for r in rows if r[key] == k]) * 100:.1f}% (n={sum(r[key] == k for r in rows)})"
                                           for k in sorted({r[key] for r in rows})) + "\n")
    for key, label in (("has_hash", "contains '#' (illegible placeholder)"), ("has_digit", "contains digits")):
        sel = [r for r in rows if r[key]]
        if sel:
            rep.append(f"  {label}: {cer([(r['ref'], r['hyp']) for r in sel]) * 100:.1f}% (n={len(sel)}) vs others "
                       f"{cer([(r['ref'], r['hyp']) for r in rows if not r[key]]) * 100:.1f}%\n")
    # hard for everyone vs only for the best system
    if others:
        allhard = [r for r in rows if r["cer"] > 0.3 and r["other_cer"] and all(v > 0.3 for v in r["other_cer"].values())]
        onlybest = [r for r in rows if r["cer"] > 0.3 and r["other_cer"] and min(r["other_cer"].values()) < 0.15]
        rep.append(f"\nlines with CER > 30 % for the best system: {sum(r['cer'] > 0.3 for r in rows)}; of those hard (>30 %) for EVERY other system too: {len(allhard)}; "
                   f"read well (<15 %) by at least one other system: {len(onlybest)}\n")
        # free ensemble: per line take the reading closest to all others (medoid), no reference used
        names = ["best"] + list(others)
        hyps = [[r["hyp"]] + [others[k][r["i"]]["hyp"] for k in others] for r in rows]
        pick = []
        for hs in hyps:
            d = [sum(lev(x, y) for y in hs) for x in hs]
            pick.append(hs[int(np.argmin(d))])
        ens = cer([(r["ref"], p) for r, p in zip(rows, pick)])
        oracle = cer([(r["ref"], min(hs, key=lambda h: counts(r["ref"], h)[0])) for r, hs in zip(rows, hyps)])
        rep.append(f"\nensemble by consensus (medoid of {len(names)} readings, no reference used): {ens * 100:.2f} %; oracle (best reading per line, uses the reference): "
                   f"{oracle * 100:.2f} %; systems: {', '.join(names)}\n")
    worst = sorted(rows, key=lambda r: -r["cer"])[: a.top]
    rep.append(f"\nworst {a.top} lines (CER, manuscript, region, crop h x w, red/blue ink, length):\n")
    for r in worst[:15]:
        rep.append(f"  {r['cer'] * 100:5.0f}% {r['manuscript']} {r['region']} {r['h']}x{r['w']} col={r['colored_frac']:.2f} n={r['n_chars']} | REF {r['ref'][:60]} | HYP {r['hyp'][:60]}\n")
    text = "".join(rep)
    (out_dir / f"{a.tag}_report.txt").write_text(text)
    (out_dir / f"{a.tag}_features.json").write_text(json.dumps([{k: v for k, v in r.items()} for r in rows], ensure_ascii=False))
    print(text)
    # gallery
    cards = []
    for r in worst:
        im = Image.open(io.BytesIO(recs[r["i"]]["image"]["bytes"])).convert("RGB")
        im.thumbnail((1100, 160))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        oc = " · ".join(f"{k}: {v * 100:.0f}%" for k, v in r["other_cer"].items())
        cards.append(f"<div class=c><div class=m>{r['cer'] * 100:.0f}% CER (greedy {r['cer_greedy'] * 100:.0f}%) · {html.escape(r['manuscript'])} · {html.escape(r['region'])} · "
                     f"{r['h']}x{r['w']} px · red/blue ink {r['colored_frac']:.2f}<br><span class=d>other systems: {html.escape(oc)}</span></div>"
                     f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
                     f"<div dir=rtl class=t><b>GT</b> {html.escape(r['ref'])}</div><div dir=rtl class=t><b>OURS</b> {html.escape(r['hyp'])}</div></div>")
    page = ("<!doctype html><html><head><meta charset=utf-8><title>Hardest lines</title><style>body{font:14px system-ui;margin:16px;max-width:1150px;background:#fbfaf7;color:#1d1c1a}"
            "@media(prefers-color-scheme:dark){body{background:#161513;color:#ece9e2}}.c{border:1px solid #8884;border-radius:8px;padding:8px;margin:10px 0}img{max-width:100%;display:block;margin:6px 0}"
            ".m{font-size:12px}.d{opacity:.7}.t{font-size:17px;margin:2px 0}</style></head><body>"
            f"<h1>{ds} test: the {a.top} hardest lines for {html.escape(Path(a.best).stem)}</h1><pre>{html.escape(text)}</pre>{''.join(cards)}</body></html>")
    (out_dir / f"hard_lines_{a.tag}.html").write_text(page)
    dash = Path.home() / "colab_staging/dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    (dash / f"hard_lines_{a.tag}.html").write_text(page)


if __name__ == "__main__":
    main()
