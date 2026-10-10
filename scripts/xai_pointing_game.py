"""Pointing game for the attribution maps against CTC character spans (X-B2).

    python scripts/xai_pointing_game.py --spans spans.json --run letters=DIR --run errors=DIR --out pointing.json

DIR is an output dir of xai_attribute_qwen3vl.py (per-line NNN/tokens.json + maps.npz).
For each scored token the target region is the union of the CTC spans of the ground-truth characters that
the token covers (aligned character by character with difflib; for a wrong token this is where the
correct letters are). A map's score is the share of its column profile that falls inside the region
(+/- a margin), divided by the same share for the region shifted to random places (1 = chance).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")
METHODS = ["attn", "attn_rel", "gxi"]


def clean_keep(t: str):
    """Strip marks and collapse whitespace; return the cleaned string and, per cleaned char, its offset in t."""
    t = unicodedata.normalize("NFC", t)
    chars, offs, prev_space = [], [], True
    for i, ch in enumerate(t):
        if STRIP.match(ch):
            continue
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        chars.append(ch)
        offs.append(i)
    while chars and chars[-1] == " ":
        chars.pop(); offs.pop()
    return "".join(chars), offs


def profile(maps: np.lib.npyio.NpzFile, method: str, t: int, n: int = 512) -> np.ndarray:
    """Positive column profile resampled to n bins over the image width."""
    if method == "gxi":
        v = np.clip(maps[f"gxicol_{t}"], 0, None)
    else:
        v = np.clip(maps[f"{method}_{t}"], 0, None).sum(0)
    v = np.interp(np.linspace(0, len(v) - 1, n), np.arange(len(v)), v)
    s = v.sum()
    return v / s if s > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, help="name=dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.015, help="slack on each side of the region, as a share of the line width")
    ap.add_argument("--min-chars", type=int, default=2, help="skip tokens with fewer aligned ground-truth characters")
    a = ap.parse_args()
    spans = {s["id"]: s for s in json.loads(a.spans.read_text(encoding="utf-8"))}
    rng = np.random.default_rng(0)
    n = 512
    recs = []
    skipped = {"no_spans": 0, "too_few_chars": 0, "tokens": 0}
    for spec in a.run:
        name, d = spec.split("=", 1)
        for ld in sorted(Path(d).glob("[0-9][0-9][0-9]")):
            tj = json.loads((ld / "tokens.json").read_text(encoding="utf-8"))
            sp = spans.get(tj["id"])
            if sp is None or not (ld / "maps.npz").exists():
                skipped["no_spans"] += 1
                continue
            maps = np.load(ld / "maps.npz")
            gt_s, gt_off = clean_keep(sp["label"])
            hyp_s, hyp_off = clean_keep(tj["hyp"])
            if len(gt_s) != len(sp["chars"]):
                skipped["no_spans"] += 1
                continue
            sm = difflib.SequenceMatcher(None, hyp_s, gt_s, autojunk=False)
            hyp2gt: dict[int, list[int]] = {}
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    for k in range(i2 - i1):
                        hyp2gt[i1 + k] = [j1 + k]
                elif tag == "replace":
                    for k in range(i1, i2):
                        hyp2gt[k] = list(range(j1, j2))
            for tok in tj["tokens"]:
                skipped["tokens"] += 1
                idx = [i for i, o in enumerate(hyp_off) if tok["start"] <= o < tok["end"] and hyp_s[i] != " "]
                gts = sorted({j for i in idx for j in hyp2gt.get(i, [])})
                gts = [j for j in gts if gt_s[j] != " "]
                if len(gts) < a.min_chars:
                    skipped["too_few_chars"] += 1
                    continue
                x0 = min(sp["chars"][j]["x0"] for j in gts)
                x1 = max(sp["chars"][j]["x1"] for j in gts)
                lo, hi = max(0.0, x0 - a.margin), min(1.0, x1 + a.margin)
                wbins = max(1, int(round((hi - lo) * n)))
                lo_b = int(lo * n)
                rec = {"panel": name, "id": tj["id"], "t": tok["t"], "err": tok["err"], "width": hi - lo, "n_chars": len(gts), "score": {}, "inside": {}}
                for m in METHODS:
                    p = profile(maps, m, tok["t"], n)
                    inside = float(p[lo_b : lo_b + wbins].sum())
                    shifts = rng.integers(0, n, 40)
                    base = float(np.mean([np.roll(p, -s)[lo_b : lo_b + wbins].sum() for s in shifts]))
                    rec["inside"][m] = inside
                    rec["score"][m] = inside / base if base > 0 else None
                    rec.setdefault("hit", {})[m] = bool(lo_b <= int(np.argmax(p)) < lo_b + wbins)
                recs.append(rec)

    def summ(sel):
        out = {"n": len(sel), "chance_inside": float(np.mean([r["width"] for r in sel])) if sel else None}
        for m in METHODS:
            sc = [r["score"][m] for r in sel if r["score"][m] is not None]
            out[m] = {"mean_inside": float(np.mean([r["inside"][m] for r in sel])) if sel else None,
                      "median_enrichment": float(np.median(sc)) if sc else None,
                      "hit_rate": float(np.mean([r["hit"][m] for r in sel])) if sel else None}
        return out

    res = {"margin": a.margin, "skipped": skipped, "all": summ(recs), "correct": summ([r for r in recs if not r["err"]]),
           "wrong": summ([r for r in recs if r["err"]]), "by_panel": {p: summ([r for r in recs if r["panel"] == p]) for p in sorted({r["panel"] for r in recs})},
           "records": recs}
    a.out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
