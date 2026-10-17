"""Rules behind detectors D4, D21 and D23 of run_detectors.py (data-and-label-audit §9, checklist 1.5).

    .geomvenv/bin/python scripts/detectors_extra.py [--out results/store/intrusion.parquet]   # D4 image measurements

D4  neighbour-line intrusion, measured on the as-stored 64 px panel image (panel.pkl and every panel_extra pickle of
    configs/analysis/sources.json). Otsu ink; rows / columns that are mostly ink (scan borders) and long thin
    components (ruling) are dropped. The line's core rows are those whose smoothed ink share is ≥ half the peak.
      edge intrusion   ≥ 4 components touch the top or bottom border while lying wholly outside the core, together
                       covering ≥ 30 % of the ink span and holding ≥ 3 % of the ink (descenders / ascenders of the
                       neighbouring lines);
      several baselines two ink-row peaks ≥ 30 % of the height apart, an almost empty gap (< 5 % of the peak) between
                       them, each spreading over ≥ 30 % of the ink span.
    Heuristic: on 26 random flagged lines about two in three show a neighbour line; the misses are ruled-paper
    fragments, Latin capitals and lone numbers. Prevalence: KITAB-KHATT 0.5 %, Muharaf 7 %, RASAM 22-27 %, NAKBA 33 %.
D21 lam-alef ligature: on the letters view, a GT lam followed by an alef (ا أ إ آ) is a ligature; it is misread when the
    lam or the alef is not aligned to itself (Levenshtein opcodes, as analyze_letters.outcomes).
D23 memorisation signature: GT quirks a reading has no visual reason to reproduce.
      whitespace quirks of the raw GT: a double space, a space before punctuation, a detached conjunction و;
      quirk words: GT words (letters view) that ≤ 10 % of the competent reference readings contain although the line is
      read well (≥ 5 references, ≥ 60 % of them at line CER ≤ 0.3); typos of the transcriber, mostly.
    A quirk is an opportunity when the reading has its context with whitespace removed, and reproduced when the
    reading has it verbatim (letters view per token, whitespace kept).
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy import ndimage

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from arabic_ocr_bench.characterize.samples import _letters_view  # noqa: E402

# ---- D4 ---------------------------------------------------------------------------------------------------------


def otsu(img: np.ndarray) -> int:
    h = np.histogram(img, 256, (0, 256))[0].astype(float)
    p = h / max(1.0, h.sum())
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return int(np.argmax(sb))


def intrusion(img: np.ndarray) -> dict:
    """D4 measurements of one grey line image (dark ink on light paper)."""
    ink = img <= otsu(img)  # Otsu class 0 is [0, t]
    H, Wd = ink.shape
    ink[ink.mean(1) > 0.5, :] = False  # scan borders and solid bars
    ink[:, ink.mean(0) > 0.85] = False
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    sl = ndimage.find_objects(lab)
    keep = np.ones(n + 1, bool)
    keep[0] = False
    for i, s in enumerate(sl):
        h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
        if (h >= 0.5 * H and w <= 0.08 * H) or (w >= 0.5 * Wd and h <= 0.08 * H):  # ruling
            keep[i + 1] = False
    ink = keep[lab]
    xs = np.where(ink.any(0))[0]
    if ink.sum() < 50 or len(xs) == 0:
        return {"ok": False, "n_edge": 0, "edge_cov": 0.0, "edge_share": 0.0, "multi": False}
    span = xs[-1] - xs[0] + 1
    sm = np.convolve(ink.mean(1), np.ones(5) / 5, "same")
    core = np.where(sm >= 0.5 * sm.max())[0]
    a, c = int(core.min()), int(core.max())
    cov, ne, epx, tot = np.zeros(Wd, bool), 0, 0, int(ink.sum())
    for i, s in enumerate(sl):
        if not keep[i + 1]:
            continue
        y0, y1 = s[0].start, s[0].stop
        if (y0 <= 1 or y1 >= H - 1) and (y1 <= a or y0 > c):
            ne += 1
            cov[s[1]] = True
            epx += int((lab[s] == i + 1).sum())
    pk = [i for i in range(1, H - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] >= 0.4 * sm.max()]
    multi = any(q - p >= 0.3 * H and sm[p:q].min() < 0.05 * sm.max()
                and ink[max(0, p - 3):p + 4].any(0).sum() >= 0.3 * span and ink[max(0, q - 3):q + 4].any(0).sum() >= 0.3 * span
                for p in pk for q in pk)
    return {"ok": True, "n_edge": ne, "edge_cov": float(cov.sum() / span), "edge_share": epx / tot, "multi": bool(multi)}


def d4_flag(m: dict) -> bool:
    return bool(m["ok"]) and ((m["n_edge"] >= 4 and m["edge_cov"] >= 0.3 and m["edge_share"] >= 0.03) or bool(m["multi"]))


# ---- D21 --------------------------------------------------------------------------------------------------------
ALEFS = "اأإآ"


def lam_alef(gt: str, hyp: str) -> tuple[int, int]:
    """(lam-alef ligatures in the GT, ligatures misread) on the letters view."""
    g, p = _letters_view(gt), _letters_view(hyp)
    pairs = [i for i in range(len(g) - 1) if g[i] == "ل" and g[i + 1] in ALEFS]
    if not pairs:
        return 0, 0
    ok = [False] * len(g)
    for op in Levenshtein.opcodes(g, p):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                ok[i] = True
    return len(pairs), sum(not (ok[i] and ok[i + 1]) for i in pairs)


def _aligned_ok(g: str, p: str) -> list[bool]:
    ok = [False] * len(g)
    for op in Levenshtein.opcodes(g, p):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                ok[i] = True
    return ok


def lam_alef_letters(gt: str, hyp: str) -> tuple[int, int]:
    """(lam and alef letters inside a ligature, of them misread): the per-letter rate compared with the base rate."""
    g, p = _letters_view(gt), _letters_view(hyp)
    lig = sorted({j for i in range(len(g) - 1) if g[i] == "ل" and g[i + 1] in ALEFS for j in (i, i + 1)})
    if not lig:
        return 0, 0
    ok = _aligned_ok(g, p)
    return len(lig), sum(not ok[i] for i in lig)


def lam_and_alef_elsewhere(gt: str, hyp: str) -> tuple[int, int]:
    """(lam and alef letters outside a ligature, of them misread): the base rate D21 is compared with."""
    g, p = _letters_view(gt), _letters_view(hyp)
    lig = {j for i in range(len(g) - 1) if g[i] == "ل" and g[i + 1] in ALEFS for j in (i, i + 1)}
    idx = [i for i, ch in enumerate(g) if (ch == "ل" or ch in ALEFS) and i not in lig]
    if not idx:
        return 0, 0
    ok = [False] * len(g)
    for op in Levenshtein.opcodes(g, p):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                ok[i] = True
    return len(idx), sum(not ok[i] for i in idx)


# ---- D23 --------------------------------------------------------------------------------------------------------
WS_QUIRKS = (("double space", re.compile(r"(?<=\S)( {2,})(?=\S)")),
             ("space before punctuation", re.compile(r"(?<=\S)( )(?=[،؛؟.,:!])")),
             ("detached و", re.compile(r"(?:(?<=\s)|^)و( )(?=[ء-ي])")))


def view_ws(s: str) -> str:
    """Letters view of each token, whitespace runs kept as they are (tabs and newlines as spaces)."""
    return "".join(t.replace("\t", " ").replace("\n", " ") if t.isspace() else _letters_view(t) for t in re.split(r"(\s+)", s or ""))


def nows(s: str) -> str:
    return re.sub(r"\s+", "", _letters_view(s or ""))


def ws_quirks(gt: str) -> list[tuple[str, str]]:
    """[(kind, context)] for every whitespace quirk in the raw GT; context = the quirk with up to 3 characters each side."""
    out = []
    for kind, rx in WS_QUIRKS:
        for m in rx.finditer(gt):
            s, e = m.span(1)
            ctx = view_ws(gt[max(0, s - 3):e + 3]).strip()
            if len(nows(ctx)) >= 3:
                out.append((kind, ctx))
    return out


def quirk_hits(quirks: list[tuple[str, str]], hyp: str) -> tuple[int, int]:
    """(opportunities, reproduced) of whitespace quirks in one reading."""
    hv, hn = view_ws(hyp), nows(hyp)
    opp = rep = 0
    for _, ctx in quirks:
        if nows(ctx) in hn:
            opp += 1
            rep += ctx in hv
    return opp, rep


def quirk_words(gt: str, refs: list[str], ref_cer: list[float], max_share: float = 0.1, min_refs: int = 5) -> set[str]:
    """GT words (letters view, ≥ 2 letters) that ≤ max_share of the reference readings contain, on a line most references
    read well (≥ 60 % at line CER ≤ 0.3)."""
    if len(refs) < min_refs or sum(c <= 0.3 for c in ref_cer) < 0.6 * len(refs):
        return set()
    sets = [set(_letters_view(h).split()) for h in refs]
    return {w for w in set(_letters_view(gt).split()) if len(w) >= 2 and sum(w in s for s in sets) <= max_share * len(refs)}


# ---- D4 measurement over the panel pickles ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=W / "configs/analysis/sources.json")
    ap.add_argument("--out", type=Path, default=W / "results/store/intrusion.parquet")
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    S = json.loads(a.sources.read_text(encoding="utf-8"))
    rows = []
    for f in [S["panel"], *S.get("panel_extra", [])]:
        if not (W / f).exists():
            continue
        for r in pickle.load(open(W / f, "rb"))["rows"]:
            if "__" in r["dataset"]:  # measure the as-stored image only
                continue
            m = intrusion(r["img"])
            rows.append({"sample_key": r["id"], "dataset": r["dataset"], **m, "d4": d4_flag(m)})
        print("measured", f, len(rows), flush=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out)
    ds = sorted({r["dataset"] for r in rows})
    for d in ds:
        x = [r["d4"] for r in rows if r["dataset"] == d]
        print(f"  {d:18s} {len(x):5d} lines  D4 {100 * np.mean(x):5.1f}%")


if __name__ == "__main__":
    main()
