"""Writer and page structure of the full KHATT v1.0 copy (checklist 2.17, map `R0.4`; feeds 2.16 / `P1.8`).

    .geomvenv/bin/python scripts/writer_khatt_pages.py [--out results/writers/khatt_pages.parquet] [--relink]

What exists: nothing. The Teklia/KHATT_v1.0 parquet has only `image` (bytes, path = None) and `text`; rows are shuffled (consecutive
rows are different writers), images are binarised (two grey levels, so no scan-background signature), and line images are tight
crops (widths are not shared within a page). The original KHATT file names (AHTD3A<writer>_Para<k>_<line>) carry writer IDs but
need a licence request (writer-adaptation D: low priority).

What can be reconstructed: almost every line is a piece of ONE fixed paragraph (CANON, 77 words, written once per form: ~1,230
paragraph instances, 187 in test). Each line is aligned to its word slots in CANON (start, end). A paragraph instance is a chain of
lines whose slots tile CANON in order, so a line ending at slot e is followed by a line starting at e + 1 written by the same hand.
Lines are linked by a greedy minimum-cost matching (predecessor -> successor) restricted to text-compatible pairs (start_j - end_i
- 1 in {-1, 0, +1}) with cost = handwriting-style distance (robust z-scores of stroke width, letter width, ink height, ink density,
slant, word gap) + a gap penalty; a pair is left unlinked when no partner beats the unlinked cost. Chains = pseudo-pages
(`page_id`; one page = one writer's paragraph). Validation without ground truth: link on half of the style features and test
whether linked pairs agree on the other half more than random text-compatible pairs (`validation` in the JSON), plus the page
gallery in khatt_fixed_text.html and, once readings exist, the page ICC in writer_khatt_fixed_text.py.

Outputs (bundled): OUT (one row per line: id, split, text_norm, start, end, slots, features, page_id, page_pos, page_len,
page_complete, link_cost, link_margin) and OUT with .json (counts, validation). Pure helpers (align, chains) are unit-tested in
writer_khatt_selftest.py.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rapidfuzz.distance import Levenshtein

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from khatt_full_panel import HUB, SNAP, SPLITS, norm_text  # noqa: E402

CANON = ("ذهب نوح مظفر ضرغام بصحبة رؤوف بن لؤي رايق ظافر عطعوط وهلال خازن عفيف للحج بدأت قوافل الحجيج حاج إثر آخر يلبي عند "
         "وصولنا طفنا وسعينا مع شيخ كان جاري في الخيمة يتكلم وهو نائم بكلمات لا أفهمها مثل انقض بغلس له الضابط لزمتك سألت راجح "
         "هل بلغ أصحابنا ظ ع ك ث خ ض ب س ش ص غ ه أننا في الحج هل تعلم فائدة الكلمات التالية لهذا النص مشمش دراق غيظ ناء بث "
         "نسر").split()
FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
FEATS = ("stroke", "letter_w", "ink_h", "density", "slant", "word_gap")
HALVES = (("stroke", "density", "slant"), ("letter_w", "ink_h", "word_gap"))


def fold(w: str) -> str:
    return w.translate(FOLD)


def word_sim(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(fold(a), fold(b))


def align(words: list[str], canon: list[str] = CANON) -> list[int | None]:
    """Semi-global word alignment of a line to the canonical paragraph (canonical ends free). Returns the canonical slot of every
    line word (None when the word is an insertion). Match score 2*sim - 1 (positive above half similar), gaps -0.6."""
    n, m = len(words), len(canon)
    if n == 0:
        return []
    gap = -0.6
    S = np.array([[2 * word_sim(w, c) - 1 for c in canon] for w in words])
    D = np.zeros((n + 1, m + 1))
    D[1:, 0] = gap * np.arange(1, n + 1)
    B = np.zeros((n + 1, m + 1), np.int8)  # 0 diag, 1 up (line word inserted), 2 left (slot skipped)
    B[1:, 0] = 1
    B[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (D[i - 1, j - 1] + S[i - 1, j - 1], D[i - 1, j] + gap, D[i, j - 1] + (gap if i < n else 0.0))
            k = int(np.argmax(opts))
            D[i, j], B[i, j] = opts[k], k
    # free leading slots: row 0 is all zeros already; free trailing slots: take the best end column in the last row
    j = int(np.argmax(D[n]))
    i, out = n, [None] * n
    while i > 0 and j > 0:
        k = B[i, j]
        if k == 0:
            out[i - 1] = j - 1 if S[i - 1, j - 1] > 0 else None
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return out


def span(slots: list[int | None]) -> tuple[int, int]:
    s = [x for x in slots if x is not None]
    return (min(s), max(s)) if s else (-1, -1)


def chains(succ: dict[int, int]) -> list[list[int]]:
    """Follow predecessor -> successor links into chains (every node appears once; isolated nodes are not listed)."""
    pred = {v: k for k, v in succ.items()}
    out = []
    for h in [k for k in succ if k not in pred]:
        c = [h]
        while c[-1] in succ:
            c.append(succ[c[-1]])
        out.append(c)
    return out


def link(start: np.ndarray, end: np.ndarray, Z: np.ndarray, unlinked: float | None = None, gap_pen: float = 0.5):
    """Min-cost matching predecessor -> successor among text-compatible pairs. Returns (succ dict, cost per linked pred, margin
    per linked pred = second-best compatible cost - chosen cost, compatible-pair costs)."""
    n = len(start)
    by_start: dict[int, list[int]] = {}
    for j, s in enumerate(start):
        by_start.setdefault(int(s), []).append(j)
    I, J, C = [], [], []
    for i in range(n):
        if end[i] < 0:
            continue
        for d in (-1, 0, 1):
            for j in by_start.get(int(end[i]) + 1 + d, []):
                if j != i:
                    I.append(i), J.append(j), C.append(float(np.abs(Z[i] - Z[j]).mean()) + gap_pen * abs(d))
    I, J, C = np.array(I), np.array(J), np.array(C)
    if unlinked is None:
        unlinked = float(np.percentile(C, 60))
    # greedy minimum-cost matching (ascending cost, each line used once as predecessor and once as successor). The exact
    # min-weight bipartite matching (scipy LAPJVsp) did not finish in 10 min on the 4,672 train lines with dummy nodes.
    succ: dict[int, int] = {}
    used_s: set[int] = set()
    for k in np.argsort(C, kind="stable"):
        i, j = int(I[k]), int(J[k])
        if C[k] >= unlinked:
            break
        if i in succ or j in used_s:
            continue
        succ[i] = j
        used_s.add(j)
    # no cycles possible: a successor always starts after its predecessor's start (start_j >= end_i >= start_i), except for
    # one-word lines with d = -1; break any cycle defensively
    for i in list(succ):
        seen, x = set(), i
        while x in succ and x not in seen:
            seen.add(x)
            x = succ[x]
        if x in seen:
            succ.pop(x)
    cost = {}
    margin = {}
    alt: dict[int, list[float]] = {}
    for i, j, c in zip(I, J, C):
        alt.setdefault(int(i), []).append(c)
    for i, j in succ.items():
        c = float(np.abs(Z[i] - Z[j]).mean()) + gap_pen * abs(int(start[j]) - int(end[i]) - 1)
        cost[i] = c
        others = sorted(alt[i])
        margin[i] = (others[1] - others[0]) if len(others) > 1 and abs(others[0] - c) < 1e-9 else (others[0] - c if others else 0.0)
    return succ, cost, margin, C


# ---- image features (binary images: ink = dark) -------------------------------------------------------------------------------

def features(img: np.ndarray, letters: int, words: int) -> dict:
    ink = img < 128
    rows, cols = np.nonzero(ink)
    if len(rows) < 20:
        return {f: np.nan for f in FEATS}
    prof = ink.sum(1).astype(float)
    cdf = np.cumsum(prof) / prof.sum()
    core = int(np.searchsorted(cdf, 0.8)) - int(np.searchsorted(cdf, 0.2)) + 1  # rows holding the middle 60 % of ink
    x0, x1 = cols.min(), cols.max()
    # stroke width: mean horizontal ink run length
    d = np.diff(np.pad(ink.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    runs = (np.nonzero(d == -1)[1] - np.nonzero(d == 1)[1])
    stroke = float(np.median(runs))
    density = float(ink.sum() / ((x1 - x0 + 1) * (rows.max() - rows.min() + 1)))
    # slant: shear (in x per y) that maximises the variance of the column projection, on a 2x downsample
    sm = ink[::2, ::2]
    yy, xx = np.nonzero(sm)
    yc = yy - yy.mean()
    best, bs = -1.0, 0.0
    for s in np.linspace(-0.6, 0.6, 25):
        h = np.bincount((xx + s * yc - (xx + s * yc).min()).astype(int))
        v = float((h.astype(float) ** 2).sum())
        if v > best:
            best, bs = v, s
    # word gap: median of the widest (words - 1) column gaps, relative to the stroke
    colink = ink[:, x0 : x1 + 1].any(0)
    e = np.diff(np.concatenate([[1], colink.astype(np.int8), [1]]))
    gaps = np.sort(np.nonzero(e == 1)[0] - np.nonzero(e == -1)[0])[::-1]
    k = max(1, words - 1)
    word_gap = float(np.median(gaps[:k])) / max(stroke, 1.0) if len(gaps) else 0.0
    return {"stroke": stroke, "letter_w": (x1 - x0 + 1) / max(1, letters), "ink_h": float(core), "density": density,
            "slant": float(bs), "word_gap": word_gap}


def robust_z(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, 0)
    mad = np.nanmedian(np.abs(X - med), 0) * 1.4826
    Z = (X - med) / np.where(mad > 0, mad, 1)
    return np.nan_to_num(np.clip(Z, -5, 5))


def logf(X: dict) -> np.ndarray:
    cols = []
    for f in FEATS:
        v = np.asarray(X[f], float)
        cols.append(np.log(np.maximum(v, 1e-3)) if f in ("stroke", "letter_w", "ink_h", "word_gap") else v)
    return np.stack(cols, 1)


def validate(start, end, Zall, reps: int = 200, seed: int = 0) -> dict:
    """Link on one half of the features; held-out half distance of linked pairs vs random text-compatible pairs."""
    rng = np.random.default_rng(seed)
    out = {}
    for a, b in (HALVES, HALVES[::-1]):
        ia = [FEATS.index(f) for f in a]
        ib = [FEATS.index(f) for f in b]
        succ, _, _, _ = link(start, end, Zall[:, ia])
        dl = np.array([np.abs(Zall[i, ib] - Zall[j, ib]).mean() for i, j in succ.items()])
        # random text-compatible pairs
        by_start: dict[int, list[int]] = {}
        for j, s in enumerate(start):
            by_start.setdefault(int(s), []).append(j)
        pairs = [(i, j) for i in range(len(start)) if end[i] >= 0 for j in by_start.get(int(end[i]) + 1, []) if j != i]
        pairs = np.array(pairs)
        dr = np.abs(Zall[pairs[:, 0]][:, ib] - Zall[pairs[:, 1]][:, ib]).mean(1)
        boots = [np.mean(rng.choice(dl, len(dl))) / np.mean(rng.choice(dr, len(dr))) for _ in range(reps)]
        out["link_on_" + "+".join(a)] = {"heldout": list(b), "linked_pairs": len(dl), "heldout_dist_linked": round(float(dl.mean()), 4),
                                         "heldout_dist_random_compatible": round(float(dr.mean()), 4),
                                         "ratio": round(float(dl.mean() / dr.mean()), 4),
                                         "ratio_ci95": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=W / "results/writers/khatt_pages.parquet")
    ap.add_argument("--relink", action="store_true", help="reuse the alignment and features in OUT; redo only the linking")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.relink:
        keep = ("id", "split", "index", "text_norm", "start", "end", "slots", "img_h", "img_w", *FEATS)
        rows = [{k: r[k] for k in keep} for r in pq.read_table(a.out).to_pylist()]
    for s in SPLITS if not rows else ():
        (f,) = glob.glob(str(HUB / SNAP.format(split=s)))
        t = pq.read_table(f)
        texts = t.column("text").to_pylist()
        imgs = t.column("image").to_pylist()
        for i, (txt, im) in enumerate(zip(texts, imgs)):
            tn = norm_text(txt)
            words = tn.split()
            sl = align(words)
            st, en = span(sl)
            g = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("L"))
            ft = features(g, len(tn.replace(" ", "")), len(words))
            rows.append({"id": f"kitab_khatt_full/{s}/{i:06d}", "split": s, "index": i, "text_norm": tn, "start": st, "end": en,
                         "slots": [-1 if x is None else x for x in sl], "img_h": g.shape[0], "img_w": g.shape[1], **ft})
        print(s, len(texts), flush=True)
    if not a.relink:
        pq.write_table(pa.Table.from_pylist(rows), a.out)  # alignment + features first, so --relink can redo only the linking
    X = logf({f: [r[f] for r in rows] for f in FEATS})
    report = {"item": "MASTER-CHECKLIST 2.17 (R0.4)", "canon_words": len(CANON), "splits": {}, "validation": {}}
    page_base = 0
    for s in SPLITS:
        ix = np.array([k for k, r in enumerate(rows) if r["split"] == s])
        Z = robust_z(X[ix])
        start = np.array([rows[k]["start"] for k in ix])
        end = np.array([rows[k]["end"] for k in ix])
        succ, cost, margin, C = link(start, end, Z)
        ch = chains(succ)
        linked = {n for c in ch for n in c}
        ch += [[n] for n in range(len(ix)) if n not in linked]
        ch.sort(key=lambda c: (start[c[0]], c[0]))
        for p, c in enumerate(ch):
            comp = bool(start[c[0]] <= 1 and end[c[-1]] >= len(CANON) - 2 and
                        all(abs(int(start[c[q + 1]]) - int(end[c[q]]) - 1) == 0 for q in range(len(c) - 1)))
            for q, n in enumerate(c):
                r = rows[ix[n]]
                r.update({"page_id": f"{s}/p{page_base + p:05d}", "page_pos": q, "page_len": len(c), "page_complete": comp,
                          "link_cost": round(cost.get(n, np.nan), 4), "link_margin": round(margin.get(n, np.nan), 4)})
        page_base += len(ch)
        lens = np.array([len(c) for c in ch])
        report["splits"][s] = {"lines": int(len(ix)), "aligned": int((start >= 0).sum()),
                               "paragraph_starts": int((start == 0).sum()), "paragraph_ends": int((end == len(CANON) - 1).sum()),
                               "links": len(succ), "pages": len(ch), "pages_len_ge2": int((lens >= 2).sum()),
                               "pages_complete": int(sum(rows[ix[c[0]]]["page_complete"] for c in ch)),
                               "page_len_hist": {int(k): int(v) for k, v in zip(*np.unique(lens, return_counts=True))},
                               "compatible_pairs": int(len(C)), "median_link_cost": round(float(np.median(list(cost.values()))), 4),
                               "median_compatible_cost": round(float(np.median(C)), 4)}
        report["validation"][s] = validate(start, end, Z)
        print(s, json.dumps(report["splits"][s]), flush=True)
    tab = pa.Table.from_pylist(rows)
    pq.write_table(tab, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["validation"], indent=1))


if __name__ == "__main__":
    main()
