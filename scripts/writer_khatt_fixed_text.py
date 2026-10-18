"""KHATT fixed-text natural experiment: same text, many writers (checklist 2.16, map `P1.8`; design writer-adaptation D).

    .geomvenv/bin/python scripts/writer_khatt_fixed_text.py [--datasets kitab_khatt_full kitab_khatt] [--out-dir results/writers]

Almost every KHATT v1.0 line is a piece of one fixed 77-word paragraph (writer_khatt_pages.CANON), so the text is held constant
and what varies between lines is the writer (and scan). Readings come from the store (results/store/lines.parquet): the CRNN panel
readings (results/panel/out/ for the 200 KITAB-KHATT lines, results/panel/khatt_full_out/ for the 1,038 v1.0 test lines once
store_build has ingested them) and, for KITAB-KHATT, every VLM that read >= 95 % of the fixed-text lines.

Decision rule (written before the run; the item is descriptive, gate V3 natural experiment):
  D1  "Writer variance exceeds text variance" is claimed for a reader when the writer partial R^2 of the additive word-error model
      (line + word slot) has a 95 % bootstrap interval entirely above the text partial R^2 interval; the reverse claim the other
      way round; otherwise "not separable".
  D2  A style feature "moves errors with the text held constant" when its within-text Spearman correlation with the line's
      residual word error survives BH-FDR q = 0.1 over features x readers (eval_stats.bh).
  D3  The reconstructed pages (writer_khatt_pages.py, 2.17) are usable as clusters for intervals when the page ICC of the residual
      line error is > 0 with its permutation p < 0.05 for the primary reader; otherwise intervals stay line-level.

Units per line (evaluation-standards item 6): CER at the CRNN-native rung (CRNN) or no-diacritics rung (VLM), WER on the letters
view, and the word-slot error matrix Y[line, slot] (1 = the GT word in that slot is not read exactly). Analyses per reader:
  variance  additive two-way fit of Y: adjusted R^2 text only (slot means), writer only (line means), both; partial shares; split-half
            reliabilities (Spearman-Brown) of writer difficulty (odd vs even slots of a line) and of word difficulty (random line
            halves). Line bootstrap, and page-cluster bootstrap where pages exist.
  same_text one-way ICC of line CER over exact-text groups (>= 2 lines): the text share; 1 - ICC is writer + scan.
  pages     ICC of the residual line error (observed - leave-one-out expected from slot difficulty) over reconstructed pages,
            with a page-label permutation null.
  style     within-text Spearman of the residual vs each style feature (writer_khatt_pages.FEATS), bootstrap p, BH-FDR.
  slots     per-word error rate with line-bootstrap p against the reader's mean, BH-FDR: the hard words of the paragraph.
Outputs (bundled): OUT_DIR/khatt_fixed_text.json (every number), OUT_DIR/khatt_fixed_text.md (tables), and
OUT_DIR/khatt_fixed_text.html (gallery: the largest same-text group best / worst / random with readings, writer-driven lines,
and reconstructed pages).
"""
from __future__ import annotations

import argparse
import base64
import collections
import glob
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from rapidfuzz.distance import Levenshtein
from scipy.stats import spearmanr

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from eval_stats import bh, boot_p  # noqa: E402
from khatt_full_panel import HUB, SNAP, norm_text  # noqa: E402
from writer_khatt_pages import CANON, FEATS, align  # noqa: E402

PANEL_SOURCES = ("results/panel/out/", "results/panel/khatt_full_out/")
PRIMARY = "crnn/base"
MIN_SLOTS = 3


# ---- pure pieces (tested in writer_khatt_selftest.py) ---------------------------------------------------------------------------

def word_errors(gt: str, hyp: str) -> list[int]:
    """1 per GT word (letters view) unless the word lies in an 'equal' block of the word-level alignment with the reading."""
    g, h = norm_text(gt).split(), norm_text(hyp).split()
    err = [1] * len(g)
    for tag, i1, i2, _, _ in Levenshtein.opcodes(g, h):
        if tag == "equal":
            for i in range(i1, i2):
                err[i] = 0
    return err


def additive_fit(Y: np.ndarray, M: np.ndarray, iters: int = 30) -> dict:
    """Y lines x slots (values where M); returns R^2 of slot-only, line-only and additive line + slot models."""
    y = Y[M]
    tot = ((y - y.mean()) ** 2).sum()
    if tot == 0:
        return {"r2_text": 0.0, "r2_writer": 0.0, "r2_both": 0.0, "partial_text": 0.0, "partial_writer": 0.0, "r2_both_raw": 0.0}
    mu = y.mean()

    def col_means(R):
        return np.where(M.sum(0) > 0, (R * M).sum(0) / np.maximum(M.sum(0), 1), 0.0)

    def row_means(R):
        return np.where(M.sum(1) > 0, (R * M).sum(1) / np.maximum(M.sum(1), 1), 0.0)

    R0 = np.where(M, Y - mu, 0.0)
    b = col_means(R0)
    r2_text = 1 - (((R0 - b[None]) * M) ** 2).sum() / tot
    a = row_means(R0)
    r2_writer = 1 - (((R0 - a[:, None]) * M) ** 2).sum() / tot
    a = np.zeros(Y.shape[0])
    b = np.zeros(Y.shape[1])
    for _ in range(iters):
        b = col_means(R0 - a[:, None])
        a = row_means(R0 - b[None])
    r2_both = 1 - (((R0 - a[:, None] - b[None]) * M) ** 2).sum() / tot
    # adjusted R^2 (degrees of freedom: observed slots - 1, observed lines - 1): line and slot means fit noise by chance
    N = int(M.sum())
    pt, pw = int((M.sum(0) > 0).sum()) - 1, int((M.sum(1) > 0).sum()) - 1

    def adj(r2, p):
        return 1 - (1 - r2) * (N - 1) / max(N - p - 1, 1)

    at, aw, ab = adj(r2_text, pt), adj(r2_writer, pw), adj(r2_both, pt + pw)
    return {"r2_text": float(at), "r2_writer": float(aw), "r2_both": float(ab),
            "partial_text": float(ab - aw), "partial_writer": float(ab - at), "r2_both_raw": float(r2_both)}


def icc1(values: list[float], groups: list) -> float:
    """One-way random-effects ICC(1) over groups with >= 2 members (unbalanced n0)."""
    by = collections.defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    gs = [np.asarray(v, float) for v in by.values() if len(v) >= 2]
    k = len(gs)
    if k < 2:
        return float("nan")
    N = sum(len(g) for g in gs)
    grand = np.concatenate(gs).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in gs)
    msb, msw = ssb / (k - 1), ssw / max(N - k, 1)
    n0 = (N - sum(len(g) ** 2 for g in gs) / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def residuals(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per line: mean over its slots of (error - leave-this-line-out slot error rate)."""
    cs, cn = (Y * M).sum(0), M.sum(0)
    loo = np.where(cn[None] > 1, (cs[None] - Y * M) / np.maximum(cn[None] - M, 1), np.nan)
    d = np.where(M, Y - loo, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(d, 1)


def sb(r: float) -> float:
    return 2 * r / (1 + r) if r > -1 else float("nan")


def split_half(Y: np.ndarray, M: np.ndarray, rng) -> tuple[float, float]:
    """(writer reliability, word reliability), Spearman-Brown corrected."""
    cs, cn = (Y * M).sum(0), M.sum(0)
    rate = np.where(cn > 0, cs / np.maximum(cn, 1), 0)
    R = np.where(M, Y - rate[None], 0.0)
    pos = np.cumsum(M, 1) * M  # 1-based position of each observed slot within its line
    odd, even = M & (pos % 2 == 1), M & (pos % 2 == 0) & (pos > 0)
    ok = (odd.sum(1) >= 2) & (even.sum(1) >= 2)
    wo = (R * odd).sum(1)[ok] / odd.sum(1)[ok]
    we = (R * even).sum(1)[ok] / even.sum(1)[ok]
    rw = sb(float(np.corrcoef(wo, we)[0, 1])) if ok.sum() > 3 else float("nan")
    half = rng.random(Y.shape[0]) < 0.5
    m1, m2 = M & half[:, None], M & ~half[:, None]
    okc = (m1.sum(0) >= 3) & (m2.sum(0) >= 3)
    r1 = (Y * m1).sum(0)[okc] / m1.sum(0)[okc]
    r2 = (Y * m2).sum(0)[okc] / m2.sum(0)[okc]
    rt = sb(float(np.corrcoef(r1, r2)[0, 1])) if okc.sum() > 3 and r1.std() > 0 and r2.std() > 0 else float("nan")
    return rw, rt


def ci(x) -> list[float]:
    x = np.asarray([v for v in x if v == v], float)
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])] if len(x) else [float("nan")] * 2


# ---- data -----------------------------------------------------------------------------------------------------------------------

def load_lines(ds: str) -> dict[str, dict[str, dict]]:
    cols = ["run_id", "family", "transform", "sample_key", "source", "hyp", "in_train", "e_crnn_native", "n_crnn_native",
            "e_no_diacritics", "n_no_diacritics"]
    t = pq.read_table(W / "results/store/lines.parquet", columns=cols, filters=[("dataset", "=", ds)]).to_pylist()
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in t:
        if r["in_train"] == "yes" or r["hyp"] is None:
            continue
        if r["family"] == "crnn":
            if not r["source"].startswith(PANEL_SOURCES) or r["transform"] not in ("none", "inkcrop"):
                continue
            key = r["run_id"] + ("" if r["transform"] == "none" else " [inkcrop]")
            e, n = r["e_crnn_native"], r["n_crnn_native"]
        elif r["family"] == "vlm" and r["transform"] == "none":
            key, e, n = r["run_id"], r["e_no_diacritics"], r["n_no_diacritics"]
        else:
            continue
        out[key][r["sample_key"]] = {"hyp": r["hyp"], "e": e, "n": n}
    if not any(k.startswith("crnn/") for k in out):
        out.update(load_panel_json(ds))
    return out


def load_panel_json(ds: str) -> dict[str, dict[str, dict]]:
    """Fallback before store_build has ingested the box results: crnn_eval_panel.py JSONs (run naming as store_build; edits are
    the stored CRNN-native ones). Only results/panel/khatt_full_out/ (none of these runs trained on KHATT test lines)."""
    out: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(str(W / "results/panel/khatt_full_out/*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        run = "crnn/" + d["run"].replace("aligner_run_", "").replace("aligner_run", "base") + ("" if d["ckpt"] == "model_final.pt" else "-best")
        for tag, suffix in ((ds, ""), (ds + "__inkcrop", " [inkcrop]")):
            m = d["per_line"].get(tag)
            if m:
                out[run + suffix] = {k: {"hyp": v[2], "e": v[0], "n": v[1]} for k, v in m.items()}
    return out


def load_samples(ds: str, pages: dict) -> list[dict]:
    t = pq.read_table(W / "results/store/samples.parquet", filters=[("dataset", "=", ds)]).to_pylist()
    v1_of = {}
    if ds == "kitab_khatt":
        fl = pq.read_table(W / "results/panel/khatt_full_flags.parquet", columns=["id", "kitab_image_id"]).to_pylist()
        v1_of = {r["kitab_image_id"]: r["id"] for r in fl if r["kitab_image_id"]}
    out = []
    for s in t:
        words = norm_text(s["gt_clean"]).split()
        slots = align(words)
        on = [x for x in slots if x is not None]
        if len(on) < MIN_SLOTS or len(on) < 0.8 * len(words):
            continue  # not fixed text
        v1 = v1_of.get(s["sample_key"], s["sample_key"])
        p = pages.get(v1, {})
        out.append({"key": s["sample_key"], "v1": v1, "gt": s["gt_clean"], "text": " ".join(words), "slots": slots,
                    "page": p.get("page_id"), **{f: p.get(f, np.nan) for f in FEATS}})
    return out


def matrix(samples: list[dict], reads: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    keep = [i for i, s in enumerate(samples) if s["key"] in reads]
    Y = np.zeros((len(keep), len(CANON)))
    M = np.zeros((len(keep), len(CANON)), bool)
    for r, i in enumerate(keep):
        s = samples[i]
        err = word_errors(s["gt"], reads[s["key"]]["hyp"])
        for e, sl in zip(err, s["slots"]):
            if sl is not None and not M[r, sl]:
                Y[r, sl], M[r, sl] = e, True
    return Y, M, keep


# ---- analysis per reader ---------------------------------------------------------------------------------------------------------

def analyse(samples, reads, reps: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Y, M, keep = matrix(samples, reads)
    S = [samples[i] for i in keep]
    n = len(S)
    cer = np.array([reads[s["key"]]["e"] / max(1, reads[s["key"]]["n"]) for s in S])
    wer = np.array([Y[i][M[i]].mean() if M[i].any() else np.nan for i in range(n)])
    fit = additive_fit(Y, M)
    rw, rt = split_half(Y, M, rng)
    pages = np.array([s["page"] or f"line{i}" for i, s in enumerate(S)])
    upages = np.unique(pages)
    page_ix = {p: np.nonzero(pages == p)[0] for p in upages}
    texts = [s["text"] for s in S]
    boots_line, boots_page = [], []
    for b in range(reps):
        ix = rng.integers(0, n, n)
        f = additive_fit(Y[ix], M[ix])
        boots_line.append((f["partial_text"], f["partial_writer"], *split_half(Y[ix], M[ix], rng)))
        pk = rng.choice(upages, len(upages))
        ixp = np.concatenate([page_ix[p] for p in pk])
        f = additive_fit(Y[ixp], M[ixp])
        boots_page.append((f["partial_text"], f["partial_writer"]))
    bl, bp = np.array(boots_line), np.array(boots_page)
    res = {"lines": n, "cer": round(float(cer.mean()), 4), "wer_slots": round(float(np.nanmean(wer)), 4),
           "variance": {**{k: round(v, 4) for k, v in fit.items()},
                        "partial_text_ci_line": ci(bl[:, 0]), "partial_writer_ci_line": ci(bl[:, 1]),
                        "partial_text_ci_page": ci(bp[:, 0]), "partial_writer_ci_page": ci(bp[:, 1]),
                        "writer_reliability_sb": round(rw, 4), "writer_reliability_ci": ci(bl[:, 2]),
                        "word_reliability_sb": round(rt, 4), "word_reliability_ci": ci(bl[:, 3])}}
    lo_w, hi_t = res["variance"]["partial_writer_ci_line"][0], res["variance"]["partial_text_ci_line"][1]
    lo_t, hi_w = res["variance"]["partial_text_ci_line"][0], res["variance"]["partial_writer_ci_line"][1]
    res["variance"]["D1"] = "writer > text" if lo_w > hi_t else ("text > writer" if lo_t > hi_w else "not separable")
    # same exact text
    cnt = collections.Counter(texts)
    grp = [t if cnt[t] >= 2 else None for t in texts]
    gi = [i for i, g in enumerate(grp) if g]
    ug = sorted({grp[i] for i in gi})
    icc = icc1([cer[i] for i in gi], [grp[i] for i in gi])
    gix = {g: [i for i in gi if grp[i] == g] for g in ug}
    icc_b = []
    for _ in range(reps):
        pick = rng.choice(len(ug), len(ug))
        vals, gg = [], []
        for k, q in enumerate(pick):
            for i in gix[ug[q]]:
                vals.append(cer[i]), gg.append(k)
        icc_b.append(icc1(vals, gg))
    sds = [float(np.std(cer[gix[g]], ddof=1)) for g in ug]
    res["same_text"] = {"groups": len(ug), "lines": len(gi), "largest": max((len(v) for v in gix.values()), default=0),
                        "icc_text": round(icc, 4), "icc_text_ci": ci(icc_b),
                        "median_within_sd_cer": round(float(np.median(sds)), 4) if sds else None,
                        "sd_cer_all": round(float(np.std(cer, ddof=1)), 4)}
    # pages: residual ICC with permutation null
    resid = residuals(Y, M)
    okr = ~np.isnan(resid)
    has_page = np.array([s["page"] is not None for s in S]) & okr
    pg = pages[has_page]
    vals = resid[has_page]
    multi = [p for p, c in collections.Counter(pg).items() if c >= 2]
    if len(multi) >= 3:
        icc_p = icc1(list(vals), list(pg))
        null = [icc1(list(vals), list(rng.permutation(pg))) for _ in range(reps)]
        upm = np.array(multi)
        pidx = {p: np.nonzero(pg == p)[0] for p in upm}
        bb = []
        for _ in range(reps):
            pk = rng.choice(upm, len(upm))
            vv, gg = [], []
            for k, p in enumerate(pk):
                for i in pidx[p]:
                    vv.append(vals[i]), gg.append(k)
            bb.append(icc1(vv, gg))
        res["pages"] = {"lines": int(has_page.sum()), "pages_ge2": len(multi), "icc_resid": round(icc_p, 4), "icc_resid_ci": ci(bb),
                        "perm_p": round(float((np.sum(np.array(null) >= icc_p) + 1) / (len(null) + 1)), 4),
                        "null_median": round(float(np.nanmedian(null)), 4)}
    else:
        res["pages"] = {"lines": int(has_page.sum()), "pages_ge2": len(multi), "note": "too few multi-line pages"}
    # style features vs residual (within text: the residual already removes slot difficulty)
    st = {}
    for f in FEATS:
        x = np.array([s[f] for s in S], float)
        ok = okr & ~np.isnan(x)
        if ok.sum() < 10:
            continue
        rho = float(spearmanr(x[ok], resid[ok]).statistic)
        idx = np.nonzero(ok)[0]
        bs = []
        for _ in range(reps):
            j = rng.choice(idx, len(idx))
            bs.append(spearmanr(x[j], resid[j]).statistic)
        bs = np.nan_to_num(np.array(bs))
        st[f] = {"n": int(ok.sum()), "rho": round(rho, 4), "ci": ci(bs), "p": boot_p(rho, bs)}
    res["style"] = st
    # slot difficulty
    rate = (Y * M).sum(0) / np.maximum(M.sum(0), 1)
    mean = float(Y[M].mean())
    sl = []
    boots = []
    for _ in range(reps):
        ix = rng.integers(0, n, n)
        boots.append((Y[ix] * M[ix]).sum(0) / np.maximum(M[ix].sum(0), 1) - (Y[ix][M[ix]].mean()))
    boots = np.array(boots)
    for j, w in enumerate(CANON):
        if M[:, j].sum() < 3:
            continue
        sl.append({"slot": j, "word": w, "n": int(M[:, j].sum()), "err": round(float(rate[j]), 4),
                   "delta": round(float(rate[j] - mean), 4), "p": boot_p(rate[j] - mean, boots[:, j])})
    rej, adj = bh([s["p"] for s in sl])
    for s, r_, a in zip(sl, rej, adj):
        s["q"], s["bh"] = round(float(a), 4), bool(r_)
    res["slots"] = sorted(sl, key=lambda s: -s["delta"])
    res["_per_line"] = {"keys": [s["key"] for s in S], "cer": cer.tolist(), "resid": resid.tolist()}
    return res


# ---- gallery --------------------------------------------------------------------------------------------------------------------

_teklia: dict[str, object] = {}


def teklia_img(v1: str) -> np.ndarray | None:
    _, split, idx = v1.split("/")
    if split not in _teklia:
        (f,) = glob.glob(str(HUB / SNAP.format(split=split)))
        _teklia[split] = pq.read_table(f, columns=["image"]).column("image")
    try:
        b = _teklia[split][int(idx)].as_py()["bytes"]
    except (IndexError, ValueError):
        return None
    return np.asarray(Image.open(io.BytesIO(b)).convert("L"))


def b64(g: np.ndarray, max_w: int = 1400) -> str:
    im = Image.fromarray(g)
    if im.width > max_w:  # downscale (never crop): full width stays visible
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("1").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def marked(gt: str, hyp: str) -> str:
    import difflib
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gt, hyp, autojunk=False).get_opcodes():
        seg = html.escape(hyp[j1:j2])
        out.append(seg if tag == "equal" else (f'<mark>{seg}</mark>' if seg else '<span class="miss">▾</span>'))
    return "".join(out)


def card(title: str, v1: str, gt: str | None, rows: list[tuple[str, str]]) -> str:
    g = teklia_img(v1) if v1.startswith("kitab_khatt_full/") else None
    img = f'<img src="{b64(g)}">' if g is not None else "<i>no image</i>"
    body = "".join(f'<div class="rd"><b>{html.escape(k)}</b> <span dir="rtl">{marked(gt, h) if gt else html.escape(h)}</span></div>'
                   for k, h in rows)
    gtl = f'<div class="gt" dir="rtl">{html.escape(gt)}</div>' if gt else ""
    return f'<div class="card"><div class="t">{html.escape(title)}</div>{img}{gtl}{body}</div>'


CSS = """body{font-family:system-ui,sans-serif;margin:16px;max-width:1500px;background:#fff;color:#111}
img{max-width:100%;border:1px solid #ccc;display:block}.card{margin:10px 0 18px}.t{font-size:12px;color:#555}
.gt{font-size:18px;color:#064}.rd{font-size:18px}mark{background:#fbb}.miss{color:#c00}h2{border-top:1px solid #ddd;padding-top:10px}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:2px 6px}
@media (prefers-color-scheme: dark){body{background:#111;color:#eee}img{filter:invert(1)}.gt{color:#7d7}mark{background:#733;color:#fff}}"""


def gallery(results: dict, samples_by_ds: dict, reads_by_ds: dict, pages_rows: list[dict], seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    parts = ["<h1>KHATT fixed text: same text, many writers (2.16) and reconstructed pages (2.17)</h1>",
             "<p>Images are shown at full width (downscaled, never cropped). Wrong characters in a reading are marked.</p>"]
    for ds, R in results.items():
        if PRIMARY not in R:
            continue
        S = {s["key"]: s for s in samples_by_ds[ds]}
        reads = reads_by_ds[ds]
        pl = R[PRIMARY]["_per_line"]
        cer = dict(zip(pl["keys"], pl["cer"]))
        resid = dict(zip(pl["keys"], pl["resid"]))
        by = collections.defaultdict(list)
        for k in pl["keys"]:
            by[S[k]["text"]].append(k)
        big = max(by.values(), key=len)
        big.sort(key=lambda k: cer[k])
        rnd = [k for k in rng.permutation(big[5:-5])[:5]] if len(big) > 10 else []
        others = [r for r in R if r != PRIMARY and not r.endswith("[inkcrop]")][:3]
        parts.append(f"<h2>{ds}: largest same-text group ({len(big)} writers), {PRIMARY}</h2><p dir='rtl'>{html.escape(S[big[0]]['text'])}</p>")
        for lab, ks in (("best", big[:5]), ("worst", big[::-1][:5]), ("random", rnd)):
            parts.append(f"<h3>{lab}</h3>")
            for k in ks:
                rows = [(PRIMARY, reads[PRIMARY][k]["hyp"])] + [(o, reads[o][k]["hyp"]) for o in others if k in reads[o]]
                parts.append(card(f"{k} (v1 {S[k]['v1']}, page {S[k]['page']}) CER {cer[k]:.3f}", S[k]["v1"], S[k]["gt"], rows))
        rk = sorted((k for k in resid if resid[k] == resid[k]), key=lambda k: resid[k])
        parts.append(f"<h2>{ds}: writer-driven lines ({PRIMARY}): most errors beyond what the words predict, and fewest</h2>")
        for lab, ks in (("worse than the words predict", rk[::-1][:5]), ("better than the words predict", rk[:5])):
            parts.append(f"<h3>{lab}</h3>")
            for k in ks:
                parts.append(card(f"{k} residual {resid[k]:+.3f} CER {cer[k]:.3f}", S[k]["v1"], S[k]["gt"], [(PRIMARY, reads[PRIMARY][k]["hyp"])]))
    # reconstructed pages (no readings needed)
    test = [r for r in pages_rows if r["split"] == "test"]
    pg = collections.defaultdict(list)
    for r in test:
        pg[r["page_id"]].append(r)
    full = [p for p, v in pg.items() if len(v) >= 4]
    parts.append(f"<h2>Reconstructed pages (2.17): 6 random test pages with at least 4 lines (of {len(full)})</h2>"
                 "<p>If the linking works, every line of a page is the same handwriting and the text runs on from line to line.</p>")
    for p in rng.permutation(full)[:6]:
        v = sorted(pg[p], key=lambda r: r["page_pos"])
        parts.append(f"<h3>{p} ({len(v)} lines, complete={v[0]['page_complete']})</h3>")
        for r in v:
            parts.append(card(f"{r['id']} slots {r['start']}–{r['end']} link cost {r['link_cost']}", r["id"], r["text_norm"], []))
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>KHATT fixed text</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"


def md_report(results: dict, pages_json: dict) -> str:
    L = ["# KHATT fixed text: same text, many writers (checklist 2.16) and writer / page IDs (2.17)", "",
         "Generated by `scripts/writer_khatt_fixed_text.py`; decision rules D1-D3 are in its docstring.", ""]
    L += ["## 2.17 pages", "", "| split | lines | links | pages | pages >= 2 lines | complete | held-out style ratio (linked / random) |",
          "|---|---|---|---|---|---|---|"]
    for s, d in pages_json["splits"].items():
        v = pages_json["validation"][s]
        rat = "; ".join(f"{x['ratio']} {x['ratio_ci95']}" for x in v.values())
        L.append(f"| {s} | {d['lines']} | {d['links']} | {d['pages']} | {d['pages_len_ge2']} | {d['pages_complete']} | {rat} |")
    for ds, R in results.items():
        L += ["", f"## 2.16 {ds}", "",
              "| reader | lines | CER | word err | R2 text | R2 writer | partial text [CI] | partial writer [CI] | D1 | writer rel. | word rel. | same-text ICC [CI] (groups) | page ICC resid (perm p) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for run, r in sorted(R.items(), key=lambda x: (x[0] != PRIMARY, x[0])):
            v, st, pg = r["variance"], r["same_text"], r["pages"]
            L.append(f"| {run} | {r['lines']} | {r['cer']} | {r['wer_slots']} | {v['r2_text']} | {v['r2_writer']} | "
                     f"{v['partial_text']} {v['partial_text_ci_line']} | {v['partial_writer']} {v['partial_writer_ci_line']} | {v['D1']} | "
                     f"{v['writer_reliability_sb']} | {v['word_reliability_sb']} | {st['icc_text']} {st['icc_text_ci']} ({st['groups']}) | "
                     f"{pg.get('icc_resid', '-')} ({pg.get('perm_p', '-')}) |")
        if PRIMARY in R:
            L += ["", f"Style features vs residual word error ({ds}; BH over features x readers): see JSON `style`, `style_bh`.",
                  "", f"Hardest words for {PRIMARY} ({ds}, BH q = 0.1):", "", "| word | n | err | delta | q |", "|---|---|---|---|---|"]
            for s in R[PRIMARY]["slots"][:12]:
                L.append(f"| {s['word']} | {s['n']} | {s['err']} | {s['delta']:+} | {s['q']}{' *' if s['bh'] else ''} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["kitab_khatt_full", "kitab_khatt"])
    ap.add_argument("--out-dir", type=Path, default=W / "results/writers")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--min-cover", type=float, default=0.95, help="a reader must have read this share of the fixed-text lines")
    a = ap.parse_args()
    pages_rows = pq.read_table(a.out_dir / "khatt_pages.parquet").to_pylist()
    pages = {r["id"]: r for r in pages_rows}
    pages_json = json.loads((a.out_dir / "khatt_pages.json").read_text(encoding="utf-8"))
    results, samples_by_ds, reads_by_ds = {}, {}, {}
    for ds in a.datasets:
        samples = load_samples(ds, pages)
        reads = load_lines(ds)
        keys = {s["key"] for s in samples}
        reads = {r: v for r, v in reads.items() if len(keys & v.keys()) >= a.min_cover * len(keys)}
        print(ds, "fixed-text lines", len(samples), "readers", len(reads), flush=True)
        if not reads:
            results[ds] = {}
            continue
        samples_by_ds[ds], reads_by_ds[ds] = samples, reads
        results[ds] = {run: analyse(samples, reads[run], a.reps) for run in sorted(reads)}
        # BH over features x readers (D2)
        cells = [(run, f, r["style"][f]["p"]) for run, r in results[ds].items() for f in r["style"]]
        rej, adj = bh([c[2] for c in cells])
        for (run, f, _), r_, q in zip(cells, rej, adj):
            results[ds][run]["style"][f]["q"], results[ds][run]["style"][f]["bh"] = round(float(q), 4), bool(r_)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "khatt_fixed_text.html").write_text(gallery(results, samples_by_ds, reads_by_ds, pages_rows), encoding="utf-8")
    slim = {ds: {run: {k: v for k, v in r.items() if k != "_per_line"} for run, r in R.items()} for ds, R in results.items()}
    status = {ds: ("readings present" if R else "no readings yet (run store_build after khatt_full_out arrives)") for ds, R in results.items()}
    (a.out_dir / "khatt_fixed_text.json").write_text(json.dumps({"item": "MASTER-CHECKLIST 2.16 (P1.8)", "status": status,
                                                                 "results": slim}, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out_dir / "khatt_fixed_text.md").write_text(md_report({d: R for d, R in slim.items() if R}, pages_json) +
                                                   "".join(f"\n{d}: {s}\n" for d, s in status.items()), encoding="utf-8")
    for ds, R in slim.items():
        if PRIMARY in R:
            r = R[PRIMARY]
            print(ds, PRIMARY, json.dumps({k: r[k] for k in ("lines", "cer", "variance", "same_text", "pages")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
