"""Checklist 2.14 / map E2: the tokenizer view (decoding-and-restoration.md §6). CPU, decoding venv.

    HF_HOME=/Volumes/PortableSSD/huggingface HF_HUB_OFFLINE=1 \
      /Volumes/PortableSSD/venvs/decoding/bin/python scripts/dec_tokenizer_view.py

(a) Do the VLM tokenizers give dot-pair words (ب/ت, ف/ق ...) unrelated token sequences? Dot swap vs same-position
    non-dot swap, token Dice.
(b) Do VLM dot errors (reading has the GT's rasm) drift toward spellings that take fewer tokens? CRNN control.
(c) Fertility (GT tokens per word) against errors: line partial Spearman with CER; word-level within-stratum slope.
Rules: results/decoding/decision_rules.json. Writes results/decoding/tokenizer_view.json and tokenizer_view_gallery.html.
"""
from __future__ import annotations

import html
import json
import random
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.stats import rankdata
from tokenizers import Tokenizer

import dec_common as C

REPS = 2000
SWAP_GROUPS = ["بتثني", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق"]  # ى/ي and ه/ة are spelling, not dot, variants
GROUP_OF = {c: g for g in SWAP_GROUPS for c in g}
BASE = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


class Tok:
    def __init__(self, repo):
        self.t = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.cache = {}

    def toks(self, w: str) -> tuple:
        """Tokens of a word as it appears mid-line (preceded by a space)."""
        if w not in self.cache:
            self.cache[w] = tuple(self.t.encode(" " + w, add_special_tokens=False).tokens)
        return self.cache[w]

    def n(self, w):
        return len(self.toks(w))


def dice(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return 2 * sum((ca & cb).values()) / (len(a) + len(b))


def edge_shared(a, b):
    """Share of tokens kept intact around the change: common prefix + common suffix tokens over the longer sequence."""
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return (p + s) / max(len(a), len(b))


def part_a(tok: Tok, words: list, lex: set, rng: random.Random):
    rows = []
    for w in words:
        pos = [i for i, c in enumerate(w) if c in GROUP_OF]
        if not pos:
            continue
        i = rng.choice(pos)
        c = w[i]
        d = rng.choice([x for x in GROUP_OF[c] if x != c])
        o = rng.choice([x for x in BASE if x not in GROUP_OF[c]])
        wd, wo = w[:i] + d + w[i + 1:], w[:i] + o + w[i + 1:]
        t, td, to = tok.toks(w), tok.toks(wd), tok.toks(wo)
        rows.append({"w": w, "wd": wd, "wo": wo, "dice_d": dice(t, td), "dice_o": dice(t, to), "edge_d": edge_shared(t, td),
                     "edge_o": edge_shared(t, to), "zero_d": dice(t, td) == 0, "zero_o": dice(t, to) == 0,
                     "dn_d": len(td) - len(t), "dn_o": len(to) - len(t), "single": len(t) == 1, "real_d": wd in lex})
    return rows


def summarise_a(rows, sel=lambda r: True):
    R = [r for r in rows if sel(r)]
    if not R:
        return None
    d = np.array([r["dice_d"] for r in R])
    o = np.array([r["dice_o"] for r in R])
    rng = np.random.default_rng(0)
    bt = [np.mean(d[ix] - o[ix]) for ix in (rng.integers(0, len(R), len(R)) for _ in range(REPS))]
    return {"n_words": len(R), "dice_dot": float(d.mean()), "dice_ctrl": float(o.mean()), "delta": float((d - o).mean()),
            "delta_ci": C.ci(bt), "zero_overlap_dot": float(np.mean([r["zero_d"] for r in R])),
            "zero_overlap_ctrl": float(np.mean([r["zero_o"] for r in R])),
            "edge_kept_dot": float(np.mean([r["edge_d"] for r in R])), "edge_kept_ctrl": float(np.mean([r["edge_o"] for r in R])),
            "token_count_changes_dot": float(np.mean([r["dn_d"] != 0 for r in R])),
            "token_count_changes_ctrl": float(np.mean([r["dn_o"] != 0 for r in R]))}


def word_table(L, toks):
    """One row per aligned GT word per reading: model, tokenizer, dataset, cluster, gt, hyp, err, sample."""
    out = []
    for r in L.itertuples(index=False):
        for g, h in C.align_words(r.gt, r.hyp):
            if not C.is_arabic_word(g):
                continue
            out.append((r.model, r.tok, r.dataset, r.cluster, r.sample_key, g, h, h != g))
    return out


def dot_errors(words, tk: Tok, models):
    """Pure dot errors per model: reading has the GT's rasm and differs."""
    res = {}
    for m in models:
        R = [w for w in words if w[0] == m and w[6] and w[6] != w[5] and C.is_arabic_word(w[6]) and C.rasm(w[6]) == C.rasm(w[5])]
        res[m] = [(w[3], tk.n(w[5]), tk.n(w[6]), w) for w in R]
    return res


def fewer_share(items):
    """items: (cluster, ntok_gt, ntok_hyp, row). Share of reading-has-fewer among unequal counts, with cluster CI."""
    it = [x for x in items if x[1] != x[2]]
    if len(it) < 5:
        return {"n_dot_errors": len(items), "n_unequal": len(it)}
    cl = np.array([x[0] for x in it])
    f = np.array([x[2] < x[1] for x in it], float)
    dn = np.array([x[2] - x[1] for x in items], float)
    b = C.cluster_boot(cl, lambda ix: f[ix].mean(), REPS)
    return {"n_dot_errors": len(items), "n_unequal": len(it), "share_fewer": float(f.mean()), "ci": C.ci(b),
            "p": C.boot_p(f.mean() - 0.5, b - 0.5), "mean_dtok_all": float(dn.mean()),
            "_f": f, "_cl": cl}


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else np.nan


def strat_slope(rows, tk: Tok):
    """Within (dataset, word length) strata: slope of error (0/1) on token count. rows: word-table tuples."""
    strata = defaultdict(list)
    for i, w in enumerate(rows):
        strata[(w[2], len(w[5]))].append(i)
    x = np.array([tk.n(w[5]) for w in rows], float)
    y = np.array([float(w[7]) for w in rows])
    xe, ye = np.zeros_like(x), np.zeros_like(y)
    for ix in strata.values():
        ix = np.array(ix)
        xe[ix] = x[ix] - x[ix].mean()
        ye[ix] = y[ix] - y[ix].mean()
    cl = np.array([w[3] for w in rows])

    def s(ix):
        d = (xe[ix] ** 2).sum()
        return (xe[ix] * ye[ix]).sum() / d if d > 0 else np.nan
    b = C.cluster_boot(cl, s, REPS)
    est = s(np.arange(len(rows)))
    return {"n_words": len(rows), "slope_err_per_token": float(est), "ci": C.ci(b), "p": C.boot_p(est, b),
            "err_rate": float(y.mean()), "mean_tokens": float(x.mean()), "_b": b}


def matched_slope_delta(rows_v, rows_c, tk: Tok, seed=0):
    """VLM slope minus CRNN-control slope on the SAME lines, joint cluster bootstrap (one resample drives both)."""
    def prep(rows):
        strata = defaultdict(list)
        for i, w in enumerate(rows):
            strata[(w[2], len(w[5]))].append(i)
        x = np.array([tk.n(w[5]) for w in rows], float)
        y = np.array([float(w[7]) for w in rows])
        for ix in strata.values():
            ix = np.array(ix)
            x[ix] -= x[ix].mean()
            y[ix] -= y[ix].mean()
        by = defaultdict(list)
        for i, w in enumerate(rows):
            by[w[3]].append(i)
        return x, y, by
    xv, yv, bv = prep(rows_v)
    xc, yc, bc = prep(rows_c)
    cls = sorted(set(bv) & set(bc))

    def sl(x, y, by, pick):
        ix = np.concatenate([by[cls[j]] for j in pick])
        d = (x[ix] ** 2).sum()
        return (x[ix] * y[ix]).sum() / d if d > 0 else np.nan
    allp = np.arange(len(cls))
    est = sl(xv, yv, bv, allp) - sl(xc, yc, bc, allp)
    rng = np.random.default_rng(seed)
    b = np.empty(REPS)
    for r in range(REPS):
        pick = rng.integers(0, len(cls), len(cls))
        b[r] = sl(xv, yv, bv, pick) - sl(xc, yc, bc, pick)
    return {"vlm_slope": float(sl(xv, yv, bv, allp)), "crnn_slope_same_lines": float(sl(xc, yc, bc, allp)), "delta": float(est),
            "ci": C.ci(b), "p": C.boot_p(est, b), "n_clusters": len(cls)}


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    S = C.load_samples()
    L = C.load_readings(S)
    lex = C.lexicon()
    toks = {k: Tok(r) for k, r in C.TOKENIZERS.items()}
    res = {"tokenizers": C.TOKENIZERS, "note_tokenizers": "Qwen2-VL, Qwen2.5-VL, Qwen3-VL, AIN, Fanar-2-Oryx, Baseer and Qari share one BPE "
           "(identical ids checked on Arabic text). PaliGemma2's tokenizer is gated; the Gemma-2 SentencePiece vocabulary (ungated mirror "
           "unsloth/gemma-2-2b-it) is the same text vocabulary without the <loc>/<seg> tokens.",
           "readings": L.groupby("model").size().to_dict()}

    # (a) dot-pair token sequences over panel GT word types
    words = sorted({w for g in S.gt_clean for w in C.key(g).split() if C.is_arabic_word(w) and len(w) >= 2})
    res["a_dot_pairs"] = {}
    rows_a = {}
    for k, tk in toks.items():
        rows_a[k] = part_a(tk, words, lex, random.Random(0))
        res["a_dot_pairs"][k] = {"all": summarise_a(rows_a[k]), "single_token_words": summarise_a(rows_a[k], lambda r: r["single"]),
                                 "dot_variant_is_real_word": summarise_a(rows_a[k], lambda r: r["real_d"])}
        print("a", k, res["a_dot_pairs"][k]["all"], flush=True)

    # (b) dot errors and token counts
    W = word_table(L, toks)
    vlms = sorted({m for (f, m), t in C.MODELS.items() if t})
    crnns = sorted({m for (f, m), t in C.MODELS.items() if t is None})
    tok_of = {m: t for (f, m), t in C.MODELS.items()}
    res["b_dot_errors"] = {}
    pv, keys = [], []
    for m in vlms:
        tk = toks[tok_of[m]]
        de = dot_errors(W, tk, [m])[m]
        r = fewer_share(de)
        r["hyp_in_lexicon"] = float(np.mean([x[3][6] in lex for x in de])) if de else None
        r["gt_in_lexicon"] = float(np.mean([x[3][5] in lex for x in de])) if de else None
        res["b_dot_errors"][m] = r
        if "p" in r:
            pv.append(r["p"])
            keys.append(m)
    rej, adj = C.bh(pv)
    for m, a_, q in zip(keys, rej, adj):
        res["b_dot_errors"][m].update({"q": float(q), "bh_survives": bool(a_), "direction": "fewer" if res["b_dot_errors"][m]["share_fewer"] > 0.5 else "more"})
    # pooled VLM vs CRNN control, per tokenizer
    res["b_pooled_vs_crnn"] = {}
    for tkey, tk in toks.items():
        vm = [m for m in vlms if tok_of[m] == tkey]
        pool = [x for m in vm for x in dot_errors(W, tk, [m])[m]]
        crn = [x for m in crnns for x in dot_errors(W, tk, [m])[m]]
        pv_, cr_ = fewer_share(pool), fewer_share(crn)
        if "share_fewer" in pv_ and "share_fewer" in cr_:
            fa, ca, fb, cb = pv_["_f"], pv_["_cl"], cr_["_f"], cr_["_cl"]
            b1 = C.cluster_boot(ca, lambda ix: fa[ix].mean(), REPS, seed=1)
            b2 = C.cluster_boot(cb, lambda ix: fb[ix].mean(), REPS, seed=2)
            d = pv_["share_fewer"] - cr_["share_fewer"]
            res["b_pooled_vs_crnn"][tkey] = {"vlm_models": vm, "vlm": {k: v for k, v in pv_.items() if not k.startswith("_")},
                                             "crnn": {k: v for k, v in cr_.items() if not k.startswith("_")},
                                             "delta": d, "delta_ci": C.ci(b1 - b2), "p": C.boot_p(d, b1 - b2)}
    for m in res["b_dot_errors"]:
        for k in ("_f", "_cl"):
            res["b_dot_errors"][m].pop(k, None)

    # (c) fertility vs errors
    res["c_line"] = []
    for (m, ds), g in L.groupby(["model", "dataset"]):
        if len(g) < 50:
            continue
        tk = toks[tok_of[m] or "qwen"]
        gw = [[w for w in C.key(t).split() if w] for t in g["gt"]]
        ok = np.array([len(x) > 0 for x in gw])
        fert = np.array([sum(tk.n(w) for w in x) / len(x) if x else np.nan for x in gw])
        mwl = np.array([np.mean([len(w) for w in x]) if x else np.nan for x in gw])
        cer = (g.e_no_diacritics / g.n_no_diacritics.clip(lower=1)).values
        fert, mwl, cer, cl = fert[ok], mwl[ok], cer[ok], g.cluster.values[ok]
        est = partial_spearman(fert, cer, mwl)
        b = C.cluster_boot(cl, lambda ix: partial_spearman(fert[ix], cer[ix], mwl[ix]), 1000)
        res["c_line"].append({"model": m, "dataset": ds, "tokenizer": tok_of[m] or "qwen (control)", "n": int(ok.sum()),
                              "partial_rho": est, "ci": C.ci(b), "p": C.boot_p(est, b), "mean_fertility": float(np.nanmean(fert))})
    rej, adj = C.bh([r["p"] for r in res["c_line"]])
    for r, a_, q in zip(res["c_line"], rej, adj):
        r.update({"q": float(q), "bh_survives": bool(a_)})
    res["c_word"] = {}
    pv, keys = [], []
    for m in vlms + crnns:
        tk = toks[tok_of[m] or "qwen"]
        rows = [w for w in W if w[0] == m]
        r = strat_slope(rows, tk)
        res["c_word"][m] = r
        if m in vlms:
            pv.append(r["p"])
            keys.append(m)
    rej, adj = C.bh(pv)
    for m, a_, q in zip(keys, rej, adj):
        res["c_word"][m].update({"q": float(q), "bh_survives": bool(a_)})
    # pooled VLM (qwen tokenizer) vs CRNN control slope
    tkq = toks["qwen"]
    qv = [w for w in W if tok_of[w[0]] == "qwen"]
    cr = [w for w in W if tok_of[w[0]] is None]
    sv, sc = strat_slope(qv, tkq), strat_slope(cr, tkq)
    d = sv["slope_err_per_token"] - sc["slope_err_per_token"]
    res["c_word_pooled_vs_crnn"] = {"vlm_qwen": {k: v for k, v in sv.items() if k != "_b"}, "crnn": {k: v for k, v in sc.items() if k != "_b"},
                                    "delta": d, "delta_ci": C.ci(sv["_b"] - sc["_b"]), "p": C.boot_p(d, sv["_b"] - sc["_b"])}
    for m in res["c_word"]:
        res["c_word"][m].pop("_b", None)
    # matched: each VLM against the CRNN control (combo) on the same lines, with the VLM's own tokenizer
    res["c_word_matched_crnn"] = {}
    for m in vlms:
        rv = [w for w in W if w[0] == m]
        keys_ = {w[4] for w in rv}
        rc = [w for w in W if w[0] == "combo" and w[4] in keys_]
        if len(rc) > 200:
            res["c_word_matched_crnn"][m] = matched_slope_delta(rv, rc, toks[tok_of[m]])
    ms = list(res["c_word_matched_crnn"])
    rej, adj = C.bh([res["c_word_matched_crnn"][m]["p"] for m in ms])
    for m, a_, q in zip(ms, rej, adj):
        res["c_word_matched_crnn"][m].update({"q": float(q), "bh_survives": bool(a_)})

    # decision outcomes
    a_out = {}
    for k, v in res["a_dot_pairs"].items():
        a = v["all"]
        a_out[k] = ("unrelated (dot pairs no closer than arbitrary swaps)" if a["delta_ci"][1] < 0.05 else
                    "related (dot pairs closer)" if a["delta_ci"][0] > 0.05 else "undecided")
    b_keep = [m for m, r in res["b_dot_errors"].items() if r.get("bh_survives") and r["share_fewer"] > 0.5]
    b_ctrl = {k: v["delta_ci"][0] > 0 for k, v in res["b_pooled_vs_crnn"].items()}
    c_keep = [m for m in vlms if res["c_word"][m].get("bh_survives") and res["c_word"][m]["slope_err_per_token"] > 0]
    c_ctrl = res["c_word_pooled_vs_crnn"]["delta_ci"][0] > 0
    res["decision"] = {"E2-a": a_out,
                       "E2-b": {"vlms_fewer_bh": b_keep, "vlm_minus_crnn_ci_above_0": b_ctrl,
                                "outcome": "KEEP" if len(b_keep) >= 2 and any(b_ctrl.values()) else "DROP"},
                       "E2-c": {"vlms_positive_slope_bh": c_keep, "vlm_slope_above_crnn": bool(c_ctrl),
                                "outcome": "KEEP" if len(c_keep) >= 2 and c_ctrl else "DROP (fertility tracks errors equally in the tokenizer-free CRNN: a rarity/length proxy)" if len(c_keep) >= 2 else "DROP"}}
    json.dump(res, open(C.OUT / "tokenizer_view.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res["decision"], ensure_ascii=False, indent=1))
    gallery(L, W, toks, tok_of, rows_a, res)


def tokstr(t):
    return " | ".join(x.replace("Ġ", "").replace("▁", "") for x in t)


def gallery(L, W, toks, tok_of, rows_a, res):
    rng = random.Random(1)
    parts = [C.gallery_css(), "<title>Tokenizer view</title><h1>Tokenizer view (checklist 2.14, E2)</h1>",
             f"<p class=mut>Decision: <code>{html.escape(json.dumps(res['decision'], ensure_ascii=False))}</code></p>"]
    # (a) random word pairs
    parts.append("<h2>(a) Dot swap vs control swap: token sequences (random words)</h2><table><tr><th>tokenizer</th><th>word</th><th>tokens</th>"
                 "<th>dot swap</th><th>tokens</th><th>control swap</th><th>tokens</th></tr>")
    for k, rows in rows_a.items():
        for r in rng.sample(rows, 10):
            tk = toks[k]
            parts.append(f"<tr><td>{k}</td><td class=ar>{r['w']}</td><td class=ar>{tokstr(tk.toks(r['w']))}</td><td class=ar>{r['wd']}</td>"
                         f"<td class=ar>{tokstr(tk.toks(r['wd']))}</td><td class=ar>{r['wo']}</td><td class=ar>{tokstr(tk.toks(r['wo']))}</td></tr>")
    parts.append("</table>")
    # (b) VLM dot errors on line images
    picks = []
    for m in ["qwen3-vl-4b-bf16", "suite-v1-qwen3vl4b-ft-bf16", "paligemma2-3b-bf16", "combo"]:
        tk = toks[tok_of[m] or "qwen"]
        de = [w for w in W if w[0] == m and w[6] and w[6] != w[5] and C.is_arabic_word(w[6]) and C.rasm(w[6]) == C.rasm(w[5])]
        fewer = [w for w in de if tk.n(w[6]) < tk.n(w[5])]
        more = [w for w in de if tk.n(w[6]) > tk.n(w[5])]
        for lab, pool in (("reading has fewer tokens", fewer), ("reading has more tokens", more), ("random", de)):
            picks += [(m, lab, w) for w in rng.sample(pool, min(4, len(pool)))]
    fm = "qwen3-vl-4b-bf16"
    g = L[L.model == fm].copy()
    tk = toks["qwen"]
    g["fert"] = [np.mean([tk.n(w) for w in C.key(t).split()] or [np.nan]) for t in g["gt"]]
    g["cer"] = g.e_no_diacritics / g.n_no_diacritics.clip(lower=1)
    g = g.dropna(subset=["fert"]).sort_values("fert")
    sel = [("lowest fertility", g.head(4)), ("highest fertility", g.tail(4)), ("random", g.sample(4, random_state=0))]
    imgs = C.load_images({w[4] for _, _, w in picks} | {k for _, d in sel for k in d.sample_key})
    gt_of = dict(zip(L.sample_key, L["gt"]))
    parts.append("<h2>(b) Dot errors (reading has the GT's rasm), with token splits</h2><p class=mut>Per model: lines where the reading's token "
                 "count fell, rose, and random. CRNN rows are the tokenizer-free control (Qwen tokenizer applied).</p>")
    for m, lab, w in picks:
        tk = toks[tok_of[m] or "qwen"]
        im = f"<img src='{C.img_b64(imgs[w[4]])}'>" if w[4] in imgs else "<p class=mut>(no image in panel.pkl)</p>"
        gt = html.escape(C.key(gt_of[w[4]])).replace(w[5], f"<span class=ok>{w[5]}</span>", 1)
        parts.append(f"<div class=item><b>{m}</b> · {lab} · <span class=mut>{w[4]}</span>{im}<div class=ar>GT: {gt}</div>"
                     f"<div class=ar>GT word <span class=ok>{w[5]}</span> → [{tokstr(tk.toks(w[5]))}] ({tk.n(w[5])}) &nbsp; "
                     f"read <span class=hi>{w[6]}</span> → [{tokstr(tk.toks(w[6]))}] ({tk.n(w[6])})</div></div>")
    # (c) highest / lowest fertility lines for one model
    parts.append(f"<h2>(c) Fertility extremes for {fm} (Qwen tokenizer)</h2>")
    for lab, d in sel:
        for r in d.itertuples():
            im = f"<img src='{C.img_b64(imgs[r.sample_key])}'>" if r.sample_key in imgs else ""
            parts.append(f"<div class=item><b>{lab}</b> · fertility {r.fert:.2f} tokens/word · CER {r.cer:.3f} · <span class=mut>{r.sample_key}</span>{im}"
                         f"<div class=ar>GT: {html.escape(r.gt)}</div><div class=ar>read: {html.escape(r.hyp or '')}</div></div>")
    (C.OUT / "tokenizer_view_gallery.html").write_text("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>" + "".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
