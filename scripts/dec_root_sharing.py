"""Checklist 2.15 / map perea-root: do real-word substitutions share the root more often than chance? CPU, decoding venv.

    CAMELTOOLS_DATA=/Volumes/PortableSSD/camel_tools_data /Volumes/PortableSSD/venvs/decoding/bin/python scripts/dec_root_sharing.py

Real-word substitution: 1:1 aligned word substitution (key view), both Arabic-letter words of >= 2 letters, different, reading
in the Common lexicon. Root share: CAMeL Tools MSA analyser (calima-msa-r13, no backoff) root sets intersect.
Chance (i): edit-distance-matched lexicon neighbours of the GT word (distance 1 enumerated, 2 sampled; up to 30 per word);
chance (ii): the model's readings shuffled across its GT words (2000 perms). Excess = observed - (i), cluster bootstrap.
Rule: results/decoding/decision_rules.json. Writes results/decoding/root_sharing.json and root_sharing_gallery.html.
"""
from __future__ import annotations

import html
import json
import random
import zlib
from collections import defaultdict

import numpy as np
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from rapidfuzz.distance import Levenshtein

import dec_common as C
N_ = C.N

REPS = 2000
K_NEIGH = 30
ALPHA = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
FAMILY = {m: f for (f, m) in C.MODELS}


class Roots:
    def __init__(self):
        self.a = Analyzer(MorphologyDB.builtin_db())
        self.cache = {}

    def __call__(self, w: str) -> frozenset:
        if w not in self.cache:
            rs = set()
            for x in self.a.analyze(w):
                r = x.get("root") or ""
                if r and "." in r and r not in ("NTWS", "PUNC", "DIGIT", "FOREIGN"):
                    rs.add(r)
            self.cache[w] = frozenset(rs)
        return self.cache[w]


def ortho(w: str) -> str:
    """Spelling-variant fold (hamza carriers, alef forms, yā'/alif maqsura, tā' marbūṭa): pairs equal under it are spelling, not
    word, substitutions and trivially share a root."""
    return N_.normalize_hamza(N_.normalize_taa_marbuta(N_.normalize_yaa(N_.normalize_alef(w))))


def edits1(w):
    s = set()
    for i in range(len(w) + 1):
        for c in ALPHA:
            s.add(w[:i] + c + w[i:])
            if i < len(w):
                s.add(w[:i] + c + w[i + 1:])
        if i < len(w):
            s.add(w[:i] + w[i + 1:])
    s.discard(w)
    return s


def neighbours(w, d, lex, rng, k=K_NEIGH):
    if d == 1:
        cand = sorted(x for x in edits1(w) if x in lex and len(x) >= 2)
    elif d == 2:
        e1 = sorted(edits1(w))
        cand = set()
        for _ in range(4000):
            x = rng.choice(e1)
            y = list(edits1(x)) if len(x) < 12 else []
            if not y:
                continue
            z = rng.choice(y)
            if z in lex and len(z) >= 2 and Levenshtein.distance(w, z) == 2:
                cand.add(z)
            if len(cand) >= 3 * k:
                break
        cand = sorted(cand)
    else:
        return []
    return rng.sample(cand, min(k, len(cand)))


def main():
    S = C.load_samples()
    L = C.load_readings(S)
    lex = C.lexicon()
    R = Roots()
    rng = random.Random(0)
    subs = []  # model, family, dataset, cluster, sample_key, gt, hyp, d, shared, dot_only
    for r in L.itertuples(index=False):
        for g, h in C.align_words(r.gt, r.hyp):
            if h is None or g == h or not (C.is_arabic_word(g) and C.is_arabic_word(h)) or len(g) < 2 or len(h) < 2:
                continue
            if h not in lex:
                continue
            rg, rh = R(g), R(h)
            if not rg or not rh:
                continue
            subs.append({"model": r.model, "family": r.family, "dataset": r.dataset, "cluster": r.cluster, "key": r.sample_key,
                         "gt": g, "hyp": h, "d": Levenshtein.distance(g, h), "shared": bool(rg & rh), "dot": C.rasm(g) == C.rasm(h), "ortho": ortho(g) == ortho(h)})
    print("substitutions", len(subs), flush=True)
    # neighbour baseline per (gt, d) for d in 1, 2
    nb = {}
    for s in subs:
        k = (s["gt"], s["d"])
        if s["d"] <= 2 and k not in nb:
            ns = [n for n in neighbours(s["gt"], s["d"], lex, random.Random(zlib.crc32(f"{k[0]}|{k[1]}".encode()))) if R(n) and ortho(n) != ortho(s["gt"])]  # spelling variants excluded from chance
            nb[k] = float(np.mean([bool(R(s["gt"]) & R(n)) for n in ns])) if ns else np.nan
    for s in subs:
        s["chance_nb"] = nb.get((s["gt"], s["d"]), np.nan)
    print("neighbour sets", len(nb), flush=True)

    res = {"n_substitutions_total": len(subs), "models": {}, "by_distance": {}, "family_pooled": {}}

    def excess(items, seed=0):
        it = [s for s in items if not np.isnan(s["chance_nb"])]
        if len(it) < 20:
            return None
        o = np.array([s["shared"] for s in it], float)
        c = np.array([s["chance_nb"] for s in it])
        cl = np.array([s["cluster"] for s in it])
        b = C.cluster_boot(cl, lambda ix: o[ix].mean() - c[ix].mean(), REPS, seed)
        bo = C.cluster_boot(cl, lambda ix: o[ix].mean(), REPS, seed)
        d = o.mean() - c.mean()
        return {"n": len(it), "observed": float(o.mean()), "observed_ci": C.ci(bo), "chance_neighbours": float(c.mean()),
                "excess": float(d), "excess_ci": C.ci(b), "p": C.boot_p(d, b),
                "ratio": float(o.mean() / c.mean()) if c.mean() > 0 else None}

    def shuffle_chance(items, seed=0):
        g = [s["gt"] for s in items]
        h = [s["hyp"] for s in items]
        r = np.random.default_rng(seed)
        vals = []
        for _ in range(200):
            p = r.permutation(len(h))
            vals.append(np.mean([bool(R(g[i]) & R(h[j])) for i, j in enumerate(p)]))
        return float(np.mean(vals))

    pv, keys = [], []
    for m in sorted({s["model"] for s in subs}):
        it = [s for s in subs if s["model"] == m]
        e = excess([s for s in it if not s["ortho"]])  # primary: spelling-variant pairs (hamza, alef, yā', tā' marbūṭa) excluded
        if e is None:
            continue
        e["family"] = FAMILY[m]
        e["n_all_distances"] = len(it)
        e["observed_all_distances"] = float(np.mean([s["shared"] for s in it]))
        e["chance_shuffle_all_distances"] = shuffle_chance(it)
        e["share_dot_only"] = float(np.mean([s["dot"] for s in it]))
        dot = excess([s for s in it if s["dot"]], seed=1)
        e["dot_only_subset"] = dot
        e["with_spelling_variants"] = excess(it, seed=2)
        e["share_spelling_variants"] = float(np.mean([s["ortho"] for s in it]))
        res["models"][m] = e
        pv.append(e["p"])
        keys.append(m)
    rej, adj = C.bh(pv)
    for m, a_, q in zip(keys, rej, adj):
        res["models"][m].update({"q": float(q), "bh_survives": bool(a_)})
    for d in (1, 2):
        res["by_distance"][d] = excess([s for s in subs if s["d"] == d], seed=d)
    fam = {"crnn": ["crnn"], "vlm": ["vlm-base", "vlm-ft", "ocr-engine"]}
    for k, fs in fam.items():
        res["family_pooled"][k] = excess([s for s in subs if s["family"] in fs], seed=7)
        res["family_pooled"][k + "_dot_only"] = excess([s for s in subs if s["family"] in fs and s["dot"]], seed=8)
        res["family_pooled"][k + "_no_spelling_variants"] = excess([s for s in subs if s["family"] in fs and not s["ortho"]], seed=9)
        res["family_pooled"][k + "_no_spelling_variants_no_dot"] = excess([s for s in subs if s["family"] in fs and not s["ortho"] and not s["dot"]], seed=10)
    keep_f = {FAMILY[m] if FAMILY[m] == "crnn" else "vlm" for m, e in res["models"].items() if e.get("bh_survives") and e["excess"] > 0}
    res["decision"] = {"families_with_bh_positive_excess": sorted(keep_f),
                       "models_positive": [m for m, e in res["models"].items() if e.get("bh_survives") and e["excess"] > 0],
                       "models_negative": [m for m, e in res["models"].items() if e.get("bh_survives") and e["excess"] < 0],
                       "outcome": "KEEP (both families)" if keep_f >= {"crnn", "vlm"} else
                       "KEEP as LM-prior effect (VLMs only)" if keep_f == {"vlm"} else "CRNN only" if keep_f == {"crnn"} else "DROP"}
    json.dump(res, open(C.OUT / "root_sharing.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res["decision"], ensure_ascii=False, indent=1))
    print(json.dumps(res["family_pooled"], ensure_ascii=False, indent=1))
    gallery(subs, L, R, res)


def gallery(subs, L, R, res):
    rng = random.Random(3)
    picks = []
    for m in ["combo", "qwen3-vl-4b-bf16", "suite-v1-qwen3vl4b-ft-bf16", "suite-v1-fanar2-oryx-ft-bf16"]:
        it = [s for s in subs if s["model"] == m]
        for lab, pool in (("root shared", [s for s in it if s["shared"]]), ("root not shared", [s for s in it if not s["shared"]])):
            picks += [(m, lab, s) for s in rng.sample(pool, min(5, len(pool)))]
    imgs = C.load_images({s["key"] for _, _, s in picks})
    gt_of = dict(zip(L.sample_key, L["gt"]))
    parts = [C.gallery_css(), "<title>Root sharing</title><h1>Root-sharing of real-word substitutions (checklist 2.15, perea-root)</h1>",
             f"<p class=mut>Decision: <code>{html.escape(json.dumps(res['decision'], ensure_ascii=False))}</code></p>",
             "<p class=mut>Random real-word substitutions per model, split by whether the CAMeL MSA root sets intersect. "
             "Roots are listed for both words; d = letter edit distance; ‘dot’ = same rasm.</p>"]
    for m, lab, s in picks:
        im = f"<img src='{C.img_b64(imgs[s['key']])}'>" if s["key"] in imgs else "<p class=mut>(no image in panel.pkl)</p>"
        gt = html.escape(C.key(gt_of[s["key"]])).replace(s["gt"], f"<span class=ok>{s['gt']}</span>", 1)
        parts.append(f"<div class=item><b>{m}</b> · {lab} · d={s['d']}{' · dot' if s['dot'] else ''} · chance(nb)={s['chance_nb']:.2f} · "
                     f"<span class=mut>{s['key']}</span>{im}<div class=ar>GT: {gt}</div>"
                     f"<div class=ar><span class=ok>{s['gt']}</span> {{{'، '.join(sorted(R(s['gt'])))}}} → "
                     f"<span class=hi>{s['hyp']}</span> {{{'، '.join(sorted(R(s['hyp'])))}}}</div></div>")
    (C.OUT / "root_sharing_gallery.html").write_text("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>" + "".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
