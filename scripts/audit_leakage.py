"""Checklist 2.9 / R4.2: text and image leakage and duplicates, within and across the panel sets and their training pools.

    .geomvenv/bin/python scripts/audit_leakage.py [--out results/audit]

Decision rule (written before the run):
  * A test line is LEAKED when its normalised text (audit_common.norm_text) equals a training line's, or is a near
    duplicate (MinHash-estimated, then verified, character-5-gram Jaccard >= 0.8), or its image is a near duplicate of a
    training image (512-bit dHash within the threshold calibrated on the 89 known KITAB-KHATT / KHATT v1.0 same-image
    pairs, positive control) AND the texts agree (normalised Levenshtein ratio >= 0.8).
  * The leak matters for a set when in-domain systems gain more on leaked lines than out-of-domain systems do:
    difference-in-differences of line CER (no_diacritics), paired line bootstrap, 95 % CI excluding 0, BH-FDR q = 0.1
    across sets. Then in-domain headline numbers are reported on clean lines only; otherwise the flag is kept and reported.

Text pools: the test sets in results/store/samples.parquet (incl. kitab_khatt_full test) and the raw HF training labels
(Muharaf train + validation, NAKBA train, KHATT v1.0 train + validation, RASAM-1 train). Image pools: the 64 px panel images
and the 64 px training images (aligner pickles for Muharaf and NAKBA, khatt_full_train.pkl, RASAM-1 train decoded from HF).
Outputs: OUT/leakage.json (tables), OUT/leakage_flags.parquet (one row per test line), OUT/leakage_gallery.html.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import pickle
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import HTML_HEAD, OUT, W, load_samples, load_systems, norm_text, panel_images, png_b64  # noqa: E402
from eval_stats import bh, boot_p  # noqa: E402

HF = "/Volumes/PortableSSD/huggingface/hub"
TRAIN_TEXT = {
    "muharaf_train": [f"{HF}/datasets--aamijar--muharaf-public/snapshots/*/data/train-*.parquet",
                      f"{HF}/datasets--aamijar--muharaf-public/snapshots/*/data/validation-*.parquet"],
    "nakba_train": [f"{HF}/datasets--U4RASD--omar-al-saleh-manuscripts-segments/snapshots/*/data/train-*.parquet"],
    "khatt_train": [f"{HF}/datasets--Teklia--KHATT_v1.0/snapshots/*/data/train-*.parquet",
                    f"{HF}/datasets--Teklia--KHATT_v1.0/snapshots/*/data/validation-*.parquet"],
    "rasam1_train": [f"{HF}/datasets--calfa-ai--RASAM-1/snapshots/*/data/train-*.parquet"],
}
TEXT_COL = {"rasam1_train": "transcription"}
# which training pool is "in-domain" for which test set, and which systems trained on it
HOME_POOL = {"muharaf_lines": "muharaf_train", "nakba_lines": "nakba_train", "nakba_blind": "nakba_train",
             "kitab_khatt": "khatt_train", "kitab_khatt_full": "khatt_train", "rasam1": "rasam1_train", "rasam2": "rasam1_train"}
SHINGLE, NPERM, BANDS = 5, 64, 16
JACC_NEAR, LEV_AGREE = 0.8, 0.8


def in_domain(run: str, ds: str) -> bool | None:
    """True: trained on ds's training pool; False: did not; None: ambiguous (left out of the DiD)."""
    r = run.lower()
    if ds.startswith("nakba"):
        return "nakba" in r
    if ds in ("muharaf_lines", "rasam1", "rasam2"):
        if r.startswith(("crnn/", "crnn+llm/")) or "suite-v1" in r or "exp3090" in r:
            return True
        return False
    if ds.startswith("kitab_khatt"):
        if r.startswith("crnn/khatt"):
            return True
        return None if r.startswith(("crnn/", "crnn+llm/")) else False
    return None


def files(globs):
    return [f for g in globs for f in sorted(glob.glob(g)) if "/._" not in f]


def train_texts() -> dict[str, list[tuple[str, str]]]:
    out = {}
    for pool, gl in TRAIN_TEXT.items():
        rows = []
        for f in files(gl):
            col = TEXT_COL.get(pool, "text")
            for i, t in enumerate(pq.read_table(f, columns=[col]).column(col).to_pylist()):
                rows.append((f"{pool}/{Path(f).stem.split('-')[0]}/{len(rows):06d}", t or ""))
        out[pool] = rows
    return out


# ---------------------------------------------------------------------------- MinHash
_P = (1 << 61) - 1
_rng = np.random.default_rng(7)
_A = _rng.integers(1, _P, NPERM, dtype=np.uint64)
_B = _rng.integers(0, _P, NPERM, dtype=np.uint64)


def shingles(s: str) -> set[int]:
    s = s.replace(" ", "_")
    if len(s) < SHINGLE:
        return {zlib.crc32(s.encode())} if s else set()
    return {zlib.crc32(s[i:i + SHINGLE].encode()) for i in range(len(s) - SHINGLE + 1)}


def minhash(sh: set[int]) -> np.ndarray:
    if not sh:
        return np.full(NPERM, np.iinfo(np.uint64).max, np.uint64)
    x = np.fromiter(sh, np.uint64)
    # (a*x + b) mod p with python ints is slow; use float-free uint64 wraparound hashing (universal enough for LSH)
    h = (x[None, :] * _A[:, None] + _B[:, None]) >> np.uint64(3)
    return h.min(1)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------------------- dHash
def dhash(img: np.ndarray, rows: int = 8, cols: int = 64) -> np.ndarray:
    im = Image.fromarray(np.asarray(img, np.uint8)).convert("L").resize((cols + 1, rows), Image.BILINEAR)
    a = np.asarray(im, np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())  # 512 bits -> 64 bytes


def to64(b: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(b)).convert("L")
    return np.asarray(im.resize((max(1, round(im.width * 64 / im.height)), 64)))


def hamming_min(Q: np.ndarray, R: np.ndarray, chunk: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """For each row of Q (n x 64 uint8) the min Hamming distance to R and its index."""
    Qw, Rw = Q.view(np.uint64), R.view(np.uint64)
    best, arg = np.full(len(Q), 10**6), np.zeros(len(Q), int)
    for i in range(0, len(Q), chunk):
        d = np.bitwise_count(Qw[i:i + chunk, None, :] ^ Rw[None, :, :]).sum(-1)
        arg[i:i + chunk] = d.argmin(1)
        best[i:i + chunk] = d[np.arange(len(d)), arg[i:i + chunk]]
    return best, arg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    test = {k: s for k, s in samples.items()}
    tnorm = {k: norm_text(s["gt_raw"]) for k, s in test.items()}
    pools = train_texts()
    print({p: len(v) for p, v in pools.items()}, flush=True)
    pnorm = {p: [(i, norm_text(t), t) for i, t in v] for p, v in pools.items()}

    # ---- exact text matches
    exact_index = defaultdict(lambda: defaultdict(list))  # norm -> pool -> ids
    for p, v in pnorm.items():
        for i, n, _ in v:
            if n:
                exact_index[n][p].append(i)
    test_by_norm = defaultdict(list)
    for k, n in tnorm.items():
        test_by_norm[n].append(k)

    # ---- MinHash LSH over everything (test + train)
    items = [("test", k, tnorm[k]) for k in test] + [(p, i, n) for p, v in pnorm.items() for i, n, _ in v]
    sh = [shingles(n) for _, _, n in items]
    sig = np.stack([minhash(s) for s in sh])
    rpb = NPERM // BANDS
    buckets = defaultdict(list)
    for j in range(len(items)):
        if len(items[j][2]) < 12:  # very short lines: exact match only (near-dup of a 2-word line is meaningless)
            continue
        for b in range(BANDS):
            buckets[(b, sig[j, b * rpb:(b + 1) * rpb].tobytes())].append(j)
    near = defaultdict(dict)  # test idx -> {other idx: jaccard}
    for js in buckets.values():
        if len(js) < 2 or len(js) > 400:
            continue
        tj = [j for j in js if items[j][0] == "test"]
        for x in tj:
            for y in js:
                if y == x or y in near[x]:
                    continue
                jac = jaccard(sh[x], sh[y])
                if jac >= JACC_NEAR:
                    near[x][y] = jac
    print("lsh done", flush=True)

    # ---- image hashes
    imgs = panel_images(set(test))
    tkeys = [k for k in test if k in imgs]
    TH = np.stack([dhash(imgs[k]) for k in tkeys])
    del imgs
    trimg = {}
    for p, pk in (("muharaf_train", "aligner_data_standard.pkl"), ("nakba_train", "aligner_data_nakba.pkl"), ("khatt_train", "khatt_full_train.pkl")):
        rows = pickle.load(open(W / "results" / "panel" / pk, "rb"))["rows"]
        trimg[p] = ([r["id"] for r in rows], [r["label"] for r in rows], np.stack([dhash(r["img"]) for r in rows]))
        del rows
    ids, labs, hs = [], [], []
    for f in files(TRAIN_TEXT["rasam1_train"]):
        t = pq.read_table(f, columns=["image", "transcription"]).to_pylist()
        for r in t:
            ids.append(f"rasam1_train/img/{len(ids):06d}")
            labs.append(r["transcription"] or "")
            hs.append(dhash(to64(r["image"]["bytes"])))
    trimg["rasam1_train"] = (ids, labs, np.stack(hs))
    print("hashes done", {p: len(v[0]) for p, v in trimg.items()}, flush=True)

    # calibration: known same-image pairs (KHATT v1.0 line == KITAB-KHATT line), and random different pairs
    flags = {r["id"]: r for r in pq.read_table(W / "results/panel/khatt_full_flags.parquet").to_pylist()}
    tix = {k: i for i, k in enumerate(tkeys)}
    kh_ids, _, kh_h = trimg["khatt_train"]
    kh_ix = {k: i for i, k in enumerate(kh_ids)}
    pos = []
    for fid, r in flags.items():
        kid = r["kitab_image_id"]
        if kid and kid in tix:
            other = TH[tix[fid]] if fid in tix else (kh_h[kh_ix[fid]] if fid in kh_ix else None)
            if other is not None:
                pos.append(int(np.bitwise_count(TH[tix[kid]].view(np.uint64) ^ other.view(np.uint64)).sum()))
    rng = np.random.default_rng(0)
    negi = rng.integers(0, len(TH), (4000, 2))
    neg = [int(np.bitwise_count(TH[i].view(np.uint64) ^ TH[j].view(np.uint64)).sum()) for i, j in negi if i != j]
    thr = int(min(np.percentile(neg, 0.5), max(pos) + 1)) if pos else int(np.percentile(neg, 0.5))
    calib = {"positive_pairs": len(pos), "positive_hamming": [int(np.min(pos)), int(np.median(pos)), int(np.max(pos))] if pos else None,
             "negative_hamming_p0.5_p50": [float(np.percentile(neg, 0.5)), float(np.median(neg))], "threshold": thr,
             "positives_recalled": float(np.mean(np.array(pos) <= thr)) if pos else None,
             "random_pairs_below_threshold": float(np.mean(np.array(neg) <= thr))}
    print(calib, flush=True)

    # image matches: test vs each train pool, and test vs test (other line)
    img_hit = {}  # test key -> (pool, id, dist, label)
    for p, (ids, labs, H) in trimg.items():
        d, ix = hamming_min(TH, H)
        for qi, k in enumerate(tkeys):
            if d[qi] <= thr and (k not in img_hit or d[qi] < img_hit[k][2]):
                img_hit[k] = (p, ids[ix[qi]], int(d[qi]), labs[ix[qi]])
    Tw = TH.view(np.uint64)
    tt_img = {}
    for i in range(0, len(TH), 64):
        d = np.bitwise_count(Tw[i:i + 64, None, :] ^ Tw[None, :, :]).sum(-1)
        for r in range(d.shape[0]):
            d[r, i + r] = 10**6
            j = int(d[r].argmin())
            if d[r, j] <= thr:
                tt_img[tkeys[i + r]] = (tkeys[j], int(d[r, j]))

    # ---- per test line flags
    rows = []
    for k, s in test.items():
        ds, n = s["dataset"], tnorm[k]
        home = HOME_POOL.get(ds)
        ex_pools = {p: ids for p, ids in exact_index.get(n, {}).items()} if n else {}
        rows.append({"sample_key": k, "dataset": ds, "norm_len": len(n),
                     "exact_home": bool(home and home in ex_pools), "exact_pools": sorted(ex_pools),
                     "exact_home_example": (ex_pools.get(home) or [""])[0] if home else "",
                     "dup_in_test_same_set": sum(test[o]["dataset"] == ds for o in test_by_norm[n] if o != k) if n else 0,
                     "dup_in_other_test_sets": sorted({test[o]["dataset"] for o in test_by_norm[n] if o != k and test[o]["dataset"] != ds}) if n else []})
    key_pos = {it[1]: j for j, it in enumerate(items) if it[0] == "test"}
    for r in rows:
        k, ds = r["sample_key"], r["dataset"]
        home = HOME_POOL.get(ds)
        nn = near.get(key_pos[k], {})
        hits = [(items[y][0], items[y][1], jac) for y, jac in nn.items()]
        home_hits = [h for h in hits if h[0] == home]
        r["near_home"] = bool(home_hits)
        r["near_home_jaccard"] = max([h[2] for h in home_hits], default=0.0)
        r["near_home_example"] = max(home_hits, key=lambda h: h[2])[1] if home_hits else ""
        r["near_other_pools"] = sorted({h[0] for h in hits if h[0] not in ("test", home)})
        r["near_in_test"] = sorted({test[h[1]]["dataset"] for h in hits if h[0] == "test"})
        ih = img_hit.get(k)
        agree = bool(ih) and Levenshtein.normalized_similarity(norm_text(ih[3]), tnorm[k]) >= LEV_AGREE
        r["img_pool"], r["img_id"], r["img_dist"], r["img_text_agree"] = (ih[0], ih[1], ih[2], agree) if ih else ("", "", -1, False)
        r["img_home"] = bool(ih) and ih[0] == home and agree
        tt = tt_img.get(k)
        r["img_dup_test"], r["img_dup_test_dist"] = (tt[0], tt[1]) if tt else ("", -1)
        r["leaked"] = r["exact_home"] or r["near_home"] or r["img_home"]
    by_k = {r["sample_key"]: r for r in rows}

    # ---- leakage table per set
    table = {}
    for ds in sorted({r["dataset"] for r in rows}):
        R = [r for r in rows if r["dataset"] == ds]
        n = len(R)
        table[ds] = {"lines": n, "home_pool": HOME_POOL.get(ds),
                     "exact_in_home_train": sum(r["exact_home"] for r in R) / n,
                     "near_in_home_train": sum(r["near_home"] and not r["exact_home"] for r in R) / n,
                     "image_in_home_train": sum(r["img_home"] for r in R) / n,
                     "image_any_train_pool": sum(bool(r["img_pool"]) for r in R) / n,
                     "leaked_any": sum(r["leaked"] for r in R) / n,
                     "exact_short_le10": sum(r["exact_home"] and r["norm_len"] <= 10 for r in R) / n,
                     "exact_in_other_train_pools": {p: sum(p in r["exact_pools"] and p != HOME_POOL.get(ds) for r in R) / n for p in pools},
                     "text_dup_within_set": sum(r["dup_in_test_same_set"] > 0 for r in R) / n,
                     "text_dup_other_test_sets": dict(Counter(d for r in R for d in r["dup_in_other_test_sets"])),
                     "image_dup_within_panel": sum(bool(r["img_dup_test"]) for r in R) / n,
                     "image_dup_examples": [(r["sample_key"], r["img_dup_test"], r["img_dup_test_dist"]) for r in R if r["img_dup_test"]][:8]}
        if ds == "muharaf_lines":
            ex = Counter(norm_text(test[r["sample_key"]]["gt_raw"]) for r in R if r["exact_home"])
            table[ds]["muharaf_6pct_check"] = {"exact": sum(r["exact_home"] for r in R), "share": sum(r["exact_home"] for r in R) / n,
                                               "median_len_leaked": float(np.median([r["norm_len"] for r in R if r["exact_home"]] or [0])),
                                               "median_len_clean": float(np.median([r["norm_len"] for r in R if not r["leaked"]])),
                                               "top_texts": ex.most_common(15)}

    # ---- does the leak pay? difference-in-differences per set
    systems = load_systems(("e_no_diacritics", "n_no_diacritics"))
    did, pv, did_keys = {}, [], []
    for ds, runs in systems.items():
        ins = [r for r in runs if in_domain(r, ds) is True]
        outs = [r for r in runs if in_domain(r, ds) is False]
        if not ins or not outs:
            continue
        keys = [k for k in by_k if by_k[k]["dataset"] == ds and sum(k in runs[r] for r in ins) and sum(k in runs[r] for r in outs)]
        lcer = lambda r, k: min(1.0, runs[r][k]["e_no_diacritics"] / max(1, runs[r][k]["n_no_diacritics"]))
        gap = np.array([np.mean([lcer(r, k) for r in ins if k in runs[r]]) - np.mean([lcer(r, k) for r in outs if k in runs[r]]) for k in keys])
        lk = np.array([by_k[k]["leaked"] for k in keys])
        ln = np.array([by_k[k]["norm_len"] for k in keys])
        if lk.sum() < 5:
            did[ds] = {"leaked_lines": int(lk.sum()), "note": "fewer than 5 leaked lines with readings"}
            continue
        # length-matched: compare within length quintiles, weighted by leaked lines
        q = np.quantile(ln, [0.2, 0.4, 0.6, 0.8])
        band = np.searchsorted(q, ln)
        def stat(idx):
            g, l_, b = gap[idx], lk[idx], band[idx]
            vals, wts = [], []
            for bb in range(5):
                m1, m0 = (b == bb) & l_, (b == bb) & ~l_
                if m1.sum() and m0.sum():
                    vals.append(g[m1].mean() - g[m0].mean())
                    wts.append(m1.sum())
            return float(np.average(vals, weights=wts)) if vals else float("nan")
        est = stat(np.arange(len(keys)))
        rng = np.random.default_rng(1)
        boots = np.array([stat(rng.integers(0, len(keys), len(keys))) for _ in range(a.reps)])
        boots = boots[~np.isnan(boots)]
        did[ds] = {"in_domain_runs": ins, "out_domain_runs": len(outs), "lines": len(keys), "leaked_lines": int(lk.sum()),
                   "gap_leaked": float(gap[lk].mean()), "gap_clean": float(gap[~lk].mean()),
                   "did_length_matched": est, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   "p": boot_p(est, boots)}
        pv.append(did[ds]["p"])
        did_keys.append(ds)
    rej, adj = bh(pv)
    for ds, r_, q_ in zip(did_keys, rej, adj):
        did[ds]["q_bh"] = float(q_)
        did[ds]["survives_fdr"] = bool(r_)
        did[ds]["decision"] = ("report in-domain headline on clean lines only" if r_ and did[ds]["ci95"][1] < 0
                               else "flag and report; headline unchanged")

    res = {"decision_rule": __doc__.split("Decision rule (written before the run):")[1].split("Text pools")[0].strip(),
           "pools": {p: len(v) for p, v in pools.items()}, "image_calibration": calib, "table": table, "did": did}
    json.dump(res, open(a.out / "leakage.json", "w"), ensure_ascii=False, indent=1)
    cols = ["sample_key", "dataset", "norm_len", "exact_home", "near_home", "near_home_jaccard", "img_home", "img_dist", "img_dup_test", "leaked",
            "exact_home_example", "near_home_example", "img_pool", "img_id"]
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in cols}), a.out / "leakage_flags.parquet")

    # ---- gallery: leaked lines with their training twin, per set (best = biggest in-domain advantage on a leaked line, worst
    # = in-domain worse, random)
    imgs = panel_images({r["sample_key"] for r in rows if r["leaked"] or r["img_dup_test"]})
    trtext = {i: t for v in pools.values() for i, t in v}
    trimg_lab = {}
    for p, (ids, labs, _) in trimg.items():
        trimg_lab.update(zip(ids, labs))
    html = [HTML_HEAD.format(title="Leakage audit"), "<h1>Leakage and duplicates (checklist 2.9, R4.2)</h1>",
            f"<p class=mut>{res['decision_rule']}</p><h2>Per set</h2><table><tr><th>set</th><th>lines</th><th>home pool</th><th>exact text in train</th>"
            "<th>near text</th><th>image+text</th><th>any leak</th><th>short (≤10) exact</th><th>text dup in set</th><th>image dup in panel</th><th>DiD (in − out, leaked − clean) [95% CI] q</th></tr>"]
    for ds, t in table.items():
        d = did.get(ds, {})
        ds_did = f"{d['did_length_matched']*100:+.1f} pt [{d['ci95'][0]*100:+.1f}, {d['ci95'][1]*100:+.1f}] q={d.get('q_bh', float('nan')):.3f}" if "ci95" in d else d.get("note", "–")
        html.append(f"<tr><td>{ds}</td><td>{t['lines']}</td><td>{t['home_pool']}</td><td>{t['exact_in_home_train']:.1%}</td><td>{t['near_in_home_train']:.1%}</td>"
                    f"<td>{t['image_in_home_train']:.1%}</td><td><b>{t['leaked_any']:.1%}</b></td><td>{t['exact_short_le10']:.1%}</td><td>{t['text_dup_within_set']:.1%}</td>"
                    f"<td>{t['image_dup_within_panel']:.1%}</td><td>{ds_did}</td></tr>")
    html.append("</table>")
    html.append(f"<p>Image-hash calibration: {json.dumps(calib)}</p>")
    rnd = random.Random(0)
    for ds in table:
        L = [r for r in rows if r["dataset"] == ds and r["leaked"] and r["sample_key"] in imgs]
        if not L:
            continue
        html.append(f"<h2>{ds}: leaked lines ({len(L)})</h2>")
        pick = sorted(L, key=lambda r: -r["norm_len"])[:4] + sorted(L, key=lambda r: r["norm_len"])[:3] + rnd.sample(L, min(5, len(L)))
        seen = set()
        for r in pick:
            if r["sample_key"] in seen:
                continue
            seen.add(r["sample_key"])
            twin = r["exact_home_example"] or r["near_home_example"] or r["img_id"]
            twin_t = trtext.get(twin) or trimg_lab.get(twin, "")
            why = "exact text" if r["exact_home"] else ("near text J=%.2f" % r["near_home_jaccard"] if r["near_home"] else f"image d={r['img_dist']}")
            html.append(f"<div class=card><div class=mut>{r['sample_key']} · {why} · twin {twin}</div><img src='{png_b64(imgs[r['sample_key']])}'>"
                        f"<div class=ar>GT: {test[r['sample_key']]['gt_raw']}</div><div class=ar>train: {twin_t}</div></div>")
    D = [r for r in rows if r["img_dup_test"] and r["sample_key"] in imgs]
    if D:
        html.append(f"<h2>Image duplicates inside the panel ({len(D)})</h2>")
        more = panel_images({r["img_dup_test"] for r in D[:40]})
        for r in D[:40]:
            o = r["img_dup_test"]
            other = f"<img src='{png_b64(more[o])}'>" if o in more else ""
            html.append(f"<div class=card><div class=mut>{r['sample_key']} ↔ {o} (d={r['img_dup_test_dist']})</div>"
                        f"<img src='{png_b64(imgs[r['sample_key']])}'>{other}"
                        f"<div class=ar>{test[r['sample_key']]['gt_raw']}<br>{test[o]['gt_raw']}</div></div>")
    html.append("</body></html>")
    (a.out / "leakage_gallery.html").write_text("\n".join(html))
    print(json.dumps({ds: {k: v for k, v in t.items() if k in ("lines", "exact_in_home_train", "near_in_home_train", "image_in_home_train", "leaked_any", "text_dup_within_set", "image_dup_within_panel")} for ds, t in table.items()}, indent=1))
    print(json.dumps(did, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
