"""Forced alignment of known transcripts to the trained CRNN-CTC: character spans for the panel lines (X-B1).

    python xai_align.py --data aligner_data.pkl --run aligner_run --ckpt model_best.pt \
        --lines lines --lines errlines --out spans_out

For every panel row it writes spans.json (logical-order characters with [x0, x1] in 0..1 image
coordinates measured from the LEFT edge of the image, plus a confidence) and an overlay PNG.
Also reports an independent sanity check: the share of aligned spaces that fall in low-ink columns,
against randomly shifted copies of the same spans.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import CRNN, normalise  # noqa: E402

NEG = -1e30


def force_align(lp: np.ndarray, tgt: list[int]):
    """CTC Viterbi. lp: (T, C) log-probs. Returns per-frame state path over the 2L+1 extended labels."""
    T, _ = lp.shape
    L = len(tgt)
    if T < L:
        return None
    S = 2 * L + 1
    ext = np.zeros(S, dtype=int)
    ext[1::2] = tgt
    allow2 = np.zeros(S, dtype=bool)
    for s in range(2, S):
        allow2[s] = ext[s] != 0 and ext[s] != ext[s - 2]
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = lp[0, 0]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.where(allow2, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        stack = np.stack([prev, c1, c2])
        bp[t] = stack.argmax(0)
        dp[t] = stack.max(0) + lp[t, ext]
    s = S - 1 if S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=int)
    for t in range(T - 1, 0, -1):
        path[t] = s
        s -= int(bp[t, s])
    path[0] = s
    return path


def char_spans(path: np.ndarray, L: int, T: int):
    """Contiguous [start, end) frame spans per visual-order character, split halfway between neighbours."""
    firsts, lasts = [], []
    for i in range(L):
        fr = np.where(path == 2 * i + 1)[0]
        firsts.append(int(fr[0]))
        lasts.append(int(fr[-1]))
    bounds = [(lasts[i] + firsts[i + 1] + 1) / 2 for i in range(L - 1)]
    return [((firsts[0] if i == 0 else bounds[i - 1]), (lasts[-1] + 1 if i == L - 1 else bounds[i])) for i in range(L)]


def otsu(g: np.ndarray) -> int:
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    tot, sa, w0, s0, best, thr = hist.sum(), (np.arange(256) * hist).sum(), 0.0, 0.0, -1.0, 127
    for t in range(256):
        w0 += hist[t]
        if w0 == 0 or w0 == tot:
            continue
        s0 += t * hist[t]
        b = w0 * (tot - w0) * (s0 / w0 - (sa - s0) / (tot - w0)) ** 2
        if b > best:
            best, thr = b, t
    return thr


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_best.pt")
    ap.add_argument("--lines", type=Path, action="append", required=True, help="dirs with manifest.json (to draw overlays on the original images)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    model = CRNN(len(chars) + 1).cuda().eval()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cuda"))
    rows = [r for r in pickle.load(open(a.data, "rb"))["rows"] if r["role"] == "panel"]
    files = {}
    for d in a.lines:
        for it in json.loads((d / "manifest.json").read_text(encoding="utf-8")):
            files[it["id"]] = d / it["file"]
    rng = np.random.default_rng(0)
    out, space_hits, space_rand, n_space, fails = [], 0, 0, 0, 0
    tall = {k: [] for k in ("alef", "alef_shift", "lam", "lam_shift", "other", "other_shift")}
    for k, r in enumerate(rows):
        arr = r["img"]
        w = arr.shape[1]
        x = np.ones((1, 1, 64, int(np.ceil(w / 4) * 4)), dtype=np.float32)
        x[0, 0, :, :w] = normalise(arr)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(torch.from_numpy(x).cuda())
        T = w // 4
        lp = logits[:T, 0].float().log_softmax(-1).cpu().numpy().astype(np.float64)
        label_vis = r["label"][::-1]
        tgt = [stoi[c] for c in label_vis if c in stoi]
        if len(tgt) != len(label_vis):
            fails += 1
            continue
        path = force_align(lp, tgt)
        if path is None:
            fails += 1
            continue
        L = len(tgt)
        spans = char_spans(path, L, T)
        conf = [float(np.exp(max(lp[t, tgt[i]] for t in np.where(path == 2 * i + 1)[0]))) for i in range(L)]
        # logical order: logical index i <-> visual index L-1-i
        logical = []
        for i, ch in enumerate(r["label"]):
            v = L - 1 - i
            s0, s1 = spans[v]
            logical.append({"ch": ch, "x0": s0 * 4 / w, "x1": min(1.0, s1 * 4 / w), "conf": conf[v]})
        # independent check: aligned spaces should sit in low-ink columns
        g = arr
        ink = (g < otsu(g)).astype(float)
        ink[: 6], ink[-6:] = 0, 0
        prof = np.convolve(ink.mean(0), np.ones(3) / 3, mode="same")
        thr = 0.5 * prof[prof > 0].mean() if (prof > 0).any() else 0
        # second independent check: alef and lam are tall single strokes, so the tallest ink column inside their span
        # should be taller than for other letters and than for the same span shifted to a random place
        col_h = ink.sum(0) / ink.shape[0]
        for c in logical:
            if c["ch"] == " ":
                continue
            a0, a1 = int(c["x0"] * w), max(int(c["x0"] * w) + 1, int(c["x1"] * w))
            kind = "alef" if c["ch"] in "اأإآ" else ("lam" if c["ch"] == "ل" else "other")
            tall[kind].append(float(col_h[a0:a1].max()))
            sh = int(rng.integers(0, w))
            tall[kind + "_shift"].append(float(np.concatenate([col_h, col_h])[(a0 + sh) % w : (a0 + sh) % w + (a1 - a0)].max()))
        for c in logical[1:-1]:
            if c["ch"] != " ":
                continue
            a0, a1 = int(c["x0"] * w), max(int(c["x0"] * w) + 1, int(c["x1"] * w))
            n_space += 1
            space_hits += int(prof[a0:a1].mean() < thr)
            sh = int(rng.integers(0, w))
            b0, b1 = (a0 + sh) % w, (a0 + sh) % w + (a1 - a0)
            seg = np.concatenate([prof, prof])[b0:b1]
            space_rand += int(seg.mean() < thr)
        out.append({"id": r["id"], "dataset": r["dataset"], "label": r["label"], "chars": logical, "mean_conf": float(np.mean(conf))})
        # overlay on the original image
        if r["id"] in files:
            im = Image.open(files[r["id"]]).convert("RGB")
            im = im.resize((1400, max(40, round(im.height * 1400 / im.width))))
            ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(ov)
            pal = [(230, 60, 60, 70), (40, 130, 230, 70), (30, 170, 90, 70)]
            for j, c in enumerate(logical):
                x0, x1 = c["x0"] * im.width, c["x1"] * im.width
                col = (240, 200, 0, 90) if c["ch"] == " " else pal[j % 3]
                dr.rectangle([x0, 0, x1, im.height], fill=col)
                dr.line([x0, 0, x0, im.height], fill=(0, 0, 0, 120), width=1)
            Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB").save(a.out / f"{k:03d}.png")
            out[-1]["overlay"] = f"{k:03d}.png"
    (a.out / "spans.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summ = {"lines": len(out), "failed": fails, "mean_conf": float(np.mean([o["mean_conf"] for o in out])) if out else None,
            "spaces_checked": n_space, "spaces_in_low_ink": space_hits, "random_shift_in_low_ink": space_rand,
            "tallest_column_height": {k: {"n": len(v), "mean": float(np.mean(v)) if v else None} for k, v in tall.items()}}
    (a.out / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
