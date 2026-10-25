"""B6: CTC prefix beam search with a character n-gram language model trained on TRAIN labels only.

    PYTHONPATH=src python scripts/rasam_lm_decode.py --data rasam_raw.pkl --run RUN_DIR --name NAME --train-sets r1 \
        --val-tag r1 --test-tags r1 [r2] [--order 6] [--val-n 150] [--beam 12] [--smoke]

The CRNN emits characters in VISUAL order (right edge of the line first), so the LM is trained on reversed labels and applied in the
same order; the output is reversed back to logical order. Witten-Bell interpolated n-gram, start and end symbols. alpha (LM weight) and
beta (per-character bonus) are tuned on a subset of the validation lines of --val-tag (never test), then used for every test line.
Reports greedy vs LM micro CER (raw profile) and writes results/rasam/lm/<name>__<tag>__test.json with the readings.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NEG = -1e30
BOS, EOS = "\x02", "\x03"


def logaddexp(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


class CharLM:
    """Witten-Bell interpolated character n-gram. logp(c, history) -> natural log probability."""

    def __init__(self, texts: list[str], order: int) -> None:
        self.order = order
        self.counts: dict[str, Counter] = defaultdict(Counter)
        vocab = set()
        for t in texts:
            s = BOS * (order - 1) + t + EOS
            vocab.update(t)
            for i in range(order - 1, len(s)):
                for k in range(order):  # context length k
                    self.counts[s[i - k:i]][s[i]] += 1
        self.v = len(vocab) + 1
        self.totals = {h: (sum(c.values()), len(c)) for h, c in self.counts.items()}
        self.cache: dict = {}

    def logp(self, c: str, hist: str) -> float:
        h = (BOS * (self.order - 1) + hist)[-(self.order - 1):] if self.order > 1 else ""
        key = (c, h)
        if key in self.cache:
            return self.cache[key]
        p = 1.0 / self.v
        for k in range(0, len(h) + 1):  # lowest order first
            ctx = h[len(h) - k:] if k else ""
            tot = self.totals.get(ctx)
            if not tot:
                continue
            n, u = tot
            p = (self.counts[ctx][c] + u * p) / (n + u)
        r = math.log(max(p, 1e-12))
        self.cache[key] = r
        return r


def prefix_beam(logp: np.ndarray, chars: list[str], lm: CharLM | None, alpha: float, beta: float, beam: int = 12, topk: int = 8) -> str:
    """logp: (T, C) log posteriors, class 0 = blank, class k = chars[k - 1]. Returns the best prefix in the model's (visual) order."""
    beams = {"": (0.0, NEG)}
    for t in range(logp.shape[0]):
        row = logp[t]
        cand = np.argsort(row)[::-1][:topk]
        nxt: dict[str, list[float]] = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            ptot = logaddexp(pb, pnb)
            for k in cand:
                p = float(row[k])
                if p < -12:
                    continue
                if k == 0:
                    e = nxt[prefix]
                    e[0] = logaddexp(e[0], ptot + p)
                    continue
                c = chars[k - 1]
                bonus = beta + (alpha * lm.logp(c, prefix) if lm is not None and alpha else 0.0)
                if prefix and c == prefix[-1]:
                    e = nxt[prefix]
                    e[1] = logaddexp(e[1], pnb + p)  # same character, no blank between: collapses into the same prefix
                    e = nxt[prefix + c]
                    e[1] = logaddexp(e[1], pb + p + bonus)  # a blank separated them: a genuine repeat
                else:
                    e = nxt[prefix + c]
                    e[1] = logaddexp(e[1], ptot + p + bonus)
        scored = sorted(nxt.items(), key=lambda kv: -logaddexp(kv[1][0], kv[1][1]))[:beam]
        beams = {k: (v[0], v[1]) for k, v in scored}
    best, best_s = "", NEG
    for prefix, (pb, pnb) in beams.items():
        s = logaddexp(pb, pnb) + (alpha * lm.logp(EOS, prefix) if lm is not None and alpha else 0.0)
        if s > best_s:
            best, best_s = prefix, s
    return best


def main() -> None:
    import importlib.util

    import torch

    from arabic_ocr_bench.metrics import aggregate, score_pair

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="model_best.pt")
    ap.add_argument("--train-sets", nargs="+", required=True)
    ap.add_argument("--val-tag", required=True)
    ap.add_argument("--test-tags", nargs="+", required=True)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--val-n", type=int, default=150)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.6, 1.0], help="LM weights to try on validation")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0], help="per-character bonuses to try on validation")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code path (10 validation lines, 20 test lines)")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("xai_train_aligner", Path(__file__).with_name("xai_train_aligner.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    cfg = json.loads((a.run / "config.json").read_text())
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    blob = pickle.load(open(a.data, "rb"))
    T.H = int(blob.get("H", 64))
    model = T.CRNN(len(chars) + 1, width=cfg["width"], height=int(cfg.get("height", 64)))
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu"))
    model.eval()
    rows = blob["rows"]
    train_txt = [r["label"][::-1] for r in rows if r["role"] == "train" and r["dataset"] in a.train_sets]
    lm = CharLM(train_txt, a.order)
    print(f"LM order {a.order} on {len(train_txt)} training labels (visual order), vocabulary {lm.v}", flush=True)

    def logits(r):
        with torch.no_grad():
            x, lens, _ = T.make_batch([r], stoi, False, "cpu")
            lp = model(x).float().log_softmax(-1)[:, 0, :]
        return lp[: max(1, int(lens[0]) // 4)].numpy()

    def greedy(lp):
        best = lp.argmax(-1)
        out, prev = [], 0
        for k in best:
            if k != prev and k != 0:
                out.append(chars[k - 1])
            prev = k
        return "".join(out)

    def cer_of(sel, decode):
        sc = [score_pair(r["label"], decode(r, lp)[::-1], ("raw",)) for r, lp in sel]
        return aggregate(sc, ("raw",))["micro_cer"]["raw"]

    val = [r for r in rows if r["role"] == "heldout" and r["dataset"] == a.val_tag]
    n_val = 10 if a.smoke else a.val_n
    step = max(1, len(val) // n_val)
    val = val[::step][:n_val]
    t0 = time.time()
    vsel = [(r, logits(r)) for r in val]
    base = cer_of(vsel, lambda r, lp: greedy(lp))
    print(f"validation subset n={len(vsel)}: greedy CER {base:.4f}", flush=True)
    best = (base, 0.0, 0.0)
    grid = [(0.3, 0.0)] if a.smoke else [(al, be) for al in a.alphas for be in a.betas]
    for al, be in grid:
        c = cer_of(vsel, lambda r, lp, al=al, be=be: prefix_beam(lp, chars, lm, al, be, a.beam))
        print(f"  alpha {al} beta {be}: validation CER {c:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if c < best[0]:
            best = (c, al, be)
    print(f"chosen on validation: alpha {best[1]} beta {best[2]} (CER {best[0]:.4f}, greedy {base:.4f})", flush=True)
    out_dir = Path("results/rasam/lm")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in a.test_tags:
        test = [r for r in rows if r["role"] == "test" and r["dataset"] == tag]
        if a.smoke:
            test = test[:20]
        g_sc, l_sc, lines = [], [], []
        for r in test:
            lp = logits(r)
            g = greedy(lp)[::-1]
            h = (prefix_beam(lp, chars, lm, best[1], best[2], a.beam) if best[1] or best[2] else greedy(lp))[::-1]
            g_sc.append(score_pair(r["label"], g, ("raw", "diac_insensitive")))
            l_sc.append(score_pair(r["label"], h, ("raw", "diac_insensitive")))
            lines.append({"id": r["id"], "manuscript": r["manuscript"], "page": r["page"], "region": r["region"], "ref": r["label"], "hyp": h, "greedy": g})
        ga, la = aggregate(g_sc, ("raw", "diac_insensitive")), aggregate(l_sc, ("raw", "diac_insensitive"))
        print(f"{a.name} {tag} test n={len(test)}: greedy {ga['micro_cer']['raw']:.4f}  with LM {la['micro_cer']['raw']:.4f} "
              f"(diac-insensitive {ga['micro_cer']['diac_insensitive']:.4f} -> {la['micro_cer']['diac_insensitive']:.4f})", flush=True)
        (out_dir / f"{a.name}__{tag}__test.json").write_text(json.dumps({
            "run": a.name, "run_dir": str(a.run), "dataset": tag, "split": "test", "n": len(test), "order": a.order, "alpha": best[1], "beta": best[2],
            "micro_cer": la["micro_cer"], "greedy_micro_cer": ga["micro_cer"], "lines": lines}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
