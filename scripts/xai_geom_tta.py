"""X-G3 / X-G8 on the 3090: shear sweep and confidence-selected test-time augmentation for the CRNN-CTC aligner.

    python xai_geom_tta.py --data eval_data.pkl --run aligner_run --ckpt model_final.pt --out tta_out.json

For every line the model reads the original and sheared copies (slant variants, in degrees; positive leans the
strokes to the right). Reports, per dataset: CER of each variant alone, CER when the reading with the highest
confidence is kept, and the oracle (best variant per line, a ceiling that no selector can beat).
Confidence = mean over non-blank frames of the best-class probability, the same quantity that separated wrong from
correct tokens in the attribution work.

STATUS: written on the laptop, where only the shear function (run it with --selftest) could be checked.
The model call copies xai_eval_aligner.py. Run a small --n first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SHEARS = [-6, 0, 6, 12, 18]  # Muharaf, the training pool, leans right by about 11 degrees; KHATT is upright (see the geometry run)


def shear(img: np.ndarray, deg: float) -> np.ndarray:
    """Slant a grey line image by deg: rows are shifted horizontally in proportion to their distance from the middle,
    paper-coloured fill. Positive deg leans the strokes to the right (top moves right)."""
    if deg == 0:
        return img
    h, w = img.shape
    t = math.tan(math.radians(deg))
    pad = int(math.ceil(abs(t) * h / 2)) + 1
    fill = int(np.median(img))
    out = np.full((h, w + 2 * pad), fill, np.uint8)
    for y in range(h):
        dx = int(round(-(y - h / 2) * t))  # top rows (y small) move right for positive deg
        out[y, pad + dx : pad + dx + w] = img[y]
    return out


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=300, help="lines per dataset")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        im = np.full((64, 200), 255, np.uint8)
        im[10:54, 100] = 0  # one vertical stroke
        for d in (-10, 0, 10):
            xs = np.nonzero(shear(im, d) < 128)
            top, bottom = xs[1][xs[0] == 12].mean(), xs[1][xs[0] == 52].mean()
            print(f"shear {d:+d}: stroke x at top row {top:.1f}, bottom row {bottom:.1f}")
        return
    import pickle

    import torch
    from xai_train_aligner import CRNN, edit_distance, greedy, make_batch

    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    model = CRNN(len(chars) + 1).cuda().eval()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cuda"))
    rows = pickle.load(open(a.data, "rb"))["rows"]
    res = {}
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds][: a.n]
        per_variant = {d: [] for d in SHEARS}  # (errors, chars, confidence) per line
        for r in rs:
            for d in SHEARS:
                row = {**r, "img": shear(r["img"], d)}
                x, lens, tgt = make_batch([row], stoi, False, "cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x).float()
                hyp = greedy(logits, lens)[0]
                p = logits.softmax(-1)[:, 0]  # T, C
                best, arg = p.max(-1)
                conf = float(best[arg != 0].mean()) if (arg != 0).any() else 0.0
                per_variant[d].append((edit_distance(hyp, tgt[0]), max(1, len(tgt[0])), conf))
        tot = lambda f: sum(f(i) for i in range(len(rs)))  # noqa: E731
        n_chars = tot(lambda i: per_variant[0][i][1])
        out = {"lines": len(rs), "variant_cer": {d: tot(lambda i: per_variant[d][i][0]) / n_chars for d in SHEARS}}
        out["confidence_selected_cer"] = tot(lambda i: per_variant[max(SHEARS, key=lambda d: per_variant[d][i][2])][i][0]) / n_chars
        out["oracle_cer"] = tot(lambda i: min(per_variant[d][i][0] for d in SHEARS)) / n_chars
        res[ds] = out
        print(ds, json.dumps(out), flush=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
