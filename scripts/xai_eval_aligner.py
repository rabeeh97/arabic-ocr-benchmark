"""CER of the trained CRNN-CTC aligner on datasets it never saw (generalisation check).

    python xai_eval_aligner.py --data eval_data.pkl --run aligner_run --ckpt model_final.pt --out eval_out.json

Reports CER per dataset, the share of reference characters that are not in the training charset
(they count as errors), and a few example readings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import CRNN, edit_distance, greedy, make_batch  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import pickle

    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    model = CRNN(len(chars) + 1).cuda().eval()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cuda"))
    rows = pickle.load(open(a.data, "rb"))["rows"]
    res: dict = {}
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds]
        err = tot = oov = 0
        per_line, ex = [], []
        for k in range(0, len(rs), 32):
            chunk = rs[k : k + 32]
            x, lens, tgt = make_batch(chunk, stoi, False, "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            for r, hyp, ref in zip(chunk, greedy(logits.float(), lens), tgt):
                e = edit_distance(hyp, ref)
                n = max(1, len(r["label"]))
                err += e
                tot += n
                oov += n - len(ref)
                per_line.append(e / n)
                if len(ex) < 4:
                    ex.append({"ref": r["label"], "hyp": "".join(itos.get(t, "?") for t in hyp)[::-1]})
        res[ds] = {"lines": len(rs), "cer": err / tot, "median_line_cer": float(np.median(per_line)), "oov_char_share": oov / tot, "examples": ex}
        print(ds, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[ds].items() if k != "examples"}, flush=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
