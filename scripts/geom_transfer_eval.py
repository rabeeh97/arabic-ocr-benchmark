"""CER of a CRNN-CTC aligner on original and transformed lines, with paired differences (X-T2).

    python geom_transfer_eval.py --data transfer_eval.pkl --run aligner_run --ckpt model_final.pt --out transfer_cer.json [--device mps]

The pickle comes from geom_transfer_export.py: the same line ids appear untouched (tag T) and transformed (tag T__as_X).
Reports CER per tag and, for every transformed tag, the paired change against the untouched lines with a 95 % bootstrap
interval over lines. Reading the model's width from run/config.json lets it score the half-width laptop replica as well as
the 3090 model. CER is the sum of edit distances over the sum of label lengths; characters outside the model's charset
count in the denominator but cannot be matched.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import CRNN, edit_distance, greedy, make_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    dev = a.device
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    cfg = json.loads((a.run / "config.json").read_text()) if (a.run / "config.json").exists() else {"width": 1.0}
    model = CRNN(len(chars) + 1, width=cfg["width"]).to(dev).eval()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location=dev))
    rows = pickle.load(open(a.data, "rb"))["rows"]
    per: dict[str, dict[str, tuple[int, int]]] = {}
    for tag in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == tag]
        per[tag] = {}
        for k in range(0, len(rs), 32):
            chunk = rs[k : k + 32]
            x, lens, tgt = make_batch(chunk, stoi, False, dev)
            with torch.no_grad():
                logits = model(x).float()
            for r, hyp, ref in zip(chunk, greedy(logits, lens), tgt):
                per[tag][r["id"]] = (edit_distance(hyp, ref), max(1, len(r["label"])))
        e = sum(v[0] for v in per[tag].values())
        n = sum(v[1] for v in per[tag].values())
        print(f"{tag:32s} lines {len(rs):4d}  CER {100 * e / n:5.1f} %", flush=True)
    res = {"cer": {t: sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for t, d in per.items()}, "lines": {t: len(d) for t, d in per.items()}, "paired": {}}
    rng = np.random.default_rng(0)
    for tag, d in per.items():
        if "__" not in tag:
            continue
        base = per[tag.split("__")[0]]
        ids = [i for i in d if i in base]
        eb = np.array([base[i][0] for i in ids], float)
        et = np.array([d[i][0] for i in ids], float)
        nn_ = np.array([base[i][1] for i in ids], float)
        diff = (et.sum() - eb.sum()) / nn_.sum()
        boots = []
        for _ in range(a.reps):
            s = rng.integers(0, len(ids), len(ids))
            boots.append((et[s].sum() - eb[s].sum()) / nn_[s].sum())
        res["paired"][tag] = {"lines": len(ids), "cer_original": float(eb.sum() / nn_.sum()), "cer_transformed": float(et.sum() / nn_.sum()), "change": float(diff),
                              "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                              "share_lines_better": float((et < eb).mean()), "share_lines_worse": float((et > eb).mean())}
        p = res["paired"][tag]
        print(f"{tag:32s} {100 * p['cer_original']:5.1f} -> {100 * p['cer_transformed']:5.1f}  change {100 * diff:+.1f} points  95% CI [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}]  better/worse lines {p['share_lines_better']:.2f}/{p['share_lines_worse']:.2f}", flush=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
