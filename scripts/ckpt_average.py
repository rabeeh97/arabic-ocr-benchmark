"""Average CRNN weight files (uniform checkpoint averaging) into one state dict.

    python scripts/ckpt_average.py RUN_DIR --glob 'model_ep*.pt' --out model_avg.pt

Float tensors (weights and BatchNorm running statistics) are averaged; integer buffers (num_batches_tracked) are taken from the last file.
Averaging BatchNorm statistics is an approximation (they are not recomputed on the training data); with the late, low-learning-rate epochs
of a one-cycle schedule the snapshots are close, so the error is small.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--glob", default="model_ep*.pt")
    ap.add_argument("--out", default="model_avg.pt")
    a = ap.parse_args()
    files = sorted(a.run.glob(a.glob), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if not files:
        raise SystemExit(f"no files match {a.glob} in {a.run}")
    states = [torch.load(f, map_location="cpu") for f in files]
    avg = {}
    for k, v in states[-1].items():
        avg[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype) if v.is_floating_point() else v.clone()
    torch.save(avg, a.run / a.out)
    print(f"averaged {len(files)} files ({files[0].name} .. {files[-1].name}) -> {a.run / a.out}")


if __name__ == "__main__":
    main()
