"""RS4: score a CRNN trained by scripts/xai_train_aligner.py on the RASAM test lines of a rasam_prepare_crnn_data.py pickle.

    PYTHONPATH=src .venv/bin/python scripts/rasam_crnn_eval.py --data rasam_raw.pkl --run RUN_DIR --name A1 [--ckpt model_final.pt]
        [--tags r1 r2] [--device cpu|mps|cuda] [--split test]

One line at a time in fp32 (no batch padding effects). Readings are reversed from the CRNN's visual order to logical order.
Characters outside the training charset in the reference count as errors (they are kept in the reference). Writes
results/rasam/crnn/<name>__<tag>__<split>.json with per-line readings and raw / light / diac_insensitive scores.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import torch

from arabic_ocr_bench.metrics import aggregate, score_pair

PROFILES = ("raw", "light", "diac_insensitive")


def load_trainer():
    spec = importlib.util.spec_from_file_location("xai_train_aligner", Path(__file__).with_name("xai_train_aligner.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--tags", nargs="+", default=["r1", "r2"])
    ap.add_argument("--split", default="test", choices=["test", "heldout"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    T = load_trainer()
    cfg = json.loads((a.run / "config.json").read_text())
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    itos = {i + 1: c for i, c in enumerate(chars)}
    model = T.CRNN(len(chars) + 1, width=cfg["width"]).to(a.device)
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location=a.device))
    model.eval()
    rows = [r for r in pickle.load(open(a.data, "rb"))["rows"] if r["role"] == a.split]
    out_dir = Path("results/rasam/crnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in a.tags:
        sel = [r for r in rows if r["dataset"] == tag]
        if not sel:
            continue
        t0 = time.time()
        scores, lines = [], []
        with torch.no_grad():
            for r in sel:
                x, lens, _ = T.make_batch([r], {c: i + 1 for i, c in enumerate(chars)}, False, a.device)
                seq = T.greedy(model(x).float(), lens)[0]
                hyp = "".join(itos[k] for k in seq)[::-1]  # visual -> logical
                scores.append(score_pair(r["label"], hyp, PROFILES))
                lines.append({"id": r["id"], "manuscript": r["manuscript"], "page": r["page"], "region": r["region"],
                              "ref": r["label"], "hyp": hyp, "w": int(r["img"].shape[1])})
        agg = aggregate(scores, PROFILES)
        res = {"run": a.name, "run_dir": str(a.run), "ckpt": a.ckpt, "config": cfg, "dataset": tag, "split": a.split, "n": len(sel),
               "micro_cer": agg["micro_cer"], "macro_cer": agg["cer_macro"], "micro_wer": agg["micro_wer"],
               "seconds": round(time.time() - t0, 1), "lines": lines}
        (out_dir / f"{a.name}__{tag}__{a.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(a.name, tag, a.split, "n =", len(sel), "micro CER", {k: round(v, 4) for k, v in agg["micro_cer"].items()}, flush=True)


if __name__ == "__main__":
    main()
