"""Score CRNN runs on the standard panel (xai_prepare_panel.py) in the store format (plan 2026-10-13, §3.2 / §3.9).

    python crnn_eval_panel.py --data panel.pkl --runs aligner_run aligner_run_rep2 ... --out-dir panel_out [--ckpt model_final.pt]

One line at a time in fp32, so a reading does not depend on batch padding or numerics (earlier evaluations padded batches
of 32 and changed 5–39 % of readings between packs). Per run, writes OUT_DIR/<run>.json:
  {"run", "ckpt", "config", "numerics": "fp32-bs1", "panel", "cer": {tag: micro CER},
   "per_line": {tag: {id: [edits, label_len, reading in logical order]}}, "conf": {tag: {id: mean max-prob of emitted frames}}}
Edits are on the model's charset with unknown label characters counted in the denominator, as geom_transfer_eval.py.
Results are written after every tag, and a rerun skips the tags already in the file.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import CRNN, edit_distance, greedy, make_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    blob = pickle.load(open(a.data, "rb"))
    rows = blob["rows"]
    tags = sorted({r["dataset"] for r in rows})
    for run in a.runs:
        out = a.out_dir / f"{run.name}.json"
        res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        chars = json.loads((run / "charset.json").read_text(encoding="utf-8"))["chars"]
        stoi = {c: i + 1 for i, c in enumerate(chars)}
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {"width": 1.0}
        if res is None:
            res = {"run": run.name, "ckpt": a.ckpt, "config": cfg, "numerics": "fp32-bs1", "panel": blob.get("split"), "cer": {}, "per_line": {}, "conf": {}}
        model = CRNN(len(chars) + 1, width=cfg.get("width", 1.0)).to(a.device).eval()
        model.load_state_dict(torch.load(run / a.ckpt, map_location=a.device))
        for tag in tags:
            if tag in res["per_line"]:
                continue
            pl, cf = {}, {}
            for r in (r for r in rows if r["dataset"] == tag):
                x, lens, tgt = make_batch([r], stoi, False, a.device)
                with torch.no_grad():
                    logits = model(x).float()
                hyp = greedy(logits, lens)[0]
                p = logits.softmax(-1)[: max(1, int(lens[0]) // 4), 0]
                best, arg = p.max(-1)
                cf[r["id"]] = round(float(best[arg != 0].mean()), 4) if (arg != 0).any() else 0.0
                pl[r["id"]] = [edit_distance(hyp, tgt[0]), max(1, len(r["label"])), "".join(chars[k - 1] for k in hyp)[::-1]]
            res["per_line"][tag], res["conf"][tag] = pl, cf
            res["cer"][tag] = sum(v[0] for v in pl.values()) / max(1, sum(v[1] for v in pl.values()))
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            tmp.replace(out)
            print(f"{run.name:24s} {tag:24s} lines {len(pl):5d}  CER {100 * res['cer'][tag]:5.1f} %", flush=True)


if __name__ == "__main__":
    main()
