"""Score CRNN runs on the standard panel (xai_prepare_panel.py) in the store format (plan 2026-10-13, §3.2 / §3.9).

    python crnn_eval_panel.py --data panel.pkl --runs aligner_run aligner_run_rep2 ... --out-dir panel_out [--ckpt model_final.pt]
        [--dump-posteriors posteriors] [--label-map skeleton_map.json]

One line at a time in fp32, so a reading does not depend on batch padding or numerics (earlier evaluations padded batches
of 32 and changed 5–39 % of readings between packs). Per run, writes OUT_DIR/<run>.json:
  {"run", "ckpt", "config", "numerics": "fp32-bs1", "panel", "cer": {tag: micro CER},
   "per_line": {tag: {id: [edits, label_len, reading in logical order]}}, "conf": {tag: {id: mean max-prob of emitted frames}}}
Edits are on the model's charset with unknown label characters counted in the denominator, as geom_transfer_eval.py.
Results are written after every tag, and a rerun skips the tags already in the file.

Optional (checklist 4.1, decoding-and-restoration §0):
  --dump-posteriors DIR  also writes DIR/<run>__<data stem>.npz, one file per run and data file, holding the CTC
                         log-posteriors of every line (frames in visual order, the valid T = width // 4 frames only):
                         logp (sum T, C) float16, offsets (n + 1) int64 (line k is logp[offsets[k]:offsets[k + 1]]),
                         ids, tags (str arrays), chars (the charset; class 0 is the CTC blank, class k is chars[k - 1]).
                         Lines are scored for the dump even when their tag is already in the JSON (the JSON is then left as
                         it is). A run whose .npz already exists is skipped for the dump (rerun-safe).
  --label-map FILE       JSON {label: new label}; every label is mapped before scoring (e.g. dotless skeleton labels for
                         the skeleton-only CRNN, checklist 4.2). A label missing from the map is an error.
Without these flags the behaviour and output are exactly as before.
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
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-posteriors", type=Path, help="directory for one <run>__<data stem>.npz of float16 CTC log-posteriors per run")
    ap.add_argument("--label-map", type=Path, help="JSON {label: mapped label} applied to every label before scoring")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    blob = pickle.load(open(a.data, "rb"))
    rows = blob["rows"]
    if a.label_map:
        lm = json.loads(a.label_map.read_text(encoding="utf-8"))
        missing = {r["label"] for r in rows} - set(lm)
        if missing:
            raise SystemExit(f"--label-map lacks {len(missing)} labels, e.g. {sorted(missing)[:3]}")
        rows = [{**r, "label": lm[r["label"]]} for r in rows]
    tags = sorted({r["dataset"] for r in rows})
    if a.dump_posteriors:
        a.dump_posteriors.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        out = a.out_dir / f"{run.name}.json"
        res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        chars = json.loads((run / "charset.json").read_text(encoding="utf-8"))["chars"]
        stoi = {c: i + 1 for i, c in enumerate(chars)}
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {"width": 1.0}
        if res is None:
            res = {"run": run.name, "ckpt": a.ckpt, "config": cfg, "numerics": "fp32-bs1", "panel": blob.get("split"), "cer": {}, "per_line": {}, "conf": {}}
            if a.label_map:
                res["label_map"] = a.label_map.name
        dump = a.dump_posteriors / f"{run.name}__{a.data.stem}.npz" if a.dump_posteriors else None
        if dump is not None and dump.exists():
            print(f"{run.name}: posteriors already in {dump}", flush=True)
            dump = None
        if dump is None and all(t in res["per_line"] for t in tags):
            continue
        model = CRNN(len(chars) + 1, width=cfg.get("width", 1.0)).to(a.device).eval()
        model.load_state_dict(torch.load(run / a.ckpt, map_location=a.device))
        d_lp, d_id, d_tag = [], [], []
        for tag in tags:
            scored = tag in res["per_line"]
            if scored and dump is None:
                continue
            pl, cf = {}, {}
            for r in (r for r in rows if r["dataset"] == tag):
                x, lens, tgt = make_batch([r], stoi, False, a.device)
                with torch.no_grad():
                    logits = model(x).float()
                T = max(1, int(lens[0]) // 4)
                if dump is not None:
                    d_lp.append(logits[:T, 0].log_softmax(-1).cpu().numpy().astype(np.float16))
                    d_id.append(r["id"])
                    d_tag.append(tag)
                if scored:
                    continue
                hyp = greedy(logits, lens)[0]
                p = logits.softmax(-1)[:T, 0]
                best, arg = p.max(-1)
                cf[r["id"]] = round(float(best[arg != 0].mean()), 4) if (arg != 0).any() else 0.0
                pl[r["id"]] = [edit_distance(hyp, tgt[0]), max(1, len(r["label"])), "".join(chars[k - 1] for k in hyp)[::-1]]
            if scored:
                print(f"{run.name:24s} {tag:24s} posteriors only ({sum(t == tag for t in d_tag)} lines)", flush=True)
                continue
            res["per_line"][tag], res["conf"][tag] = pl, cf
            res["cer"][tag] = sum(v[0] for v in pl.values()) / max(1, sum(v[1] for v in pl.values()))
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            tmp.replace(out)
            print(f"{run.name:24s} {tag:24s} lines {len(pl):5d}  CER {100 * res['cer'][tag]:5.1f} %", flush=True)
        if dump is not None:
            off = np.zeros(len(d_lp) + 1, dtype=np.int64)
            off[1:] = np.cumsum([len(v) for v in d_lp])
            tmp = dump.with_name(dump.stem + ".tmp.npz")
            np.savez_compressed(tmp, logp=np.concatenate(d_lp), offsets=off, ids=np.array(d_id), tags=np.array(d_tag), chars=np.array(chars),
                                run=np.array(run.name), ckpt=np.array(a.ckpt), data=np.array(a.data.name))
            tmp.replace(dump)
            print(f"{run.name}: posteriors of {len(d_id)} lines -> {dump} ({dump.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
