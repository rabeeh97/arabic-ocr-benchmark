"""Run the real CTC aligner (std-split CRNN) on one real line and save what the timeline needs to draw how letter spans are made:
the frames, the per-frame belief, the greedy reading, and the forced-alignment spans -> docs/research/timeline_ctc.json.

  PYTHONPATH=scripts .geomvenv/bin/python scripts/extract_ctc_example.py
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "scripts")
from xai_align import char_spans, force_align  # noqa: E402
from xai_train_aligner import CRNN, normalise  # noqa: E402

RUN = Path("/Volumes/PortableSSD/aocr-box-backup/xai-2026-10-13/xai/aligner/aligner_run_std_s0")
OUT = Path("docs/research/timeline_ctc.json")
chars = json.loads((RUN / "charset.json").read_text(encoding="utf-8"))["chars"]
stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 is the CTC blank
model = CRNN(len(chars) + 1).eval()
model.load_state_dict(torch.load(RUN / "model_final.pt", map_location="cpu"))
rows = pickle.load(open("results/panel/panel.pkl", "rb"))["rows"]
cands = [r for r in rows if r["dataset"] == "kitab_khatt__inkcrop" and r.get("ctc_ok") and 18 <= len(r["label"]) <= 30 and 320 <= r["img"].shape[1] <= 640]
print(len(cands), "candidate lines")
best = None
for r in cands[:60]:
    arr = r["img"]
    w = arr.shape[1]
    x = np.ones((1, 1, 64, int(np.ceil(w / 4) * 4)), dtype=np.float32)
    x[0, 0, :, :w] = normalise(arr)
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
    T = w // 4
    lp = logits[:T, 0].float().log_softmax(-1).numpy().astype(np.float64)
    vis = r["label"][::-1]  # the network scans left to right, Arabic is written right to left
    tgt = [stoi[c] for c in vis if c in stoi]
    if len(tgt) != len(vis):
        continue
    path = force_align(lp, tgt)
    if path is None:
        continue
    pred = lp.argmax(-1)
    greedy = [c for k, c in enumerate(pred) if c != 0 and (k == 0 or c != pred[k - 1])]
    greedy_vis = "".join(chars[c - 1] for c in greedy)
    # prefer a line the model reads nearly right, so the three stages tell one consistent story
    mism = sum(a != b for a, b in zip(greedy_vis, vis)) + abs(len(greedy_vis) - len(vis))
    if best is None or mism < best[0]:
        best = (mism, r, lp, path, tgt, vis, greedy_vis, w, T)
    if mism <= 1:
        break
mism, r, lp, path, tgt, vis, greedy_vis, w, T = best
L = len(tgt)
spans = char_spans(path, L, T)
conf = [float(np.exp(max(lp[t, tgt[i]] for t in np.where(path == 2 * i + 1)[0]))) for i in range(L)]
frames = [[int(lp[t].argmax()), round(float(np.exp(lp[t].max())), 3)] for t in range(T)]
img = Image.fromarray(r["img"])
buf = io.BytesIO()
img.resize((img.width * 2, img.height * 2), Image.LANCZOS).save(buf, format="WEBP", quality=82, method=6)
out = {
    "id": r["id"], "label": r["label"], "label_visual": vis, "w": int(w), "T": int(T), "px_per_frame": 4, "run": RUN.name, "alphabet": chars,
    "frames": frames,
    "spans": [{"ch": vis[v], "f0": int(spans[v][0]), "f1": int(spans[v][1]), "conf": round(conf[v], 3)} for v in range(L)],
    "greedy_visual": greedy_vis, "greedy": greedy_vis[::-1], "mismatches": int(mism),
    "img": "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(),
}
OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf8")
widths = [s["f1"] - s["f0"] + 1 for s in out["spans"]]
print("line", r["id"], "frames", T, "letters", L, "greedy mismatches", mism, "median span frames", int(np.median(widths)))
print("truth :", r["label"])
print("greedy:", out["greedy"])
