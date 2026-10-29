# Runs INSIDE the Colab VM (or locally). Cross-fitted readings: a fold model reads the pack's test-role lines (the other fold's pages).
#   python colab_remote_readcf.py PACK RUN_DIR OUT_JSON [--ckpt model_best.pt]
# Writes {line key: {"hyp", "ref", "ms", "page", "region", "w"}}; greedy CTC, one line at a time, readings in logical order.
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("pack")
ap.add_argument("run", type=Path)
ap.add_argument("out")
ap.add_argument("--ckpt", default="model_best.pt")
ap.add_argument("--fold", help="folds.json: read every line (any split) whose page is NOT in --fold-index (pages in no fold: fold-index 0 "
                "reads them) instead of the test-role lines")
ap.add_argument("--fold-index", type=int, default=0)
a = ap.parse_args()
folds = json.load(open(a.fold)) if a.fold else None
spec = importlib.util.spec_from_file_location("xai_train_aligner", Path(__file__).with_name("xai_train_aligner.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.loads((a.run / "config.json").read_text())
chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
itos = {i + 1: c for i, c in enumerate(chars)}
stoi = {c: i + 1 for i, c in enumerate(chars)}
blob = pickle.load(open(a.pack, "rb"))
T.H = int(blob.get("H", 64))
model = T.build_model(len(chars) + 1, cfg).to(dev)
model.load_state_dict(torch.load(a.run / a.ckpt, map_location=dev))
model.eval()
out = {}
with torch.no_grad():
    for r in blob["rows"]:
        if folds is None and r["role"] != "test":
            continue
        if folds is not None:
            f = folds.get(f"{r['manuscript']}|{r['page']}")
            if (f is None and a.fold_index != 0) or f == a.fold_index:
                continue
        x, lens, _ = T.make_batch([r], stoi, False, dev)
        seq = T.greedy(model(x).float(), lens)[0]
        out[r["key"]] = {"hyp": "".join(itos[k] for k in seq)[::-1], "ref": r["label"], "ms": r["manuscript"], "page": r["page"],
                         "region": r["region"], "w": int(r["img"].shape[1])}
json.dump(out, open(a.out, "w"), ensure_ascii=False)
print("read", len(out), "lines ->", a.out, flush=True)
