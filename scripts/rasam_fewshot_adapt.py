"""Few-shot adaptation to an unseen manuscript (2026-10-28): how many transcribed lines of a new manuscript close the leave-manuscript-out gap?

    python scripts/rasam_fewshot_adapt.py --pack PACK_LOMO.pkl --run FOLD_RUN_DIR --out OUT.json [--ks 0 10 25 50 100 0] [--seeds 0 1]

For each RASAM-2 manuscript held out of the fold model's training (pack rows tagged r2ho, see rasam_prepare_crnn_data.py --holdout):
  - adaptation pool = its lines outside the official test split (orig_split train / validation); K lines are sampled from it (seeded);
    K = -1 means the whole pool (about 225 lines, close to what the in-split models saw of the manuscript);
  - the fold model is fine-tuned on the K lines plus the same number (at least 32) of replay lines from its own training data, for
    --epochs passes over that mix (AdamW, cosine schedule, the trainer's augmentation), so it does not forget the other manuscripts;
  - the manuscript's OFFICIAL test lines are read greedily and with the char 8-gram LM (training labels of r1 / r2 / tm, built once per
    fold; the K lines are left out of it, they would be under 2 % of its text; alpha / beta fixed to the leave-manuscript-out run's 0.4 / 1.0).
K = -1 runs one seed only (the sample is the whole pool).
Caveat: RASAM-2's split is random by line, so the adaptation lines come from the same pages as the test lines, as in the in-split setting.
Writes per-manuscript, per-K, per-seed micro CER (raw: NFC, whitespace collapsed) and the readings.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import pickle
import random
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = load("xai_train_aligner")
D = load("rasam_lm_decode")


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def micro(pairs) -> float:
    e = n = 0
    for ref, hyp in pairs:
        r, h = norm(ref), norm(hyp)
        e += T.edit_distance(list(r), list(h))
        n += len(r)
    return 100 * e / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_best.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 10, 25, 50, 100, -1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    cfg = json.loads((a.run / "config.json").read_text())
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    blob = pickle.load(open(a.pack, "rb"))
    T.H = int(blob.get("H", 96))
    T.GEOM_AUG = True
    base = T.build_model(len(chars) + 1, cfg)
    base.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu"))
    rows = blob["rows"]
    train = [r for r in rows if r["role"] == "train" and r["dataset"] in ("r1", "r2", "tm")]
    train = [r for r in train if r["label"] and r["img"].shape[1] // 4 >= len(r["label"]) + 2]
    lm_text = [r["label"][::-1] for r in train]
    lm = D.CharLM(lm_text, 8)
    held = sorted({r["manuscript"] for r in rows if r["dataset"] == "r2ho"})
    print(f"fold model {a.run.name}; held-out manuscripts {held}; {len(train)} replay candidates; device {dev}", flush=True)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    res = {"run": str(a.run), "ckpt": a.ckpt, "epochs": a.epochs, "lr": a.lr, "alpha": a.alpha, "beta": a.beta, "results": []}

    def read(model, test, lm):
        model.eval()
        out = []
        with torch.no_grad():
            for r in test:
                x, lens, _ = T.make_batch([r], stoi, False, dev)
                lp = model(x).float().log_softmax(-1)[:, 0, :][: max(1, int(lens[0]) // 4)].cpu().numpy()
                best, prev, g = lp.argmax(-1), 0, []
                for k in best:
                    if k != prev and k != 0:
                        g.append(chars[k - 1])
                    prev = k
                h = D.prefix_beam(lp, chars, lm, a.alpha, a.beta, 12)
                out.append({"id": r["id"], "ref": r["label"], "greedy": "".join(g)[::-1], "hyp": h[::-1]})
        return out

    for ms in held:
        pool = [r for r in rows if r["dataset"] == "r2ho" and r["manuscript"] == ms and r.get("orig_split") != "test"]
        pool = [r for r in pool if r["label"] and r["img"].shape[1] // 4 >= len(r["label"]) + 2]
        test = [r for r in rows if r["dataset"] == "r2ho" and r["manuscript"] == ms and r.get("orig_split") == "test"]
        for k in a.ks:
            for seed in (a.seeds if k > 0 else [0]):
                t0 = time.time()
                rng = random.Random(1000 * seed + 7)
                sel = list(pool) if k < 0 else rng.sample(pool, min(k, len(pool)))
                model = copy.deepcopy(base).to(dev)
                steps = 0
                if sel:
                    random.seed(seed); torch.manual_seed(seed)
                    replay = rng.sample(train, max(32, len(sel)))
                    mix = sel + replay
                    per_ep = math.ceil(len(mix) / a.bs)
                    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * per_ep)
                    model.train()
                    for ep in range(a.epochs):
                        order = list(range(len(mix)))
                        random.shuffle(order)
                        for b in range(0, len(order), a.bs):
                            x, lens, tgt = T.make_batch([mix[i] for i in order[b:b + a.bs]], stoi, True, dev)
                            flat = torch.tensor([t for s in tgt for t in s], device=dev)
                            tl = torch.tensor([len(s) for s in tgt], device=dev)
                            lp = model(x).float().log_softmax(-1)
                            loss = ctc(lp, flat, torch.clamp(lens // 4, min=1), tl)
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                            opt.step(); sched.step()
                            steps += 1
                lines = read(model, test, lm)
                g = micro([(x["ref"], x["greedy"]) for x in lines])
                l = micro([(x["ref"], x["hyp"]) for x in lines])
                res["results"].append({"manuscript": ms, "k": k, "k_used": len(sel), "pool": len(pool), "seed": seed, "steps": steps,
                                       "n_test": len(test), "greedy": round(g, 2), "lm": round(l, 2), "lines": lines})
                print(f"{ms:22} K={k:4} (used {len(sel):3}/{len(pool)}) seed {seed}: greedy {g:6.2f}  LM {l:6.2f}  "
                      f"[{steps} steps, {time.time() - t0:.0f}s]", flush=True)
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
