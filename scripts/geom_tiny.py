"""A tiny CRNN-CTC that trains on the laptop CPU, for the geometry-transfer and augmentation tests.

    .geomvenv/bin/python scripts/geom_tiny.py pack  OUT_DIR [--n-muharaf 6000]          # training data -> memmap
    .geomvenv/bin/python scripts/geom_tiny.py packeval OUT_DIR results/geometry/transfer_eval.pkl
    .geomvenv/bin/python scripts/geom_tiny.py train OUT_DIR RUN [--aug base|geom] [--epochs 25]
    .geomvenv/bin/python scripts/geom_tiny.py eval  OUT_DIR RUN                           # per-line errors on every eval tag

Same architecture as the 3090 aligner (xai_train_aligner.CRNN) at a quarter of the channels (about 0.6 M parameters),
64 px input. Images live in one uint8 memmap on the SSD so the training process stays small (the Mac is short of RAM).
Batches are capped by total pixels, the model checkpoints every epoch and resumes.

Augmentation
    base  what the 3090 trainer does: width scale, brightness and contrast, Gaussian noise, a small roll.
    geom  base plus, with probability 0.8, the geometry-transfer steps with random strengths inside the range measured
          across Muharaf, NAKBA and KHATT: shear -12..+20 degrees (catalogue item 1, safe), stroke width -0.5..+0.7 px on
          each axis (item 3, mild dilation and erosion only), blur or unsharp, ink and paper tone, paper noise (item 4).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
H = 64


# ------------------------------------------------------------------ packing
def pack(rows, out: Path, name: str):
    widths = np.array([r["img"].shape[1] for r in rows], np.int64)
    offs = np.concatenate([[0], np.cumsum(widths * H)])
    mm = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.uint8, shape=(int(offs[-1]),))
    for r, o, w in zip(rows, offs, widths):
        mm[o : o + w * H] = r["img"].reshape(-1)
    mm.flush()
    meta = [{"id": r["id"], "dataset": r["dataset"], "role": r.get("role", ""), "label": r["label"], "w": int(w), "off": int(o)} for r, o, w in zip(rows, offs, widths)]
    (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(name, len(rows), "lines", f"{offs[-1] / 1e6:.0f} MB")


def load(out: Path, name: str):
    mm = np.load(out / f"{name}.npy", mmap_mode="r")
    meta = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    return mm, meta


def img(mm, m):
    return np.asarray(mm[m["off"] : m["off"] + m["w"] * H]).reshape(H, m["w"])


# ------------------------------------------------------------------ augmentation
def aug_geom(g: np.ndarray, rng: random.Random) -> np.ndarray:
    from geom_transfer import apply
    P = dict(shear=rng.uniform(-12, 20), tx=rng.uniform(-0.5, 0.7), ty=rng.uniform(-0.5, 0.7),
             blur=rng.choice([0.0, 0.0, rng.uniform(0.2, 0.8)]), unsharp=rng.choice([0.0, 0.0, rng.uniform(0.2, 0.8)]),
             ink_level=rng.uniform(0, 100), paper_level=rng.uniform(205, 255), noise=rng.uniform(0, 4), paper_on=True)
    return apply(g, P, seed=rng.randrange(1 << 30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("out", type=Path)
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--n-muharaf", type=int, default=6000)
    ap.add_argument("--aug", default="base")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--pix", type=int, default=12800, help="max sum of line widths per batch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.cmd == "pack":
        import subprocess
        tmp = a.out / "_train.pkl"
        subprocess.run([sys.executable, str(Path(__file__).parent / "geom_prepare_replica_data.py"), str(tmp), "--n-muharaf", str(a.n_muharaf)], check=True)
        pack(pickle.load(open(tmp, "rb"))["rows"], a.out, "train")
        tmp.unlink()
        return
    if a.cmd == "packeval":
        pack(pickle.load(open(a.arg, "rb"))["rows"], a.out, "eval")
        return

    import torch
    from torch import nn
    from xai_train_aligner import CRNN, edit_distance, greedy, normalise

    torch.set_num_threads(a.threads)
    run = a.out / a.arg
    run.mkdir(exist_ok=True)

    def batch(gs, labels, stoi, train, rng):
        ws = [g.shape[1] for g in gs]
        wmax = int(math.ceil(max(ws) / 4) * 4)
        x = np.ones((len(gs), 1, H, wmax), np.float32)
        for i, g in enumerate(gs):
            x[i, 0, :, : ws[i]] = normalise(g)
        x = torch.from_numpy(x)
        lens = torch.tensor(ws)
        if train:  # the 3090 trainer's augmentation
            s = rng.uniform(0.9, 1.1)
            nw = max(8, int(round(wmax * s / 4)) * 4)
            x = torch.nn.functional.interpolate(x, size=(H, nw), mode="bilinear", align_corners=False)
            lens = torch.clamp((lens.float() * nw / wmax).round().long(), max=nw)
            c = torch.empty(len(gs), 1, 1, 1).uniform_(0.8, 1.2)
            b = torch.empty(len(gs), 1, 1, 1).uniform_(-0.15, 0.15)
            x = (x * c + b + 0.03 * torch.randn_like(x)).clamp(-1.2, 1.2)
            x = torch.roll(x, shifts=rng.randint(-3, 3), dims=3)
        tgt = [[stoi[ch] for ch in lab[::-1] if ch in stoi] for lab in labels]
        return x, lens, tgt

    if a.cmd == "train":
        mm, meta = load(a.out, "train")
        train = [m for m in meta if m["role"] == "train"]
        held = [m for m in meta if m["role"] == "heldout"]
        chars = sorted({ch for m in train for ch in m["label"]}, key=lambda c: (c == " ", c))
        stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        (run / "charset.json").write_text(json.dumps({"chars": chars}, ensure_ascii=False), encoding="utf-8")
        (run / "config.json").write_text(json.dumps({"width": a.width, "aug": a.aug, "epochs": a.epochs}), encoding="utf-8")
        pool = []
        for m in train:  # the small Maghrebi sets are repeated, as on the 3090
            pool += [m] * (3 if m["dataset"].startswith("rasam") else 1)
        rng = random.Random(0)
        torch.manual_seed(0)

        def make_batches():
            order = sorted(range(len(pool)), key=lambda i: pool[i]["w"] * rng.uniform(0.9, 1.1))
            bs, cur, curmax = [], [], 0
            for i in order:
                w = pool[i]["w"]
                if cur and (max(curmax, w) * (len(cur) + 1) > a.pix or len(cur) >= 16):
                    bs.append(cur); cur, curmax = [], 0
                cur.append(i); curmax = max(curmax, w)
            if cur:
                bs.append(cur)
            rng.shuffle(bs)
            return bs

        nb = len(make_batches())
        model = CRNN(len(chars) + 1, width=a.width)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (nb + 20), pct_start=0.15)
        ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        start = 0
        ck = run / "ckpt_last.pt"
        if ck.exists():
            st = torch.load(ck)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["epoch"]
            rng = random.Random(start)
            print("resumed at epoch", start, flush=True)
        print(f"train {len(train)} (pool {len(pool)})  heldout {len(held)}  classes {len(chars) + 1}  params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M  batches/epoch {nb}  aug {a.aug}", flush=True)
        for ep in range(start, a.epochs):
            t0, tot, n = time.time(), 0.0, 0
            model.train()
            for idx in make_batches():
                ms = [pool[i] for i in idx]
                gs = [img(mm, m) for m in ms]
                if a.aug == "geom":
                    gs = [aug_geom(g, rng) if rng.random() < 0.8 else g for g in gs]
                x, lens, tgt = batch(gs, [m["label"] for m in ms], stoi, True, rng)
                flat = torch.tensor([t for s in tgt for t in s])
                tl = torch.tensor([len(s) for s in tgt])
                loss = ctc(model(x).log_softmax(-1), flat, torch.clamp(lens // 4, min=1), tl)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                try:
                    sched.step()
                except ValueError:
                    pass
                tot += float(loss.detach()); n += 1
            msg = f"epoch {ep + 1}/{a.epochs}  loss {tot / n:.3f}  {time.time() - t0:.0f}s"
            if (ep + 1) % 5 == 0 or ep + 1 == a.epochs:
                model.eval()
                e = {}
                with torch.no_grad():
                    for k in range(0, len(held), 16):
                        ms = held[k : k + 16]
                        x, lens, tgt = batch([img(mm, m) for m in ms], [m["label"] for m in ms], stoi, False, rng)
                        for m, hyp, ref in zip(ms, greedy(model(x), lens), tgt):
                            d = e.setdefault(m["dataset"], [0, 0]); d[0] += edit_distance(hyp, ref); d[1] += max(1, len(m["label"]))
                msg += "  heldout CER " + json.dumps({k: round(v[0] / v[1], 4) for k, v in e.items()})
            print(msg, flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1}, ck)
        torch.save(model.state_dict(), run / "model_final.pt")
        print("done", flush=True)
        return

    if a.cmd == "eval":
        chars = json.loads((run / "charset.json").read_text(encoding="utf-8"))["chars"]
        cfg = json.loads((run / "config.json").read_text())
        stoi = {c: i + 1 for i, c in enumerate(chars)}
        model = CRNN(len(chars) + 1, width=cfg["width"])
        model.load_state_dict(torch.load(run / "model_final.pt"))
        model.eval()
        mm, meta = load(a.out, "eval")
        per = {}
        with torch.no_grad():
            for k in range(0, len(meta), 16):
                ms = meta[k : k + 16]
                x, lens, tgt = batch([img(mm, m) for m in ms], [m["label"] for m in ms], stoi, False, None)
                for m, hyp, ref in zip(ms, greedy(model(x), lens), tgt):
                    per.setdefault(m["dataset"], {})[m["id"]] = [edit_distance(hyp, ref), max(1, len(m["label"]))]
        (run / "eval_per_line.json").write_text(json.dumps(per), encoding="utf-8")
        for t, d in sorted(per.items()):
            print(f"{t:40s} {len(d):4d} lines  CER {100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()):5.1f} %", flush=True)


if __name__ == "__main__":
    main()
