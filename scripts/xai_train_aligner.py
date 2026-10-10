"""Train a CRNN-CTC line recogniser to serve as a forced aligner (X-B1). Runs on the 3090.

    python xai_train_aligner.py --data aligner_data.pkl --out aligner_run [--epochs 30]

Labels are fed in VISUAL order (the logical string reversed), so the left-to-right scan of the image
matches the label order. Checkpoints every epoch (resumable). Held-out CER is reported per dataset.
The panel lines are part of the training set on purpose: we align lines whose text we already know.
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
import torch
import torch.nn.functional as F
from torch import nn

H = 64


def conv(cin: int, cout: int) -> list:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """VGG-style CNN (height 64 -> 2, width / 4) + 3-layer BiLSTM + CTC head. T = W // 4."""

    def __init__(self, n_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            *conv(1, 64), nn.MaxPool2d(2, 2),
            *conv(64, 128), nn.MaxPool2d(2, 2),
            *conv(128, 256), *conv(256, 256), nn.MaxPool2d((2, 1), (2, 1)),
            *conv(256, 384), *conv(384, 384), nn.MaxPool2d((2, 1), (2, 1)),
            *conv(384, 512), nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.proj = nn.Linear(512 * 2, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=3, dropout=0.25, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B,1,64,W -> T,B,C logits
        f = self.cnn(x)  # B,512,2,T
        b, c, h, t = f.shape
        f = f.permute(3, 0, 1, 2).reshape(t, b, c * h)
        f, _ = self.rnn(F.relu(self.proj(f)))
        return self.fc(f)


def normalise(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, (2, 98))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-3), 0, 1) * 2 - 1


def make_batch(rows: list[dict], stoi: dict, train: bool, dev: str):
    ws = [r["img"].shape[1] for r in rows]
    wmax = int(math.ceil(max(ws) / 4) * 4)
    x = np.ones((len(rows), 1, H, wmax), dtype=np.float32)
    for i, r in enumerate(rows):
        x[i, 0, :, : ws[i]] = normalise(r["img"])
    x = torch.from_numpy(x).to(dev)
    lens = torch.tensor(ws, device=dev)
    if train:
        s = random.uniform(0.9, 1.1)  # one width scale per batch
        nw = max(8, int(round(wmax * s / 4)) * 4)
        x = F.interpolate(x, size=(H, nw), mode="bilinear", align_corners=False)
        lens = torch.clamp((lens.float() * nw / wmax).round().long(), max=nw)
        c = torch.empty(len(rows), 1, 1, 1, device=dev).uniform_(0.8, 1.2)
        b = torch.empty(len(rows), 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
        x = (x * c + b + 0.03 * torch.randn_like(x)).clamp(-1.2, 1.2)
        x = torch.roll(x, shifts=random.randint(-3, 3), dims=2)
    tgt = [[stoi[ch] for ch in r["label"][::-1] if ch in stoi] for r in rows]  # visual order
    return x, lens, tgt


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy(logits: torch.Tensor, lens: torch.Tensor) -> list[list[int]]:
    best = logits.argmax(-1).T.cpu().numpy()  # B,T
    out = []
    for row, n in zip(best, (lens // 4).cpu().numpy()):
        seq, prev = [], 0
        for k in row[: max(1, n)]:
            if k != prev and k != 0:
                seq.append(int(k))
            prev = k
        out.append(seq)
    return out


@torch.no_grad()
def evaluate(model, rows, stoi, dev) -> dict:
    model.eval()
    errs: dict = {}
    for k in range(0, len(rows), 32):
        chunk = rows[k : k + 32]
        x, lens, tgt = make_batch(chunk, stoi, False, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        for r, hyp, ref in zip(chunk, greedy(logits.float(), lens), tgt):
            e = errs.setdefault(r["dataset"], [0, 0])
            e[0] += edit_distance(hyp, ref)
            e[1] += max(1, len(r["label"]))  # unknown characters count as errors
    model.train()
    return {d: round(e[0] / e[1], 4) for d, e in errs.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    blob = pickle.load(open(a.data, "rb"))
    rows = blob["rows"]
    train = [r for r in rows if r["role"] in ("train", "panel")]
    held = [r for r in rows if r["role"] == "heldout"]
    chars = sorted({ch for r in train for ch in r["label"]}, key=lambda c: (c == " ", c))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    (a.out / "charset.json").write_text(json.dumps({"chars": chars}, ensure_ascii=False), encoding="utf-8")
    print(f"train {len(train)}  heldout {len(held)}  classes {len(chars) + 1}", flush=True)
    # the small Maghrebi set and the panel lines are repeated so the model fits them
    pool = []
    for r in train:
        pool += [r] * (4 if r["dataset"].startswith("rasam") or r["role"] == "panel" else 1)
    steps_per_epoch = math.ceil(len(pool) / a.bs)
    model = CRNN(len(chars) + 1).to(dev)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", "steps/epoch", steps_per_epoch, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler()
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    start, best = 0, 9.0
    ck = a.out / "ckpt_last.pt"
    if ck.exists():  # resume
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, best = st["epoch"], st.get("best", 9.0)
        print("resumed at epoch", start, flush=True)
    for ep in range(start, a.epochs):
        t0 = time.time()
        order = sorted(range(len(pool)), key=lambda i: pool[i]["img"].shape[1] * random.uniform(0.9, 1.1))
        batches = [order[k : k + a.bs] for k in range(0, len(order), a.bs)]
        random.shuffle(batches)
        tot = 0.0
        for bi, idx in enumerate(batches):
            x, lens, tgt = make_batch([pool[i] for i in idx], stoi, True, dev)
            flat = torch.tensor([t for s in tgt for t in s], device=dev)
            tl = torch.tensor([len(s) for s in tgt], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            lp = logits.float().log_softmax(-1)
            loss = ctc(lp, flat, torch.clamp(lens // 4, min=1), tl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss)
        msg = f"epoch {ep + 1}/{a.epochs}  loss {tot / len(batches):.3f}  {time.time() - t0:.0f}s"
        if (ep + 1) % 2 == 0 or ep + 1 == a.epochs:
            cer = evaluate(model, held, stoi, dev)
            msg += f"  heldout CER {cer}"
            mean = sum(cer.values()) / max(1, len(cer))
            if mean < best:
                best = mean
                torch.save(model.state_dict(), a.out / "model_best.pt")
        print(msg, flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "best": best}, ck)
    torch.save(model.state_dict(), a.out / "model_final.pt")
    (a.out / "heldout_cer.json").write_text(json.dumps(evaluate(model, held, stoi, dev)), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
