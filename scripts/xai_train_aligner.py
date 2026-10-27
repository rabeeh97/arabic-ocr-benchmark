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
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

H = 64


def conv(cin: int, cout: int) -> list:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """VGG-style CNN (height 64 -> 2, width / 4) + 3-layer BiLSTM + CTC head. T = W // 4."""

    def __init__(self, n_classes: int, hidden: int = 256, width: float = 1.0, height: int = 64) -> None:
        super().__init__()
        c = lambda v: max(8, int(round(v * width)))  # noqa: E731  width < 1 gives the small laptop replica
        hidden = c(hidden)
        self.cnn = nn.Sequential(
            *conv(1, c(64)), nn.MaxPool2d(2, 2),
            *conv(c(64), c(128)), nn.MaxPool2d(2, 2),
            *conv(c(128), c(256)), *conv(c(256), c(256)), nn.MaxPool2d((2, 1), (2, 1)),
            *conv(c(256), c(384)), *conv(c(384), c(384)), nn.MaxPool2d((2, 1), (2, 1)),
            *conv(c(384), c(512)), nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.proj = nn.Linear(c(512) * (height // 32), hidden)  # the CNN divides height by 32: 64 px -> 2 rows, 96 -> 3, 128 -> 4
        self.rnn = nn.LSTM(hidden, hidden, num_layers=3, dropout=0.25, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B,1,64,W -> T,B,C logits
        f = self.cnn(x)  # B,512,2,T
        b, c, h, t = f.shape
        f = f.permute(3, 0, 1, 2).reshape(t, b, c * h)
        f, _ = self.rnn(F.relu(self.proj(f)))
        return self.fc(f)


class CalfaCRNN(nn.Module):
    """Calfa's per-corpus CTC recogniser as described in the AMIDDA paper (HIP 2026, section 4 (vi)): greyscale line at 120 px, three conv blocks
    (64, 128, 256 channels, GroupNorm) that reduce height to H/4 (30 rows at 120 px) and width by 4, four BiLSTM layers of 512 units with 0.3 dropout,
    CTC head, no attention. Unlike CRNN above it keeps 30 rows of vertical detail (dots, vocalisation) for the recurrent layers. T = W // 4."""

    def __init__(self, n_classes: int, height: int = 120, hidden: int = 512, layers: int = 4) -> None:
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(16, o), nn.ReLU(inplace=True)]
        self.cnn = nn.Sequential(*block(1, 64), nn.MaxPool2d(2, 2), *block(64, 128), nn.MaxPool2d(2, 2), *block(128, 256))
        self.rnn = nn.LSTM(256 * (height // 4), hidden, num_layers=layers, dropout=0.3, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)  # B,256,H/4,T
        b, c, h, t = f.shape
        f, _ = self.rnn(f.permute(3, 0, 1, 2).reshape(t, b, c * h))
        return self.fc(self.drop(f))


def build_model(n_classes: int, cfg: dict) -> nn.Module:
    """The model a run's config.json describes (used by the trainer and the RASAM scoring scripts)."""
    if cfg.get("arch", "vgg") == "calfa":
        return CalfaCRNN(n_classes, height=int(cfg.get("height", 120)))
    return CRNN(n_classes, width=cfg.get("width", 1.0), height=int(cfg.get("height", 64)))


def normalise(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, (2, 98))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-3), 0, 1) * 2 - 1


GEOM_AUG = False  # set by --geom-aug


def geom_aug(x: torch.Tensor, lens: torch.Tensor):
    """Slant and stroke-width augmentation on the GPU (X-T2 follow-up). Shear -12..+20 degrees with p = 0.8 (positive leans the
    strokes right; Muharaf leans about +11, KHATT about -2), then fractional thickening or thinning of vertical and horizontal
    strokes separately (blend with a 3-wide min or max pool, up to half way), each with p = 0.5. Pads 12 px of paper on both
    sides first so sheared ends are not cut; returns the new lengths."""
    P = 12
    x = F.pad(x, (P, P, 0, 0), value=1.0)
    lens = lens + 2 * P
    b, _, h, w = x.shape
    dev = x.device
    deg = torch.empty(b, device=dev).uniform_(-12, 20) * (torch.rand(b, device=dev) < 0.8)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 1] = torch.tan(torch.deg2rad(deg)) * h / w
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x - 1, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + 1  # paper (+1) fills the gaps
    for k in ((1, 3), (3, 1)):  # (1, 3) widens vertical strokes, (3, 1) thickens horizontal ones
        pad = (k[0] // 2, k[1] // 2)
        a = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.5, 0.5) * (torch.rand(b, 1, 1, 1, device=dev) < 0.5)
        thick = -F.max_pool2d(-x, k, 1, pad)
        thin = F.max_pool2d(x, k, 1, pad)
        x = torch.where(a > 0, x + a * (thick - x), x - a * (thin - x))
    return x, lens


NEIGHBOR_P = 0.0  # set by --neighbor-aug: probability of pasting ink strips of other lines above and below (tall RASAM crops contain neighbours)
FADE_P = 0.0  # set by --fade-aug: probability of fading the ink towards the paper (faint RASAM-2 lines)
DONORS: list = []  # training images that provide the neighbour strips


def neighbor_aug(img: np.ndarray) -> np.ndarray:
    """Paste the bottom part of one random training line above and the top part of another below, then shrink back to height H, so the
    model learns to read the central line with foreign ink at the margins (as in the tall crops of the published RASAM test split)."""
    h, w = img.shape
    parts = []
    a, b = random.choice(DONORS), random.choice(DONORS)
    ha, hb = int(h * random.uniform(0.2, 0.45)), int(h * random.uniform(0.2, 0.45))
    for donor, take_bottom, hh in ((a, True, ha), (b, False, hb)):
        d = np.asarray(Image.fromarray(donor).resize((w, h)))
        parts.append(d[h - hh:] if take_bottom else d[:hh])
    comp = np.concatenate([parts[0], img, parts[1]], axis=0)
    nw = max(16, int(round(w * h / comp.shape[0])))
    return np.asarray(Image.fromarray(comp).resize((nw, h)), dtype=np.uint8)


def make_batch(rows: list[dict], stoi: dict, train: bool, dev: str):
    imgs = [neighbor_aug(r["img"]) if (train and NEIGHBOR_P and DONORS and random.random() < NEIGHBOR_P) else r["img"] for r in rows]
    ws = [im.shape[1] for im in imgs]
    wmax = int(math.ceil(max(ws) / 4) * 4)
    x = np.ones((len(rows), 1, H, wmax), dtype=np.float32)
    for i, im in enumerate(imgs):
        x[i, 0, :, : ws[i]] = normalise(im)
    x = torch.from_numpy(x).to(dev)
    lens = torch.tensor(ws, device=dev)
    if train and GEOM_AUG:
        x, lens = geom_aug(x, lens)
        wmax = x.shape[3]
    if train:
        s = random.uniform(0.9, 1.1)  # one width scale per batch
        nw = max(8, int(round(wmax * s / 4)) * 4)
        x = F.interpolate(x, size=(H, nw), mode="bilinear", align_corners=False)
        lens = torch.clamp((lens.float() * nw / wmax).round().long(), max=nw)
        c = torch.empty(len(rows), 1, 1, 1, device=dev).uniform_(0.8, 1.2)
        b = torch.empty(len(rows), 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
        x = (x * c + b + 0.03 * torch.randn_like(x)).clamp(-1.2, 1.2)
        if FADE_P:  # pull the ink towards the paper level (+1) on some lines: faint and low-contrast lines are RASAM-2's hardest
            sel = (torch.rand(len(rows), 1, 1, 1, device=dev) < FADE_P).float()
            f = torch.empty(len(rows), 1, 1, 1, device=dev).uniform_(0.2, 0.6) * sel
            x = x + f * (1.0 - x)
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
        with torch.autocast("cuda" if dev == "cuda" else "cpu", dtype=torch.float16, enabled=dev == "cuda"):
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
    ap.add_argument("--device", default="cuda", help="cuda (the 3090), mps or cpu (laptop replica; set PYTORCH_ENABLE_MPS_FALLBACK=1 for mps)")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--geom-aug", action="store_true", help="add slant and stroke-width augmentation (geom_aug)")
    ap.add_argument("--neighbor-aug", type=float, default=0.0, help="probability of pasting other lines' ink above/below a training line")
    ap.add_argument("--fade-aug", type=float, default=0.0, help="probability of fading a training line's ink towards the paper")
    ap.add_argument("--repeat-file", type=Path, help="JSON {\"extra\": {row_index: n}}: add n extra copies of these rows (indices into the pickle's rows)")
    ap.add_argument("--width", type=float, default=1.0, help="channel multiplier; 1.0 is the 3090 model")
    ap.add_argument("--max-train", type=int, default=0, help="use at most this many training lines (0 = all)")
    ap.add_argument("--train-sets", nargs="+", help="only rows whose dataset tag is listed (train and heldout alike); default all")
    ap.add_argument("--seed", type=int, help="seed python, numpy and torch (unset = unseeded, as the runs before 2026-10-13)")
    ap.add_argument("--ckpt-every", type=int, default=200, help="also checkpoint every N steps inside an epoch (resume loses at most N steps)")
    ap.add_argument("--ema", type=float, default=0.0, help="also keep an exponential moving average of the weights with this decay per step (0 = off); "
                    "evaluated next to the raw weights, saved as model_ema_best.pt / model_ema_final.pt")
    ap.add_argument("--arch", choices=["vgg", "calfa"], default="vgg", help="vgg: our CRNN (height / 32); calfa: the AMIDDA paper's per-corpus "
                    "CRNN (3 conv blocks, height / 4, 4 x BiLSTM 512); --width does not apply to calfa")
    ap.add_argument("--init", type=Path, help="start from these weights (a model_*.pt in a run dir) and keep that run's charset.json: fine-tuning, "
                    "e.g. specialising a shared model on one corpus with --train-sets")
    ap.add_argument("--keep-last", type=int, default=0, help="save the weights after each of the last K epochs (model_ep<N>.pt) for checkpoint averaging")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    if dev == "cuda" and os.environ.get("AOCR_CUDA_MEM_FRACTION"):
        # night 2 (2026-10-18): a CRNN shares the 3090 with a 4B VLM. Its true peak is about 4.2 GB (widest batch), but the caching
        # allocator grows to about 13 GB over an epoch; the cap makes it free cached blocks instead of growing past the share.
        torch.cuda.set_per_process_memory_fraction(float(os.environ["AOCR_CUDA_MEM_FRACTION"]))
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    global GEOM_AUG, NEIGHBOR_P, FADE_P
    GEOM_AUG = a.geom_aug
    NEIGHBOR_P, FADE_P = a.neighbor_aug, a.fade_aug
    blob = pickle.load(open(a.data, "rb"))
    global H
    H = int(blob.get("H", 64))  # line height is a property of the data pack (rasam_prepare_crnn_data.py --height)
    if H % (4 if a.arch == "calfa" else 32):
        raise SystemExit(f"line height {H} must be a multiple of {4 if a.arch == 'calfa' else 32} for --arch {a.arch}")
    rows = blob["rows"]
    extra = {int(k): int(v) for k, v in json.loads(a.repeat_file.read_text())["extra"].items()} if a.repeat_file else {}
    for j, r in enumerate(rows):
        r["_extra"] = extra.get(j, 0)
    if a.train_sets:  # RASAM arms (A1-A4): train and select on the named dataset tags only
        rows = [r for r in rows if r["dataset"] in a.train_sets]
    train = [r for r in rows if r["role"] in ("train", "panel")]
    DONORS[:] = [r["img"] for r in train]
    if a.max_train:
        random.Random(0).shuffle(train)
        train = train[: a.max_train]
    held = [r for r in rows if r["role"] == "heldout"]
    chars = sorted({ch for r in train for ch in r["label"]}, key=lambda c: (c == " ", c))
    if a.init:  # the output layer must keep the parent model's classes
        chars = json.loads((a.init.parent / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    (a.out / "charset.json").write_text(json.dumps({"chars": chars}, ensure_ascii=False), encoding="utf-8")
    print(f"train {len(train)}  heldout {len(held)}  classes {len(chars) + 1}", flush=True)
    # the small Maghrebi set and the panel lines are repeated so the model fits them
    pool = []
    for r in train:
        pool += [r] * ((4 if r["dataset"].startswith("rasam") or r["role"] == "panel" else 1) + r["_extra"])
    steps_per_epoch = math.ceil(len(pool) / a.bs)
    split = blob.get("split", "aligner")
    (a.out / "config.json").write_text(json.dumps({"width": a.width, "height": H, "geom_aug": a.geom_aug, "neighbor_aug": a.neighbor_aug, "fade_aug": a.fade_aug, "repeat_file": str(a.repeat_file) if a.repeat_file else None, "extra_rows": len(extra),
                                                   "seed": a.seed, "split": split, "data": str(a.data), "train_lines": len(train), "epochs": a.epochs,
                                                   "ema": a.ema, "keep_last": a.keep_last, "init": str(a.init) if a.init else None, "arch": a.arch,
                                                   "train_sets": a.train_sets, "lr": a.lr}), encoding="utf-8")
    model = build_model(len(chars) + 1, {"arch": a.arch, "width": a.width, "height": H}).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
        print("initialised from", a.init, flush=True)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", "steps/epoch", steps_per_epoch, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    # EMA of the weights (BatchNorm buffers averaged too, so the copy can be evaluated as is)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema), use_buffers=True) if a.ema else None
    start, best, best_ema = 0, 9.0, 9.0
    resume_batches, resume_bi, resume_tot = None, 0, 0.0
    ck = a.out / "ckpt_last.pt"
    if ck.exists():  # resume: at an epoch boundary, or inside an epoch with the same batch order
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        if ema is not None and "ema" in st:
            ema.load_state_dict(st["ema"])
        start, best, best_ema = st["epoch"], st.get("best", 9.0), st.get("best_ema", 9.0)
        if st.get("batches") is not None:
            resume_batches, resume_bi, resume_tot = st["batches"], st["bi"], st.get("tot", 0.0)
        print("resumed at epoch", start, "step", resume_bi, flush=True)

    def save_ckpt(epoch, batches=None, bi=0, tot=0.0):
        tmp = ck.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "best_ema": best_ema, "batches": batches, "bi": bi, "tot": tot,
                    **({"ema": ema.state_dict()} if ema is not None else {})}, tmp)
        tmp.replace(ck)  # atomic, so a crash during the save keeps the previous checkpoint

    for ep in range(start, a.epochs):
        t0 = time.time()
        if resume_batches is not None:
            batches, first, tot = resume_batches, resume_bi, resume_tot
            resume_batches = None
        else:
            order = sorted(range(len(pool)), key=lambda i: pool[i]["img"].shape[1] * random.uniform(0.9, 1.1))
            batches = [order[k : k + a.bs] for k in range(0, len(order), a.bs)]
            random.shuffle(batches)
            first, tot = 0, 0.0
        for bi in range(first, len(batches)):
            idx = batches[bi]
            x, lens, tgt = make_batch([pool[i] for i in idx], stoi, True, dev)
            flat = torch.tensor([t for s in tgt for t in s], device=dev)
            tl = torch.tensor([len(s) for s in tgt], device=dev)
            with torch.autocast("cuda" if dev == "cuda" else "cpu", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x)
            lp = logits.float().log_softmax(-1)
            loss = ctc(lp, flat, torch.clamp(lens // 4, min=1), tl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += float(loss)
            if a.ckpt_every and (bi + 1) % a.ckpt_every == 0 and bi + 1 < len(batches):
                save_ckpt(ep, batches, bi + 1, tot)
        msg = f"epoch {ep + 1}/{a.epochs}  loss {tot / len(batches):.3f}  {time.time() - t0:.0f}s"
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            cer = evaluate(model, held, stoi, dev)
            msg += f"  heldout CER {cer}"
            mean = sum(cer.values()) / max(1, len(cer))
            if mean < best:
                best = mean
                torch.save(model.state_dict(), a.out / "model_best.pt")
            if ema is not None:
                ecer = evaluate(ema.module, held, stoi, dev)
                msg += f"  ema CER {ecer}"
                emean = sum(ecer.values()) / max(1, len(ecer))
                if emean < best_ema:
                    best_ema = emean
                    torch.save(ema.module.state_dict(), a.out / "model_ema_best.pt")
        if a.keep_last and ep + 1 > a.epochs - a.keep_last:
            torch.save(model.state_dict(), a.out / f"model_ep{ep + 1}.pt")
        print(msg, flush=True)
        save_ckpt(ep + 1)
    torch.save(model.state_dict(), a.out / "model_final.pt")
    final = evaluate(model, held, stoi, dev)
    if ema is not None:
        torch.save(ema.module.state_dict(), a.out / "model_ema_final.pt")
        final = {"raw": final, "ema": evaluate(ema.module, held, stoi, dev)}
    (a.out / "heldout_cer.json").write_text(json.dumps(final), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
