"""CPU CTC training loop with step checkpoints and resume."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from arabic_ocr_bench.charctc.charset import Charset, build_charset
from arabic_ocr_bench.charctc.config import load_charctc_config
from arabic_ocr_bench.charctc.decode import blank_fraction, greedy_from_logits, penalize_blank_logits
from arabic_ocr_bench.charctc.prepare import load_label_rows
from arabic_ocr_bench.charctc.preprocess import line_to_array, time_steps
from arabic_ocr_bench.metrics import cer
from arabic_ocr_bench.training.checkpoints import latest_checkpoint


class LineCacheDataset:
    def __init__(
        self,
        cache: Path,
        split: str,
        charset: Charset,
        *,
        height: int,
        max_width: int,
    ) -> None:
        self.cache = Path(cache)
        self.rows = load_label_rows(self.cache, split)
        self.charset = charset
        self.height = height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = Image.open(self.cache / row["path"])
        arr = line_to_array(img, height=self.height, max_width=self.max_width)
        ids = self.charset.encode(row["text"])
        if not ids:
            ids = [self.charset.stoi(" ")]
        t = time_steps(arr.shape[-1])
        if len(ids) > t:
            ids = ids[:t]
        return {
            "image": arr,
            "targets": np.asarray(ids, dtype=np.int32),
            "id": row["id"],
            "text": row["text"],
        }


def _collate(batch: list[dict]):
    import torch

    max_w = max(item["image"].shape[-1] for item in batch)
    h = batch[0]["image"].shape[-2]
    images = np.zeros((len(batch), 1, h, max_w), dtype=np.float32)
    targets = []
    target_lens = []
    input_lens = []
    texts = []
    ids = []
    for i, item in enumerate(batch):
        w = item["image"].shape[-1]
        images[i, :, :, :w] = item["image"]
        targets.append(item["targets"])
        target_lens.append(len(item["targets"]))
        input_lens.append(time_steps(w))
        texts.append(item["text"])
        ids.append(item["id"])
    return {
        "images": torch.from_numpy(images),
        "targets": torch.from_numpy(np.concatenate(targets)).long(),
        "target_lens": torch.tensor(target_lens, dtype=torch.long),
        "input_lens": torch.tensor(input_lens, dtype=torch.long),
        "texts": texts,
        "ids": ids,
    }


def _torch_load(path: Path, device, *, weights_only: bool = True):
    import torch

    kwargs = {"map_location": device}
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _rotate_ckpts(out: Path, keep: int) -> None:
    ckpts = []
    for child in out.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-", 1)[1])
            except ValueError:
                continue
            ckpts.append((step, child))
    ckpts.sort()
    for _, path in ckpts[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def _save_ckpt(
    out: Path,
    *,
    step: int,
    epoch: int,
    model,
    optimizer,
    charset: Charset,
    extra: dict,
    keep: int,
) -> Path:
    import torch

    dest = out / f"checkpoint-{step}"
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest / "model.pt")
    torch.save(optimizer.state_dict(), dest / "optimizer.pt")
    charset.save(dest / "charset.json")
    state = {"step": step, "epoch": epoch, **extra}
    (dest / "train_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    charset.save(out / "charset.json")
    (out / "train_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    _rotate_ckpts(out, keep)
    return dest


def _load_latest(out: Path, model, optimizer, device):
    ckpt = latest_checkpoint(out)
    if not ckpt:
        return 0, 0, {}
    path = Path(ckpt)
    model.load_state_dict(_torch_load(path / "model.pt", device))
    opt_path = path / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(_torch_load(opt_path, device, weights_only=False))
    extra = {}
    state_path = path / "train_state.json"
    if state_path.exists():
        extra = json.loads(state_path.read_text(encoding="utf-8"))
    step = int(extra.get("step", path.name.split("-", 1)[1]))
    epoch = int(extra.get("epoch", 0))
    print(f"[charctc] resume from {path} (step {step}, epoch {epoch})", flush=True)
    return step, epoch, extra


def _predict_batch(model, images, input_lens, charset, *, blank_penalty: float) -> list[str]:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(images)
        out = greedy_from_logits(logits, input_lens, charset, blank_penalty=blank_penalty)
    model.train()
    return out


def _format_samples(refs: list[str], hyps: list[str], n: int = 2) -> str:
    parts = []
    for ref, hyp in list(zip(refs, hyps))[:n]:
        parts.append(f"gt={ref!r} hyp={hyp!r}")
    return " | ".join(parts)


def _val_cer(model, loader, charset, device, max_batches: int, *, blank_penalty: float) -> float:
    scores = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = batch["images"].to(device)
        hyps = _predict_batch(
            model, images, batch["input_lens"], charset, blank_penalty=blank_penalty
        )
        for hyp, ref in zip(hyps, batch["texts"]):
            scores.append(cer(ref, hyp))
    return float(sum(scores) / len(scores)) if scores else 1.0


def train(cfg_path: Path, cache: Path, out: Path, *, device_name: str = "cpu") -> Path:
    import torch
    from torch.nn.functional import ctc_loss
    from torch.utils.data import DataLoader

    cfg = load_charctc_config(cfg_path)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    print(f"[charctc] device={device} torch={torch.__version__}", flush=True)

    charset_path = out / "charset.json"
    if charset_path.exists():
        charset = Charset.load(charset_path)
        print(f"[charctc] loaded charset {charset.n_classes} classes", flush=True)
    else:
        texts = [r["text"] for r in load_label_rows(cache)]
        charset = build_charset(texts)
        charset.save(charset_path)
        print(f"[charctc] built charset {charset.n_classes} classes", flush=True)

    spec = cfg.train
    train_ds = LineCacheDataset(
        cache, "train", charset, height=spec.img_height, max_width=spec.max_width
    )
    val_ds = LineCacheDataset(
        cache, "val", charset, height=spec.img_height, max_width=spec.max_width
    )
    print(f"[charctc] train={len(train_ds)} val={len(val_ds)}", flush=True)
    if len(train_ds) == 0:
        raise SystemExit("train: empty training cache")

    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds or train_ds,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    from arabic_ocr_bench.charctc.model import CharCRNN

    model = CharCRNN(charset.n_classes, hidden=spec.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr)
    step = 0
    epoch0 = 0
    best = 1.0
    if spec.resume:
        step, epoch0, extra = _load_latest(out, model, optimizer, device)
        best = float(extra.get("best_val_cer", 1.0))

    n_epochs = int(max(1, round(spec.epochs)))
    blank_penalty = float(spec.blank_penalty)
    t0 = time.monotonic()
    running = 0.0
    running_n = 0
    running_blank = 0.0
    done_marker = out / "train_complete.json"
    if done_marker.exists() and epoch0 < n_epochs:
        done_marker.unlink()
        print(
            f"[charctc] continuing past complete marker: epoch {epoch0} -> {n_epochs}, "
            f"blank_penalty={blank_penalty}",
            flush=True,
        )
    print(f"[charctc] blank_penalty={blank_penalty}", flush=True)
    model.train()
    for epoch in range(epoch0, n_epochs):
        print(f"[charctc] epoch {epoch + 1}/{n_epochs} starting at step {step}", flush=True)
        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            input_lens = batch["input_lens"].to(device)
            target_lens = batch["target_lens"].to(device)
            logits = penalize_blank_logits(model(images), blank_penalty)
            log_probs = logits.log_softmax(2)
            loss = ctc_loss(
                log_probs,
                targets,
                input_lens,
                target_lens,
                blank=0,
                zero_infinity=True,
            )
            if torch.isnan(loss):
                print(f"[charctc] skip nan loss at step {step}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            running += float(loss.item())
            running_n += 1
            running_blank += blank_fraction(logits.detach())
            if step % spec.log_every == 0:
                elapsed = time.monotonic() - t0
                avg = running / max(running_n, 1)
                bf = running_blank / max(running_n, 1)
                with torch.no_grad():
                    hyps = greedy_from_logits(
                        logits.detach(), input_lens, charset, blank_penalty=0.0
                    )
                samples = _format_samples(batch["texts"], hyps)
                print(
                    f"[charctc] step {step} epoch {epoch} loss={avg:.4f} "
                    f"blank_frac={bf:.3f} elapsed={elapsed:.0f}s | {samples}",
                    flush=True,
                )
                running = 0.0
                running_n = 0
                running_blank = 0.0
            if step % spec.save_steps == 0:
                val = _val_cer(
                    model,
                    val_loader,
                    charset,
                    device,
                    max_batches=max(1, spec.val_samples // spec.batch_size),
                    blank_penalty=blank_penalty,
                )
                best = min(best, val)
                dest = _save_ckpt(
                    out,
                    step=step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    charset=charset,
                    extra={
                        "loss": float(loss.item()),
                        "val_cer": val,
                        "best_val_cer": best,
                        "n_train": len(train_ds),
                        "n_classes": charset.n_classes,
                        "device": str(device),
                        "blank_penalty": blank_penalty,
                    },
                    keep=spec.save_total_limit,
                )
                print(
                    f"[charctc] saved {dest.name} val_cer={val:.4f} best={best:.4f}",
                    flush=True,
                )

    val = _val_cer(
        model,
        val_loader,
        charset,
        device,
        max_batches=max(1, spec.val_samples // spec.batch_size),
        blank_penalty=blank_penalty,
    )
    best = min(best, val)
    dest = _save_ckpt(
        out,
        step=step,
        epoch=n_epochs,
        model=model,
        optimizer=optimizer,
        charset=charset,
        extra={
            "loss": None,
            "val_cer": val,
            "best_val_cer": best,
            "n_train": len(train_ds),
            "n_classes": charset.n_classes,
            "device": str(device),
            "blank_penalty": blank_penalty,
            "complete": True,
        },
        keep=spec.save_total_limit,
    )
    payload = {
        "complete": True,
        "step": step,
        "epochs": n_epochs,
        "val_cer": val,
        "best_val_cer": best,
        "blank_penalty": blank_penalty,
        "checkpoint": str(dest),
    }
    done_marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[charctc] train complete step={step} val_cer={val:.4f} -> {dest}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    train(Path(args.config), Path(args.cache), Path(args.out), device_name=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
