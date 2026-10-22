"""RS2: re-score a published Kraken recognition model on RASAM line images (recognition only, whole image = one line box).

Run from the kraken venv (not the project venv):
    PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/rasam_kraken_score.py \
        --model /Volumes/PortableSSD/models/rasam-replication/phoenix/model.mlmodel --name phoenix \
        --dataset RASAM-1 --split test [--limit 50]

Writes results/rasam/replication/<name>__<dataset>__<split>.json with per-line readings (kraken logical order) and
corpus scores under the raw / light / diac_insensitive profiles. Also logs the CER of the reversed reading on the first lines,
as a text-direction check (CORRECTIONS C1, C5).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from kraken import rpred
from kraken.containers import BBoxLine, Segmentation
from kraken.lib import models
from PIL import Image
from skimage.filters import threshold_otsu

from arabic_ocr_bench.metrics import aggregate, score_pair

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
PROFILES = ("raw", "light", "diac_insensitive")


def load_rows(dataset: str, split: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(f"{ROOT}{dataset}/snapshots/*/data/{split}-*.parquet")):
        rows.extend(pq.read_table(f).to_pylist())
    return rows


def prep(img: Image.Image, variant: str) -> Image.Image:
    """Input variants (the published models were trained on kraken-extracted lines, not on these HF crops)."""
    if variant == "raw":
        return img
    a = np.asarray(img.convert("L")).astype(np.float32)
    if variant == "gray":
        return Image.fromarray(a.astype(np.uint8)).convert("RGB")
    mask = a < 12  # black fill outside the rotated line polygon
    fill = float(np.median(a[~mask])) if (~mask).any() else 255.0
    a = np.where(mask, fill, a)
    if variant == "fill":
        return Image.fromarray(a.astype(np.uint8)).convert("RGB")
    if variant == "bin":  # local-contrast stretch then Otsu
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        b = np.clip((a - lo) / max(hi - lo, 1), 0, 1)
        t = threshold_otsu(b)
        return Image.fromarray(((b > t) * 255).astype(np.uint8)).convert("RGB")
    raise ValueError(variant)


def read_line(net, img: Image.Image, direction: str, name: str) -> str:
    w, h = img.size
    seg = Segmentation(type="bbox", imagename=name, text_direction=direction, script_detection=False,
                       lines=[BBoxLine(id="l0", bbox=(0, 0, w, h), text_direction=direction)])
    recs = list(rpred.rpred(net, img, seg))
    return recs[0].prediction if recs else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True, choices=["RASAM-1", "RASAM-2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--direction", default="horizontal-rl")
    ap.add_argument("--variant", default="raw", choices=["raw", "gray", "fill", "bin"])
    ap.add_argument("--max-height", type=int, default=0, help="only lines at most this tall (0 = all); diagnostic")
    a = ap.parse_args()

    net = models.load_any(a.model, device="cpu")
    sha = hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
    rows = load_rows(a.dataset, a.split)
    if a.max_height:
        rows = [r for r in rows if Image.open(io.BytesIO(r["image"]["bytes"])).size[1] <= a.max_height]
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.time()
    per_line, scores, rev_scores = [], [], []
    for i, r in enumerate(rows):
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        ref = unicodedata.normalize("NFC", r["transcription"])
        hyp = unicodedata.normalize("NFC", read_line(net, prep(img, a.variant), a.direction, f"{a.dataset}/{a.split}/{i:06d}"))
        scores.append(score_pair(ref, hyp, PROFILES))
        if i < 50:
            rev_scores.append(score_pair(ref, hyp[::-1], PROFILES))
        per_line.append({"id": f"{a.dataset}/{a.split}/{i:06d}", "manuscript": r["manuscript_name"], "page": r["image_name"],
                         "region": r["region_type"], "ref": ref, "hyp": hyp, "w": img.size[0], "h": img.size[1]})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    agg = aggregate(scores, PROFILES)
    rev = aggregate(rev_scores, PROFILES) if rev_scores else None
    out = {
        "model": a.name, "model_path": a.model, "model_sha256": sha, "kraken": version("kraken"),
        "dataset": a.dataset, "split": a.split, "variant": a.variant, "max_height": a.max_height, "n": len(rows), "direction": a.direction,
        "micro_cer": agg["micro_cer"], "macro_cer": agg["cer_macro"], "micro_wer": agg["micro_wer"],
        "reversed_check_first50_micro_cer": rev["micro_cer"] if rev else None,
        "seconds": round(time.time() - t0, 1), "lines": per_line,
    }
    d = Path("results/rasam/replication")
    d.mkdir(parents=True, exist_ok=True)
    suffix = (f"__{a.variant}" if a.variant != "raw" else "") + (f"__h{a.max_height}" if a.max_height else "")
    p = d / f"{a.name}__{a.dataset}__{a.split}{suffix}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(a.name, a.dataset, a.split, "n =", len(rows))
    print("  micro CER", {k: round(v, 4) for k, v in agg["micro_cer"].items()})
    if rev:
        print("  reversed first-50 micro CER", {k: round(v, 4) for k, v in rev["micro_cer"].items()})
    print("  wrote", p)


if __name__ == "__main__":
    main()
