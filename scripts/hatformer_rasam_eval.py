"""Run HATFormer trained on AraMS-28k (Archatext/hatformer-arams28k, 334 M, CC BY-NC-SA) on the official HF test lines of RASAM-1, RASAM-2 and TariMa,
on the Mac CPU. Preprocessing and decoding are the model card's, verbatim (64 px height, RTL flip, 384x384 black canvas with wrapped rows,
mean/std 0.5, beam 3, length_penalty 0, max_new_tokens 64). The model never saw RASAM or TariMa (AraMS-28k train: 9 other manuscripts).

    PYTHONPATH=src python scripts/hatformer_rasam_eval.py [--limit N]

Writes results/rasam/external/hatformer_<set>__test.json ({"lines": [{"ref", "hyp"}]}) and prints micro CER (project metric, raw / light /
diacritics-insensitive). Resumable: lines already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import PreTrainedTokenizerFast, VisionEncoderDecoderModel

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

HUB = "/Volumes/PortableSSD/huggingface/hub"
SETS = {"r1": "calfa-ai--RASAM-1", "r2": "calfa-ai--RASAM-2", "tm": "calfa-ai--tarima"}
OUT = Path("results/rasam/external")


def hatformer_canvas(img, height=64, canvas=384):
    img = img.convert("RGB")
    w, h = img.size
    resized = img.resize((max(1, int(height * w / h)), height)).transpose(Image.FLIP_LEFT_RIGHT)
    out = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    if resized.width <= canvas:
        out.paste(resized, (0, 0))
    else:
        for i in range((resized.width + canvas - 1) // canvas):
            seg = resized.crop((i * canvas, 0, min((i + 1) * canvas, resized.width), height))
            out.paste(seg, (0, i * height))
    return out


def pixel_values(img):
    arr = np.asarray(hatformer_canvas(img), dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def test_rows(repo):
    f = [x for x in glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/test-*.parquet") if "/._" not in x]
    rows = []
    for x in sorted(f):
        d = pq.read_table(x, columns=["image", "transcription"]).to_pydict()
        rows += [(d["image"][i]["bytes"], d["transcription"][i]) for i in range(len(d["transcription"]))]
    return rows


def cer(pairs, profile):
    e = n = 0
    for ref, hyp in pairs:
        c = char_edit_counts(normalize(ref, profile).strip(), normalize(hyp, profile).strip())
        e += c["sub"] + c["del"] + c["ins"]
        n += c["ref_len"]
    return round(e / max(1, n) * 100, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sets", nargs="+", default=["r1", "r2", "tm"])
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(4)
    local = Path("/Volumes/PortableSSD/models/hatformer-arams28k")  # fetched with curl (resumable) when the hub download stalls
    ckpt = str(local) if (local / "model.safetensors").exists() and (local / "model.safetensors").stat().st_size > 1_300_000_000 \
        else snapshot_download("Archatext/hatformer-arams28k")
    model = VisionEncoderDecoderModel.from_pretrained(ckpt).eval()
    tok = PreTrainedTokenizerFast.from_pretrained(ckpt)
    tok.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>", "cls_token": "<s>", "bos_token": "<s>"})
    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.sets:
        rows = test_rows(SETS[s])[: a.limit or None]
        fo = OUT / f"hatformer_{s}__test.json"
        done = json.loads(fo.read_text())["lines"] if fo.exists() else []
        t0 = time.time()
        for k in range(len(done), len(rows), a.bs):
            chunk = rows[k : k + a.bs]
            pv = torch.stack([pixel_values(Image.open(io.BytesIO(b))) for b, _ in chunk])
            with torch.no_grad():
                ids = model.generate(pixel_values=pv, num_beams=3, length_penalty=0, max_new_tokens=64)
            hyps = tok.batch_decode(ids, skip_special_tokens=True)
            done += [{"ref": r, "hyp": h} for (_, r), h in zip(chunk, hyps)]
            fo.write_text(json.dumps({"model": "Archatext/hatformer-arams28k", "set": s, "lines": done}, ensure_ascii=False))
            print(f"{s} {len(done)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        pairs = [(l["ref"], l["hyp"]) for l in done]
        print(f"HATFORMER {s} test n={len(pairs)} micro CER raw {cer(pairs, 'raw')} light {cer(pairs, 'light')} "
              f"diac_insensitive {cer(pairs, 'diac_insensitive')}", flush=True)


if __name__ == "__main__":
    main()
