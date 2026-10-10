"""Zero-shot probe: can LocateAnything-3B find Arabic handwriting at word / PAW / letter / dot level?

Throwaway research spike (runs on the 3090 box in its own venv: transformers 4.57.1, torch cu124).

    python xai_probe_locateanything.py --lines DIR [--lines DIR2] --worker EAGLE/Embodied --out OUT [--limit N]

For each line image (manifest.json: file, gt) it asks, with the model's own "text grounding" prompt:
  det   detect all text (what units does it return?)
  W1/W2/W3  the first, a middle and the last word of the ground truth
  P     one PAW (connected piece of a word)
  L     one letter (lam or kha if present)
  D     the dots of one dotted letter (ba, ta, tha, nun)
and draws the returned boxes. Two automatic proxies, for a go/no-go before any fine-tuning:
  order_ok    the three word boxes run right to left in reading order
  width_err   mean |box width / image width - word chars / line chars| over the word boxes
The model loads pinned to an exact Hugging Face revision because its code runs with trust_remote_code.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image, ImageDraw

NONJOIN = set("اأإآدذرزوؤءةى")
HARAKAT = re.compile("[ً-ْٰـ]")
DOTTED = "بتثن"
COLORS = {"det": (120, 120, 120), "W1": (230, 60, 60), "W2": (40, 130, 230), "W3": (30, 170, 90),
          "P": (240, 150, 20), "L": (170, 60, 200), "D": (0, 170, 190)}


def strip(s: str) -> str:
    return HARAKAT.sub("", unicodedata.normalize("NFC", s))


def paws(word: str) -> list[str]:
    out, cur = [], ""
    for ch in strip(word):
        cur += ch
        if ch in NONJOIN:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def load_worker(worker_dir: Path):
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    sys.path.insert(0, str(worker_dir))
    from locateanything_worker import LocateAnythingWorker

    repo = "nvidia/LocateAnything-3B"
    sha = HfApi().model_info(repo).sha
    print("revision", sha, flush=True)
    w = LocateAnythingWorker.__new__(LocateAnythingWorker)  # same object as __init__, but with a pinned revision
    w.device, w.dtype, w.use_batch_runtime, w.scheduler, w.group_size = "cuda", torch.bfloat16, False, "pipeline", 0
    w.tokenizer = AutoTokenizer.from_pretrained(repo, revision=sha, trust_remote_code=True)
    w.processor = AutoProcessor.from_pretrained(repo, revision=sha, trust_remote_code=True)
    w.model = AutoModel.from_pretrained(repo, revision=sha, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    return w, sha, LocateAnythingWorker


def ask(w, fn, *args):
    t0 = time.time()
    try:
        ans = fn(*args, temperature=0.0, verbose=False)["answer"]
    except Exception as e:  # noqa: BLE001
        ans = f"ERROR {type(e).__name__}: {e}"
    return ans, time.time() - t0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=Path, action="append", required=True)
    ap.add_argument("--worker", type=Path, required=True, help="Eagle/Embodied (holds locateanything_worker.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="lines per --lines dir")
    ap.add_argument("--max-width", type=int, default=0, help="downscale wider lines first (0 = keep size); padding a full-size line OOMs")
    ap.add_argument("--pad", action="store_true", help="pad each line to a paper-coloured square before querying; boxes are mapped back")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w, sha, LA = load_worker(a.worker)
    results = []
    for d in a.lines:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if a.limit:
            man = man[: a.limit]
        for item in man:
            img = Image.open(d / item["file"]).convert("RGB")
            W, H = img.size
            sc = min(1.0, a.max_width / W) if a.max_width else 1.0
            qimg = img.resize((round(W * sc), max(1, round(H * sc))), Image.BICUBIC) if sc < 1 else img
            ox = oy = 0
            if a.pad:  # a 6.5:1 line is far from the aspect ratios a grounding model sees; give it a square canvas
                S = max(qimg.size)
                canvas = Image.new("RGB", (S, S), img.resize((1, 1), Image.BOX).getpixel((0, 0)))
                oy = (S - qimg.height) // 2
                canvas.paste(qimg, (0, oy))
                qimg = canvas
            QW, QH = qimg.size

            def boxes_of(ans, QW=QW, QH=QH, ox=ox, oy=oy, sc=sc, W=W, H=H):
                out = []
                for b in LA.parse_boxes(ans, QW, QH):
                    x1, y1 = min(W, max(0, (b["x1"] - ox) / sc)), min(H, max(0, (b["y1"] - oy) / sc))
                    x2, y2 = min(W, max(0, (b["x2"] - ox) / sc)), min(H, max(0, (b["y2"] - oy) / sc))
                    out.append({"x1": x1, "y1": y1, "x2": max(x1, x2), "y2": max(y1, y2)})  # zero area = box fell in the padding
                return out
            gt = strip(item["gt"]).strip()
            words = [x for x in gt.split() if len(x) >= 2]
            if len(words) < 3:
                continue
            pick = {"W1": words[0], "W2": words[len(words) // 2], "W3": words[-1]}
            if len({*pick.values()}) < 3:
                continue
            q = {}
            for wd in words:
                ps = [p for p in paws(wd) if len(p) >= 2]
                if len(ps) >= 2:
                    q["P"] = ps[1]
                    break
            for ch in "لخ":
                if ch in gt:
                    q["L"] = ch
                    break
            for ch in DOTTED:
                if ch in gt:
                    q["D"] = ch
                    break
            rec = {"id": item["id"], "dataset": item["dataset"], "gt": item["gt"], "size": [W, H], "q": {}}
            ans, dt = ask(w, w.detect_text, qimg)
            rec["q"]["det"] = {"text": None, "answer": ans, "sec": dt, "boxes": boxes_of(ans)}
            for k, text in pick.items():
                ans, dt = ask(w, w.ground_text, qimg,text)
                rec["q"][k] = {"text": text, "answer": ans, "sec": dt, "boxes": boxes_of(ans)}
            if "P" in q:
                ans, dt = ask(w, w.ground_text, qimg,q["P"])
                rec["q"]["P"] = {"text": q["P"], "answer": ans, "sec": dt, "boxes": boxes_of(ans)}
            if "L" in q:
                ans, dt = ask(w, w.ground_text, qimg,q["L"])
                rec["q"]["L"] = {"text": q["L"], "answer": ans, "sec": dt, "boxes": boxes_of(ans)}
            if "D" in q:
                ans, dt = ask(w, w.ground_single, qimg,f"the dots of the letter {q['D']}")
                rec["q"]["D"] = {"text": q["D"], "answer": ans, "sec": dt, "boxes": boxes_of(ans)}
            # proxies
            cx = []
            for k in ("W1", "W2", "W3"):
                b = rec["q"][k]["boxes"]
                cx.append((b[0]["x1"] + b[0]["x2"]) / 2 if b else None)
            rec["order_ok"] = None if None in cx else bool(cx[0] > cx[1] > cx[2])
            errs = []
            for k in ("W1", "W2", "W3"):
                b = rec["q"][k]["boxes"]
                if b:
                    errs.append(abs((b[0]["x2"] - b[0]["x1"]) / W - len(rec["q"][k]["text"]) / max(1, len(gt.replace(" ", "")) )))
            rec["width_err"] = sum(errs) / len(errs) if errs else None
            rec["n_boxes"] = {k: len(v["boxes"]) for k, v in rec["q"].items()}
            # a box covering >= 95 % of the image in both directions is the model's "whole image" non-answer
            rec["degenerate"] = {k: bool(v["boxes"]) and (v["boxes"][0]["x2"] - v["boxes"][0]["x1"]) >= 0.95 * W
                                 and (v["boxes"][0]["y2"] - v["boxes"][0]["y1"]) >= 0.95 * H for k, v in rec["q"].items()}
            # overlay
            ov = img.copy()
            dr = ImageDraw.Draw(ov)
            for k, v in rec["q"].items():
                for i, b in enumerate(v["boxes"][:6]):
                    dr.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=COLORS[k], width=max(2, H // 90))
                    dr.text((b["x1"] + 2, b["y1"] + 2 + 11 * (i % 3)), k, fill=COLORS[k])
            fn = f"{len(results):03d}.png"
            ov.save(a.out / fn)
            rec["overlay"] = fn
            results.append(rec)
            print(f"[{len(results)}] {item['id']} order_ok={rec['order_ok']} width_err={rec['width_err']} boxes={rec['n_boxes']}", flush=True)
    (a.out / "results.json").write_text(json.dumps({"revision": sha, "lines": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(results)
    errs = sum(1 for r in results for v in r["q"].values() if v["answer"].startswith("ERROR"))
    summ = {"lines": n, "padded": a.pad, "max_width": a.max_width, "query_errors": errs}
    for k in ("det", "W1", "W2", "W3", "P", "L", "D"):
        rs = [r for r in results if k in r["q"]]
        summ[k] = {"n": len(rs), "no_box": sum(1 for r in rs if not r["q"][k]["boxes"]), "whole_image": sum(1 for r in rs if r["degenerate"][k]),
                   "outside_line": sum(1 for r in rs if r["q"][k]["boxes"] and (r["q"][k]["boxes"][0]["x2"] - r["q"][k]["boxes"][0]["x1"]) < 1
                                       or r["q"][k]["boxes"] and (r["q"][k]["boxes"][0]["y2"] - r["q"][k]["boxes"][0]["y1"]) < 1)}
    ok = [r["order_ok"] for r in results if r["order_ok"] is not None]
    summ["order_ok"] = {"n": len(ok), "true": sum(ok)}
    we = [r["width_err"] for r in results if r["width_err"] is not None]
    summ["mean_width_err"] = sum(we) / len(we) if we else None
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    print("done", n)


if __name__ == "__main__":
    main()
