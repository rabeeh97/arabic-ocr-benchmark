# Runs INSIDE a Colab GPU VM. Calfa's AMIDDA 1.0 Line 4B (Qwen3.5-VL-4B LoRA, merged) on the official HF test lines of RASAM-1, RASAM-2 and TariMa,
# with the model card's settings verbatim: max_pixels 401408, bf16, the card's prompt, enable_thinking=False, greedy, max_new_tokens 128.
# The only change is batching (left padding), which does not change greedy outputs beyond float noise.
# Writes /content/crnn_out/amidda/preds_amidda_<set>.json ({"lines": [{"ref", "hyp"}]}) after every batch, so the HF pusher and the Mac sync
# keep partial results; resumes from those files.
import io
import json
import os
import sys
import time

import torch
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "calfa-ai/amidda1.0Line4B"
PROMPT = "Transcribe the handwritten text in this line image. Output ONLY the transcription."
SETS = {"r1": "calfa-ai/RASAM-1", "r2": "calfa-ai/RASAM-2", "tm": "calfa-ai/tarima"}
OUT = "/content/crnn_out/amidda"
BS = int(os.environ.get("AMIDDA_BS", "8"))
os.makedirs(OUT, exist_ok=True)


def test_rows(repo):
    rows = []
    for f in sorted(x for x in list_repo_files(repo, repo_type="dataset") if x.startswith("data/test-") and x.endswith(".parquet")):
        d = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=["image", "transcription"]).to_pydict()
        rows += [(d["image"][i]["bytes"], d["transcription"][i]) for i in range(len(d["transcription"]))]
    return rows


def load(b):
    """Line image as RGB. The Qwen processor refuses images more than 200x wider than tall (the 80 broken MS 1936 crops in RASAM-2 are 1-7 px
    tall); those are padded with white above and below to a 150:1 ratio. They have no readable ink either way."""
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    if w > 150 * h:
        nh = -(-w // 150)
        canvas = Image.new("RGB", (w, nh), (255, 255, 255))
        canvas.paste(im, (0, (nh - h) // 2))
        im = canvas
    return im


def read(images):
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}] for im in images]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                           padding=True, enable_thinking=False).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return [h.strip() for h in processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]


t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL, max_pixels=401408)
processor.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
print(f"loaded in {time.time() - t0:.0f}s on {model.device}", flush=True)
for s in (sys.argv[1:] or ["r1", "r2", "tm"]):
    rows = test_rows(SETS[s])
    fo = f"{OUT}/preds_amidda_{s}.json"
    done = json.load(open(fo))["lines"] if os.path.exists(fo) else []
    t1 = time.time()
    for k in range(len(done), len(rows), BS):
        chunk = rows[k : k + BS]
        images = [load(b) for b, _ in chunk]
        try:
            hyps, errs = read(images), [None] * len(chunk)
        except Exception as e:  # one bad line must not stop the run: retry line by line, empty reading for a line that still fails
            print(f"batch at {k} failed ({e}); line by line", flush=True)
            hyps, errs = [], []
            for im in images:
                try:
                    hyps.append(read([im])[0]); errs.append(None)
                except Exception as e2:
                    hyps.append(""); errs.append(str(e2)[:200])
        done += [{"ref": r, "hyp": h, **({"error": er} if er else {})} for (_, r), h, er in zip(chunk, hyps, errs)]
        json.dump({"model": MODEL, "set": s, "lines": done}, open(fo + ".tmp", "w"), ensure_ascii=False)
        os.replace(fo + ".tmp", fo)
        print(f"epoch-like progress {s} {len(done)}/{len(rows)} {time.time() - t1:.0f}s", flush=True)
    print(f"SET DONE {s} {len(done)} lines {time.time() - t1:.0f}s", flush=True)
print("ALLDONE", flush=True)
