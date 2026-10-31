# Runs INSIDE a Colab CPU VM. Reads the 25 re-cut official test lines (/content/recut_test.json from scripts/clean_rerun_local.py --stage export)
# with a public model, the same way the full-test runs did:
#   python colab_recut_rerun.py hafith   HAFITH (mdnaseif/hafith), recognize_batch on the RGB crops (as colab_hafith_eval.py)
#   python colab_recut_rerun.py amidda   AMIDDA 1.0 Line 4B with the card's settings (as colab_amidda_eval.py), one line at a time
# Writes /content/crnn_out/recut/<model>.json {key: hyp} after every line (RASAM-1/2 only for HAFITH, as in its full-test run).
import base64
import io
import json
import os
import sys
import time

from PIL import Image

OUT = "/content/crnn_out/recut"
os.makedirs(OUT, exist_ok=True)
lines = json.load(open("/content/recut_test.json"))
which = sys.argv[1]
fo = f"{OUT}/{which}.json"
done = json.load(open(fo)) if os.path.exists(fo) else {}
t0 = time.time()


def img(d):
    return Image.open(io.BytesIO(base64.b64decode(d["png"]))).convert("RGB")


if which == "hafith":
    from hafith import HAFITHRecognizer

    rec = HAFITHRecognizer.from_pretrained("mdnaseif/hafith")
    todo = [d for d in lines if d["set"] in ("r1", "r2") and d["key"] not in done]
    hyps = rec.recognize_batch([img(d) for d in todo], batch_size=8)
    done.update({d["key"]: h for d, h in zip(todo, hyps)})
    json.dump(done, open(fo, "w"), ensure_ascii=False)
elif which == "amidda":
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    MODEL = "calfa-ai/amidda1.0Line4B"
    PROMPT = "Transcribe the handwritten text in this line image. Output ONLY the transcription."
    torch.set_num_threads(os.cpu_count() or 2)
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=401408)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    for d in lines:
        if d["key"] in done:
            continue
        im = img(d)
        w, h = im.size
        if w > 150 * h:  # as colab_amidda_eval.py (the processor refuses > 200:1)
            c = Image.new("RGB", (w, -(-w // 150)), (255, 255, 255))
            c.paste(im, (0, (c.height - h) // 2))
            im = c
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                               padding=True, enable_thinking=False)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        done[d["key"]] = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        json.dump(done, open(fo + ".tmp", "w"), ensure_ascii=False)
        os.replace(fo + ".tmp", fo)
        print(f"amidda {len(done)}/{len(lines)} {time.time() - t0:.0f}s", flush=True)
print(f"DONE {which} {len(done)} lines {time.time() - t0:.0f}s", flush=True)
