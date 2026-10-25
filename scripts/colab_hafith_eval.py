# Runs INSIDE the Colab VM after `pip install -e /content/hafith` (see configs/colab/hafith_eval_t4.json).
# 1. Overlap check: HAFITH's fine-tuning data (mdnaseif/hafith-combined-benchmark, source == RASAM) vs the official RASAM-1 / RASAM-2 test lines,
#    by normalised text, per HAFITH split. Any overlap of HAFITH TRAIN lines with our test lines would inflate its score.
# 2. Predictions of HAFITH on the full official test splits (RASAM-1 test 734, RASAM-2 test 372), saved raw; CER is computed on the Mac with the
#    project's own metric (scripts/rasam_score_preds.py) so the comparison is identical to ours.
import json
import os
import re
import time
import unicodedata

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

OUT = "/content/crnn_out/hafith"
os.makedirs(OUT, exist_ok=True)
BIDI = re.compile("[​-‏‪-‮⁦-⁩﻿؜]")


def norm(t):
    return " ".join(BIDI.sub("", unicodedata.normalize("NFC", t or "")).split())


def rows(repo, split, cols=None):
    d = snapshot_download(repo, repo_type="dataset", cache_dir="/content/hfcache", allow_patterns=[f"data/{split}-*"])
    out = []
    for root, _, fs in os.walk(d):
        for f in sorted(fs):
            if f.startswith(split) and f.endswith(".parquet"):
                p = os.path.join(root, f)
                use = cols
                if cols == "noimage":  # text-only comparison: never load the image column into memory
                    use = [c for c in pq.ParquetFile(p).schema_arrow.names if "image" not in c.lower()]
                out += pq.read_table(p, columns=use).to_pylist()
    return out


t0 = time.time()
tests = {"r1": rows("calfa-ai/RASAM-1", "test"), "r2": rows("calfa-ai/RASAM-2", "test")}
test_txt = {k: {t for t in (norm(r["transcription"]) for r in v) if len(t) >= 12} for k, v in tests.items()}
ov = {}
for split in ("train", "val", "test"):
    # The published benchmark has only `image` and `text` (no `source` column, despite its README), so ALL its lines are compared:
    # only RASAM lines can coincide with RASAM test lines. Lines shorter than 12 characters are ignored (common short phrases).
    hb = rows("mdnaseif/hafith-combined-benchmark", split, cols="noimage")
    txt_col = next((c for c in ("text", "transcription", "label") if hb and c in hb[0]), None)
    hset = {t for t in (norm(r[txt_col]) for r in hb) if len(t) >= 12} if txt_col else set()
    ov[split] = {"hafith_lines": len(hb), "text_column": txt_col,
                 "overlap_with_official_r1_test": len(hset & test_txt["r1"]), "overlap_with_official_r2_test": len(hset & test_txt["r2"])}
json.dump(ov, open(f"{OUT}/hafith_overlap.json", "w"), ensure_ascii=False, indent=1)
print("OVERLAP", json.dumps(ov), flush=True)

from PIL import Image  # noqa: E402
import io  # noqa: E402
from hafith import HAFITHRecognizer  # noqa: E402

rec = HAFITHRecognizer.from_pretrained("mdnaseif/hafith")
timing = {"setup_s": round(time.time() - t0, 1)}
for k, data in tests.items():
    t = time.time()
    imgs = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in data]
    hyps = rec.recognize_batch(imgs, batch_size=16)
    json.dump([{"i": i, "ref": r["transcription"], "hyp": h, "manuscript": r["manuscript_name"]} for i, (r, h) in enumerate(zip(data, hyps))],
              open(f"{OUT}/preds_{k}.json", "w"), ensure_ascii=False)
    timing[f"{k}_s"] = round(time.time() - t, 1)
    print(f"PRED {k}: {len(hyps)} lines in {timing[k + '_s']} s", flush=True)
json.dump(timing, open(f"{OUT}/hafith_timing.json", "w"))
print("DONE", flush=True)
