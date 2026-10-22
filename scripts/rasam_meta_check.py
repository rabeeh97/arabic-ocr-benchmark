"""RS1/RS3: manuscript and page overlap across RASAM splits; writes results/rasam/rasam_meta.json."""
import glob
import json
from collections import Counter

import pyarrow.parquet as pq

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
COLS = ["transcription", "support_type", "region_type", "manuscript_name", "image_name", "script"]
out = {}
for name in ("RASAM-1", "RASAM-2"):
    per = {}
    for f in sorted(glob.glob(f"{ROOT}{name}/snapshots/*/data/*.parquet")):
        sp = f.rsplit("/", 1)[1].split("-")[0]
        t = pq.read_table(f, columns=COLS).to_pylist()
        per.setdefault(sp, []).extend(t)
    info = {"manuscripts": {}, "pages": {}, "lines": {}, "region_type": {}, "support_type": {}, "script": {}}
    for sp, rows in per.items():
        info["lines"][sp] = len(rows)
        info["manuscripts"][sp] = dict(Counter(r["manuscript_name"] for r in rows))
        info["pages"][sp] = sorted({(r["manuscript_name"], r["image_name"]) for r in rows})
        info["region_type"][sp] = dict(Counter(r["region_type"] for r in rows))
        info["support_type"][sp] = dict(Counter(r["support_type"] for r in rows))
        info["script"][sp] = dict(Counter(r["script"] for r in rows))
    pages = {sp: set(map(tuple, p)) for sp, p in info["pages"].items()}
    info["page_overlap"] = {
        f"{a}&{b}": len(pages[a] & pages[b])
        for a in pages for b in pages if a < b
    }
    info["n_pages"] = {sp: len(p) for sp, p in pages.items()}
    info["pages"] = None
    # exact-text overlap of test lines with train
    train_txt = {r["transcription"] for r in per["train"]}
    info["test_text_in_train"] = sum(r["transcription"] in train_txt for r in per["test"]) / len(per["test"])
    out[name] = info
    print(name, "lines", info["lines"], "pages", info["n_pages"], "page_overlap", info["page_overlap"],
          "test text seen in train: %.1f%%" % (100 * info["test_text_in_train"]))
    for sp in ("train", "validation", "test"):
        print("  ", sp, info["manuscripts"][sp])
    print("   region", info["region_type"]["test"], "support", info["support_type"]["test"], "script", info["script"]["test"])
json.dump(out, open("results/rasam/rasam_meta.json", "w"), indent=1, ensure_ascii=False)
