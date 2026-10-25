# Runs INSIDE the Colab VM. Builds CRNN data packs described by the env var PACKS (JSON list), e.g.
#   PACKS='[{"out": "/content/pack_r12tm_h96.pkl", "sets": ["RASAM-1", "RASAM-2", "tarima"], "height": 96},
#           {"out": "/content/pack_r1all_h96.pkl", "sets": ["RASAM-1", "RASAM-2"], "height": 96, "all_train": ["RASAM-1"]}]'
# Downloads every needed calfa-ai dataset once, then runs the repo's prep script per pack. Idempotent (existing packs are skipped).
import json
import os
import subprocess
import sys
import time

packs = json.loads(os.environ["PACKS"])
t0 = time.time()
from huggingface_hub import snapshot_download

for s in sorted({s for p in packs for s in p["sets"]}):
    snapshot_download(f"calfa-ai/{s}", repo_type="dataset", cache_dir="/content/hfcache", allow_patterns=["data/*"])
timing = {"download_s": round(time.time() - t0, 1), "cpus": os.cpu_count()}
env = dict(os.environ, RASAM_HF_ROOT="/content/hfcache/datasets--calfa-ai--")
for p in packs:
    if os.path.exists(p["out"]):
        continue
    t = time.time()
    cmd = [sys.executable, "/content/rasam_prepare_crnn_data.py", p["out"], "--sets", *p["sets"], "--labels", p.get("labels", "raw"), "--height", str(p.get("height", 64))]
    if p.get("all_train"):
        cmd += ["--all-train-sets", *p["all_train"]]
    if p.get("repair"):
        cmd += ["--repair", p["repair"]]
    if p.get("channel"):
        cmd += ["--channel", p["channel"]]
    subprocess.run(cmd, check=True, env=env)
    timing[os.path.basename(p["out"])] = round(time.time() - t, 1)
os.makedirs("/content/crnn_out", exist_ok=True)
json.dump(timing, open("/content/crnn_out/packs_timing.json", "w"))
print("TIMING", json.dumps(timing), flush=True)
