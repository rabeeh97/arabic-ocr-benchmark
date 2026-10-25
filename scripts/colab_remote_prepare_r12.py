# Runs INSIDE the Colab VM. Downloads RASAM-1 and RASAM-2 from the Hugging Face Hub and packs both with the repo's prep script (A3 data).
# Heights come from the env var RASAM_HEIGHTS (default "64"): 64 -> /content/rasam_r12.pkl, other heights -> /content/rasam_r12_h<H>.pkl.
# Idempotent: packs that already exist are skipped (an exec whose connection dropped can simply be re-sent).
import os
import subprocess
import sys
import time

heights = [int(h) for h in os.environ.get("RASAM_HEIGHTS", "64").split()]
t0 = time.time()
from huggingface_hub import snapshot_download

for repo in ("calfa-ai/RASAM-1", "calfa-ai/RASAM-2"):
    snapshot_download(repo, repo_type="dataset", cache_dir="/content/hfcache", allow_patterns=["data/*"])
t1 = time.time()
env = dict(os.environ, RASAM_HF_ROOT="/content/hfcache/datasets--calfa-ai--")
for h in heights:
    out = "/content/rasam_r12.pkl" if h == 64 else f"/content/rasam_r12_h{h}.pkl"
    if os.path.exists(out):
        continue
    t = time.time()
    subprocess.run([sys.executable, "/content/rasam_prepare_crnn_data.py", out, "--sets", "RASAM-1", "RASAM-2", "--labels", "raw", "--height", str(h)], check=True, env=env)
    print("TIMING prep_h%d_s=%.1f" % (h, time.time() - t), flush=True)
print("TIMING download_s=%.1f total_s=%.1f cpus=%d" % (t1 - t0, time.time() - t0, os.cpu_count()), flush=True)
