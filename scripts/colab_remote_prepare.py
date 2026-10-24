# Runs INSIDE the Colab kernel (sent by `colab exec -f`). Downloads RASAM-1 from the Hugging Face Hub and packs it with the repo's prep script.
import os
import subprocess
import sys
import time

t0 = time.time()
from huggingface_hub import snapshot_download

snapshot_download("calfa-ai/RASAM-1", repo_type="dataset", cache_dir="/content/hfcache", allow_patterns=["data/*"])
t1 = time.time()
env = dict(os.environ, RASAM_HF_ROOT="/content/hfcache/datasets--calfa-ai--")
subprocess.run([sys.executable, "/content/rasam_prepare_crnn_data.py", "/content/rasam_r1.pkl", "--sets", "RASAM-1", "--labels", "raw"], check=True, env=env)
print("TIMING download_s=%.1f prep_s=%.1f" % (t1 - t0, time.time() - t1), flush=True)
