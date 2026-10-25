# Runs INSIDE the Colab VM. Downloads RASAM-1 and RASAM-2 from the Hugging Face Hub and packs both with the repo's prep script (A3 data).
# Idempotent: if the pickle already exists it only reprints the recorded timing.
import os
import subprocess
import sys
import time

if os.path.exists("/content/rasam_r12.pkl") and os.path.exists("/content/prep_timing.txt"):
    print(open("/content/prep_timing.txt").read(), flush=True)
else:
    t0 = time.time()
    from huggingface_hub import snapshot_download

    for repo in ("calfa-ai/RASAM-1", "calfa-ai/RASAM-2"):
        snapshot_download(repo, repo_type="dataset", cache_dir="/content/hfcache", allow_patterns=["data/*"])
    t1 = time.time()
    env = dict(os.environ, RASAM_HF_ROOT="/content/hfcache/datasets--calfa-ai--")
    subprocess.run([sys.executable, "/content/rasam_prepare_crnn_data.py", "/content/rasam_r12.pkl", "--sets", "RASAM-1", "RASAM-2", "--labels", "raw"], check=True, env=env)
    msg = "TIMING download_s=%.1f prep_s=%.1f cpus=%d" % (t1 - t0, time.time() - t1, os.cpu_count())
    open("/content/prep_timing.txt", "w").write(msg)
    print(msg, flush=True)
