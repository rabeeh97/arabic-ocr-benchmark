# Runs INSIDE the Colab kernel (sent by `colab exec -f` after the driver replaces @EPOCHS@ and @WIDTH@ and @BS@).
# Same trainer, flags and seed as the Mac CPU run (A1): xai_train_aligner.py, RASAM-1 train only, geom-aug, seed 0.
import subprocess
import sys
import time

import torch

print("GPU", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuda", torch.version.cuda, flush=True)
t0 = time.time()
cmd = [sys.executable, "-u", "/content/xai_train_aligner.py", "--data", "/content/rasam_r1.pkl", "--out", "/content/crnn_out",
       "--epochs", "@EPOCHS@", "--bs", "@BS@", "--device", "cuda", "--width", "@WIDTH@", "--train-sets", "r1",
       "--eval-every", "@EPOCHS@", "--seed", "0", "--geom-aug"]
p = subprocess.run(cmd, capture_output=True, text=True)
for line in (p.stdout + p.stderr).splitlines():
    if line.startswith(("train ", "params", "epoch", "done", "Traceback")) or "Error" in line:
        print(line, flush=True)
print("TIMING train_total_s=%.1f rc=%d" % (time.time() - t0, p.returncode), flush=True)
