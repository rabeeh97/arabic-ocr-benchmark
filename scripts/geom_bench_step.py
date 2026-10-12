"""Time and peak memory of CRNN training steps on a device: python geom_bench_step.py DEVICE BATCH WIDTH_MULT LINE_WIDTH"""
import resource
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import CRNN  # noqa: E402

torch.set_num_threads(4)
dev, bs, wm, W = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
m = CRNN(150, width=wm).to(dev)
print("params", sum(p.numel() for p in m.parameters()) / 1e6, "M")
opt = torch.optim.AdamW(m.parameters(), 1e-3)
ctc = nn.CTCLoss(zero_infinity=True)
x = torch.randn(bs, 1, 64, W, device=dev)
tgt = torch.randint(1, 150, (bs * 40,))
tl, il = torch.full((bs,), 40), torch.full((bs,), W // 4)
for i in range(4):
    t = time.time()
    loss = ctc(m(x).log_softmax(-1), tgt, il, tl)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {i} {time.time() - t:.2f}s  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB", flush=True)
