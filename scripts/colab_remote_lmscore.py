# Runs INSIDE a Colab VM (a CPU session is enough). Scores every N-best candidate in /content/NB_*.json with a pretrained Arabic causal LM and
# writes /content/crnn_out/lmscore/preds_lmscore_<file>.json: per line, per candidate [total log-prob in nats, number of tokens].
# Usage: python colab_remote_lmscore.py [MODEL]   (default aubmindlab/aragpt2-base)
import glob
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = sys.argv[1] if len(sys.argv) > 1 else "aubmindlab/aragpt2-base"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 2)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).to(dev).eval()
bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
out_dir = "/content/crnn_out/lmscore"
os.makedirs(out_dir, exist_ok=True)


@torch.no_grad()
def score(texts):
    res = []
    for t in texts:
        ids = [bos] + tok(t or " ", add_special_tokens=False)["input_ids"][:254]
        x = torch.tensor([ids], device=dev)
        lp = model(x).logits.float().log_softmax(-1)[0, :-1]
        tgt = x[0, 1:]
        res.append([float(lp.gather(1, tgt[:, None]).sum()), len(ids) - 1])
    return res


t0 = time.time()
for f in sorted(glob.glob("/content/NB_*.json")):
    d = json.load(open(f))
    out = [score([c[0] for c in line.get("nbest", [])]) for line in d["lines"]]
    base = os.path.basename(f)[:-5]
    json.dump({"model": name, "file": base, "scores": out}, open(f"{out_dir}/preds_lmscore_{base}.json", "w"))
    print(f"SCORED {base}: {len(out)} lines, {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
