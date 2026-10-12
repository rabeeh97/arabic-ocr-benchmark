"""Shape-targeted oversampling (X-G11 lite): find training lines whose letters look like KHATT's unusual letters. Runs on the 3090.

    python xai_shape_select.py --train-data aligner_data.pkl --eval-data eval_data.pkl --run aligner_run --out shape_select

No KHATT labels are used for the selection. KHATT lines are split by id into half A (selection) and half B (evaluation).
1. Half A: the CRNN's own greedy reading gives each predicted letter a span (its run of frames on the best path); each letter
   gets the mean CNN feature over that span.
2. Training bank: every training line is force-aligned with its own label; each letter gets the same feature and its line index.
3. A half-A letter is "unusual" when its novelty (cosine distance to the nearest training letter of the same predicted
   character, over the median nearest-neighbour distance in the bank) exceeds the Muharaf 95th percentile from X-G10.
4. Each unusual letter marks its 25 nearest training letters of the same character; a training line's score is how many marks
   it got. The top-scoring lines (up to 12 % of training lines) get 3 extra copies; a control file gives 3 extra copies to the
   same number of lines drawn at random.
Writes targeted.json, random.json (for --repeat-file) and halves.json (the half-B ids for evaluation).
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from xai_align import char_spans, force_align  # noqa: E402
from xai_letter_shapes import LETTERS  # noqa: E402
from xai_train_aligner import CRNN, normalise  # noqa: E402


def run_line(model, arr):
    w = arr.shape[1]
    x = np.ones((1, 1, 64, int(np.ceil(w / 4) * 4)), dtype=np.float32)
    x[0, 0, :, :w] = normalise(arr)
    xt = torch.from_numpy(x).cuda()
    with torch.no_grad():
        f = model.cnn(xt)
        logits = model(xt)
    T = w // 4
    cols = f[0].permute(2, 0, 1).reshape(f.shape[3], -1)[:T].float()
    return cols, logits[:T, 0].float()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--eval-data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=2.128, help="novelty above which a letter counts as unusual (X-G10 Muharaf 95th pct)")
    ap.add_argument("--share", type=float, default=0.12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    model = CRNN(len(chars) + 1).cuda().eval()
    model.load_state_dict(torch.load(a.run / "model_final.pt", map_location="cuda"))

    # 1. KHATT half A, unlabeled
    kh = sorted([r for r in pickle.load(open(a.eval_data, "rb"))["rows"] if r["dataset"] == "kitab_khatt"], key=lambda r: r["id"])
    half_a, half_b = kh[0::2], kh[1::2]
    qf, qc = [], []
    for r in half_a:
        cols, logits = run_line(model, r["img"])
        path = logits.argmax(-1).cpu().numpy()
        t = 0
        while t < len(path):
            k = path[t]
            u = t
            while u + 1 < len(path) and path[u + 1] == k:
                u += 1
            if k != 0 and itos.get(int(k)) in LETTERS:
                qf.append(cols[t : u + 1].mean(0))
                qc.append(itos[int(k)])
            t = u + 1
    Q = torch.nn.functional.normalize(torch.stack(qf), dim=1)
    print("half A lines", len(half_a), "predicted letters", len(qc), flush=True)

    # 2. training bank, force-aligned with the training labels
    rows = pickle.load(open(a.train_data, "rb"))["rows"]
    train_idx = [j for j, r in enumerate(rows) if r["role"] in ("train", "panel")]
    bf, bc, bl = [], [], []
    for n, j in enumerate(train_idx):
        r = rows[j]
        cols, logits = run_line(model, r["img"])
        vis = r["label"][::-1]
        tgt = [stoi[c] for c in vis if c in stoi]
        if len(tgt) != len(vis):
            continue
        lp = logits.log_softmax(-1).cpu().numpy().astype(np.float64)
        path = force_align(lp, tgt)
        if path is None:
            continue
        try:
            spans = char_spans(path, len(tgt), lp.shape[0])
        except IndexError:  # a letter with no frame on the path
            continue
        for v, ch in enumerate(vis):
            if ch in LETTERS:
                s0, s1 = (int(x) for x in spans[v])
                bf.append(cols[s0 : max(s1, s0 + 1)].mean(0).half())
                bc.append(ch)
                bl.append(j)
        if n % 4000 == 0:
            print("bank lines", n, "letters", len(bc), flush=True)
    B = torch.nn.functional.normalize(torch.stack(bf).float(), dim=1)
    groups = defaultdict(list)
    for i, c in enumerate(bc):
        groups[c].append(i)
    gidx = {c: torch.tensor(v, device="cuda") for c, v in groups.items()}
    scale = {}
    for c, idx in gidx.items():
        sub = idx[torch.randperm(len(idx), device="cuda")[:6000]]
        X = B[sub]
        d = 1 - X @ X.T
        d.fill_diagonal_(9)
        scale[c] = max(float(d.min(1).values.median()), 1e-3)

    # 3. unusual half-A letters, 4. marks on their nearest training letters
    marks = defaultdict(int)
    unusual = 0
    for q, c in zip(Q, qc):
        if c not in gidx or len(gidx[c]) < 30:
            continue
        sims = B[gidx[c]] @ q
        if (1 - float(sims.max())) / scale[c] < a.threshold:
            continue
        unusual += 1
        for i in gidx[c][sims.topk(min(25, len(sims))).indices].tolist():
            marks[bl[i]] += 1
    n_sel = int(a.share * len(train_idx))
    ranked = sorted(marks.items(), key=lambda kv: -kv[1])[:n_sel]
    targeted = {str(j): 3 for j, _ in ranked}
    rnd = random.Random(0).sample(train_idx, len(targeted))
    (a.out / "targeted.json").write_text(json.dumps({"extra": targeted}))
    (a.out / "random.json").write_text(json.dumps({"extra": {str(j): 3 for j in rnd}}))
    (a.out / "halves.json").write_text(json.dumps({"half_a": [r["id"] for r in half_a], "half_b": [r["id"] for r in half_b]}))
    summ = {"half_a_letters": len(qc), "unusual_letters": unusual, "lines_marked": len(marks), "lines_selected": len(targeted), "train_lines": len(train_idx),
            "min_marks_selected": ranked[-1][1] if ranked else 0, "examples": [{"row": j, "marks": m, "label": rows[j]["label"]} for j, m in ranked[:12]]}
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "examples"}), flush=True)


if __name__ == "__main__":
    main()
