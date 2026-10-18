"""Letter features for the novel-shape clusters (checklist 2.2, shape-programme step 1), CPU.

    python scripts/shape_letter_feats.py --out results/characterize/shape/letter_feats.npz [--n-bank 4000 --n-nakba 1000]

Same embedding as X-G10 (xai_letter_shapes.py): the Muharaf-trained aligner CRNN (aligner_run/model_final.pt) force-aligns
each line to its label; a letter's feature is the mean CNN column (1024 numbers) inside its CTC span, L2-normalised. Runs on
the CPU and keeps the line's greedy reading so the atlas can show readings.

Sets (stored images, not ink-cropped, to match the bank):
  train        n-bank random Muharaf training lines (aligner_data.pkl, role train)
  muharaf_test the Muharaf test panel (in-domain control: should populate clusters like train does)
  khatt        KHATT v1.0 full test (results/panel/khatt_full_panel.pkl, tag kitab_khatt_full)
  nakba_lines / nakba_blind  n-nakba random lines each from panel.pkl

Output: one npz bundle (features PCA-reduced to 128 dims, fitted on the bank, float16; per-letter meta arrays; per-line
id / set / label / reading / image index) plus lines.pkl with the line images, so the atlas can cut crops later.
"""
from __future__ import annotations

import argparse
import difflib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from xai_align import char_spans, force_align  # noqa: E402
from xai_letter_shapes import LETTERS, form_of  # noqa: E402
from xai_train_aligner import CRNN, greedy, normalise  # noqa: E402

BOX = Path("/Volumes/PortableSSD/aocr-box-backup/xai-2026-10-13/xai/aligner")
PANEL = Path(__file__).resolve().parents[1] / "results/panel"


def run_lines(model, rows, stoi, itos, set_name):
    feats, meta, lines = [], [], []
    t0 = time.time()
    for n, r in enumerate(rows):
        arr = r["img"]
        w = arr.shape[1]
        x = np.ones((1, 1, 64, int(np.ceil(w / 4) * 4)), dtype=np.float32)
        x[0, 0, :, :w] = normalise(arr)
        xt = torch.from_numpy(x)
        with torch.no_grad():
            f = model.cnn(xt)
            b, c, h, t = f.shape
            g = f.permute(3, 0, 1, 2).reshape(t, b, c * h)
            g, _ = model.rnn(torch.nn.functional.relu(model.proj(g)))
            logits = model.fc(g)
        T = w // 4
        cols = f[0].permute(2, 0, 1).reshape(f.shape[3], -1)[:T]
        lp = logits[:T, 0].log_softmax(-1).numpy().astype(np.float64)
        lab = r["label"]
        vis = lab[::-1]
        tgt = [stoi[ch] for ch in vis if ch in stoi]
        if len(tgt) != len(vis) or not tgt:
            continue
        path = force_align(lp, tgt)
        if path is None:
            continue
        try:
            spans = char_spans(path, len(tgt), T)
        except IndexError:  # Viterbi path skipped a label (line too short for its text): drop the line
            continue
        hyp = "".join(itos.get(k, "?") for k in greedy(logits[:T], torch.tensor([w]))[0])[::-1]
        ok = np.zeros(len(lab), bool)
        for op, i1, i2, _, _ in difflib.SequenceMatcher(None, lab, hyp, autojunk=False).get_opcodes():
            if op == "equal":
                ok[i1:i2] = True
        li = len(lines)
        lines.append({"id": r["id"], "set": set_name, "label": lab, "hyp": hyp, "img": arr})
        L = len(lab)
        for i, ch in enumerate(lab):
            if ch not in LETTERS:
                continue
            s0, s1 = (int(v) for v in spans[L - 1 - i])
            s1 = max(s1, s0 + 1)
            feats.append(cols[s0:s1].mean(0).numpy().astype(np.float16))
            meta.append((li, i, ch, form_of(lab, i), s0 * 4, s1 * 4, bool(ok[i])))
        if n % 500 == 0:
            print(set_name, n, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    return feats, meta, lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bank", type=int, default=4000)
    ap.add_argument("--n-nakba", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--inkcrop", action="store_true", help="use the ink-cropped copies of every set (bank: aligner_data_inkcrop.pkl; "
                    "panel: the __inkcrop tags), a control for scale and margin differences between datasets")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: lines per set")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = BOX / "aligner_run"
    chars = json.loads((run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    model = CRNN(len(chars) + 1).eval()
    model.load_state_dict(torch.load(run / "model_final.pt", map_location="cpu"))

    sfx = "__inkcrop" if a.inkcrop else ""
    tr = pickle.load(open(BOX / ("aligner_data_inkcrop.pkl" if a.inkcrop else "aligner_data.pkl"), "rb"))["rows"]
    bank = [r for r in tr if r["role"] == "train" and r["dataset"] == "muharaf_lines"]
    del tr
    random.Random(0).shuffle(bank)  # same order as X-G10, so its 2,000-line bank is the first half of ours
    sets = {"train": bank[: a.n_bank]}
    del bank
    pn = pickle.load(open(PANEL / "panel.pkl", "rb"))["rows"]
    sets["muharaf_test"] = [r for r in pn if r["dataset"] == "muharaf_lines" + sfx]
    for tag in ("nakba_lines", "nakba_blind"):
        rr = [r for r in pn if r["dataset"] == tag + sfx]
        random.Random(1).shuffle(rr)
        sets[tag] = rr[: a.n_nakba]
    del pn
    sets["khatt"] = [r for r in pickle.load(open(PANEL / "khatt_full_panel.pkl", "rb"))["rows"] if r["dataset"] == "kitab_khatt_full" + sfx]
    F, M, LINES = [], [], []
    for name, rows in sets.items():
        if a.limit:
            rows = rows[: a.limit]
        f, m, ln = run_lines(model, rows, stoi, itos, name)
        off = len(LINES)
        F += f
        M += [(li + off, *rest) for li, *rest in m]
        LINES += ln
        print(name, "lines", len(ln), "letters", len(m), flush=True)
    X = np.stack(F).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    nb = sum(1 for m in M if LINES[m[0]]["set"] == "train")
    mu = X[:nb].mean(0)
    _, sv, vt = np.linalg.svd(X[:nb][:: max(1, nb // 60000)] - mu, full_matrices=False)
    P = vt[:128]
    Z = (X - mu) @ P.T
    expl = float((sv[:128] ** 2).sum() / (sv ** 2).sum())
    print("PCA 128 explained variance", round(expl, 3), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, z=Z.astype(np.float16), pca_explained=expl,
        line=np.array([m[0] for m in M], np.int32), pos=np.array([m[1] for m in M], np.int16),
        ch=np.array([m[2] for m in M]), form=np.array([m[3] for m in M]),
        x0=np.array([m[4] for m in M], np.int32), x1=np.array([m[5] for m in M], np.int32), ok=np.array([m[6] for m in M]),
        line_id=np.array([l["id"] for l in LINES]), line_set=np.array([l["set"] for l in LINES]),
        line_label=np.array([l["label"] for l in LINES]), line_hyp=np.array([l["hyp"] for l in LINES]))
    with open(a.out.with_name("letter_lines.pkl"), "wb") as fh:
        pickle.dump([l["img"] for l in LINES], fh, protocol=4)
    print("wrote", a.out, X.shape, flush=True)


if __name__ == "__main__":
    main()
