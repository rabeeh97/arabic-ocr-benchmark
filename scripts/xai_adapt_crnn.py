"""CRNN few-shot writer adaptation (checklist 4.3, `writer-adapt-crnn`) and the few-shot Maghrebi curve (checklist 4.4, `E6`).
Design: docs/research/designs/writer-adaptation.md, part A. Runs on the 3090.

    python xai_adapt_crnn.py --testbed rasam2 --run aligner_run_std_s0 --panel panel.pkl --bank-data aligner_data_standard.pkl \
        --out adapt_out\\rasam2.json --seed 0 --ckpt-every 10
    python xai_adapt_crnn.py --testbed nakba  ... --pool-data aligner_data_nakba.pkl --out adapt_out\\nakba.json --seed 0 --ckpt-every 10
    python xai_adapt_crnn.py --testbed rasam1 ... --out adapt_out\\rasam1.json --seed 0 --ckpt-every 10

Base model: the standard-split CRNN (never saw RASAM, NAKBA or KHATT); its charset is kept (characters outside it are
dropped from the targets and count as errors in the denominator, as crnn_eval_panel.py).

Testbeds (panel lines as stored, not ink-cropped):
  rasam2  unit = manuscript (15). Adaptation lines come from the unit; test = the unit's remaining lines. K in {4, 8},
          run only when at least 16 lines remain (design rule).
  nakba   one writer. Adaptation lines from a fixed random pool of --pool-size NAKBA train lines; test = a fixed random
          sample of --test-per-tag lines from each of NAKBA test and blind (the same lines for every configuration).
          K in {4, 8, 16}.
  rasam1  the Maghrebi curve (E6). Per manuscript, the first 60 % of its page images (image_name order) form the
          adaptation pool and the rest the test pages; K lines are drawn from the pooled adaptation pages of all three
          manuscripts. K in {16, 100, 300}. Per-manuscript CER is in the output through per_line.
Selection (design A): random (--draws draws), coverage (greedy set cover over letter x positional form of the labels;
when nothing new can be covered the cover restarts), novelty (lines with the most predicted letters whose CNN-feature
novelty exceeds the 95th percentile of held-out Muharaf letters; bank = --bank-lines standard-split training lines,
force-aligned, as xai_shape_select.py / xai_letter_shapes.py). Novelty uses the image and the base model only, no labels.
Methods: full (all weights) and top (BiLSTM and output layer only); batch-norm statistics stay frozen in both.
--steps optimiser steps (AdamW, constant LR), batch = min(K, 16) lines drawn from the K with the training augmentation.
Every configuration is also run at K = 0 (base model) once per unit.

Crash safety: the results file is rewritten after every configuration and finished configurations are skipped on rerun;
inside a configuration the trainable weights and optimiser are checkpointed every --ckpt-every steps (one file,
<out stem>_ckpt.pt) and every step's batch and augmentation are seeded from (seed, configuration, step), so a resumed
configuration repeats the same batches.

Output JSON: {"meta": {...}, "results": {key: {"unit", "K", "selection", "draw", "method", "adapt_ids", "cer",
  "q_to_f": [GT qaf aligned, read as feh], "per_line": {id: [edits, label_len, reading]}}}}; key = unit|K|selection|draw|method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from xai_align import char_spans, force_align  # noqa: E402
from xai_letter_shapes import LETTERS, form_of  # noqa: E402
from xai_train_aligner import CRNN, edit_distance, greedy, make_batch  # noqa: E402

KS = {"rasam2": [4, 8], "nakba": [4, 8, 16], "rasam1": [16, 100, 300]}


def align_pairs(ref: str, hyp: str) -> list[tuple[str, str | None]]:
    """Levenshtein alignment; returns (ref char, aligned hyp char or None for a deletion)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0:
        if j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            j -= 1
    return out[::-1]


class Base:
    def __init__(self, run: Path, ckpt: str, dev: str):
        self.chars = json.loads((run / "charset.json").read_text(encoding="utf-8"))["chars"]
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        self.width = cfg.get("width", 1.0)
        self.state = torch.load(run / ckpt, map_location="cpu")
        self.dev = dev

    def model(self) -> CRNN:
        m = CRNN(len(self.chars) + 1, width=self.width)
        m.load_state_dict(self.state)
        return m.to(self.dev).eval()


@torch.no_grad()
def evaluate(model, rows, base: Base) -> dict:
    model.eval()
    pl, nq, nqf = {}, 0, 0
    for r in rows:
        x, lens, tgt = make_batch([r], base.stoi, False, base.dev)
        logits = model(x).float()
        hyp = greedy(logits, lens)[0]
        reading = "".join(base.chars[k - 1] for k in hyp)[::-1]
        pl[r["id"]] = [edit_distance(hyp, tgt[0]), max(1, len(r["label"])), reading]
        if "ق" in r["label"]:
            for a, b in align_pairs(r["label"], reading):
                if a == "ق":
                    nq += 1
                    nqf += b == "ف"
    cer = sum(v[0] for v in pl.values()) / max(1, sum(v[1] for v in pl.values()))
    return {"cer": cer, "q_to_f": [nq, nqf], "per_line": pl}


@torch.no_grad()
def letter_feats(model, r, base: Base, use_label: bool):
    """Per-letter CNN features (mean over the span). use_label: force-align the label (bank); else greedy runs (no labels)."""
    x, lens, _ = make_batch([r], base.stoi, False, base.dev)
    f = model.cnn(x)
    logits = model(x).float()
    T = max(1, int(lens[0]) // 4)
    cols = f[0].permute(2, 0, 1).reshape(f.shape[3], -1)[:T].float()
    out = []
    if use_label:
        vis = r["label"][::-1]
        tgt = [base.stoi[c] for c in vis if c in base.stoi]
        if len(tgt) != len(vis) or not tgt:
            return out
        path = force_align(logits[:T, 0].log_softmax(-1).cpu().numpy().astype(np.float64), tgt)
        if path is None:
            return out
        try:
            spans = char_spans(path, len(tgt), T)
        except IndexError:
            return out
        for v, ch in enumerate(vis):
            if ch in LETTERS:
                s0, s1 = int(spans[v][0]), int(spans[v][1])
                out.append((ch, cols[s0:max(s1, s0 + 1)].mean(0)))
        return out
    path = logits[:T, 0].argmax(-1).cpu().numpy()
    t = 0
    while t < len(path):
        k, u = path[t], t
        while u + 1 < len(path) and path[u + 1] == k:
            u += 1
        ch = base.itos.get(int(k))
        if k != 0 and ch in LETTERS:
            out.append((ch, cols[t:u + 1].mean(0)))
        t = u + 1
    return out


class Novelty:
    def __init__(self, base: Base, bank_rows: list, held_rows: list):
        model = base.model()
        feats, chars = [], []
        for r in bank_rows:
            for ch, v in letter_feats(model, r, base, True):
                feats.append(v); chars.append(ch)
        self.B = nn.functional.normalize(torch.stack(feats), dim=1)
        groups = defaultdict(list)
        for i, c in enumerate(chars):
            groups[c].append(i)
        self.groups = {c: torch.tensor(v, device=self.B.device) for c, v in groups.items() if len(v) >= 20}
        self.scale = {}
        for c, idx in self.groups.items():
            X = self.B[idx[:6000]]
            d = 1 - X @ X.T
            d.fill_diagonal_(9)
            self.scale[c] = max(float(d.min(1).values.median()), 0.02)
        held = [s for r in held_rows for s in self.line(model, r, base)]
        self.threshold = float(np.percentile(held, 95)) if held else 9.0
        print(f"novelty bank: {len(chars)} letters from {len(bank_rows)} lines; threshold (p95 of {len(held)} held-out letters) {self.threshold:.3f}", flush=True)

    def line(self, model, r, base) -> list[float]:
        out = []
        for ch, v in letter_feats(model, r, base, False):
            if ch in self.groups:
                q = nn.functional.normalize(v, dim=0)
                out.append((1 - float((self.B[self.groups[ch]] @ q).max())) / self.scale[ch])
        return out

    def scores(self, base: Base, rows: list) -> dict:
        model = base.model()
        return {r["id"]: sum(s > self.threshold for s in self.line(model, r, base)) for r in rows}


def coverage_order(rows: list, k: int) -> list:
    """Greedy set cover over (letter, positional form); ties go to the shorter line; restart when nothing new is covered."""
    forms = {r["id"]: {(ch, form_of(r["label"], i)) for i, ch in enumerate(r["label"]) if ch in LETTERS} for r in rows}
    length = {r["id"]: len(r["label"]) for r in rows}
    left, chosen, covered = [r["id"] for r in rows], [], set()
    while left and len(chosen) < k:
        best = max(left, key=lambda i: (len(forms[i] - covered), -length[i], i))
        if not forms[best] - covered and covered:
            covered = set()
            continue
        chosen.append(best); left.remove(best); covered |= forms[best]
    return chosen


def adapt(base: Base, train_rows: list, method: str, a, key: str, ckpt_path: Path):
    model = base.model()
    for p in model.parameters():
        p.requires_grad = method == "full"
    if method == "top":
        for mod in (model.rnn, model.fc):
            for p in mod.parameters():
                p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr_full if method == "full" else a.lr_top, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    h = int(hashlib.md5(f"{a.seed}|{key}".encode()).hexdigest()[:8], 16)
    start = 0
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=base.dev, weights_only=False)
        if st.get("key") == key:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
            print(f"  resumed {key} at step {start}", flush=True)
    bs = min(len(train_rows), 16)
    for step in range(start, a.steps):
        random.seed(h + step); torch.manual_seed(h + step)
        model.train()
        for mod in model.modules():  # frozen batch-norm statistics: a handful of lines must not move them
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        batch = random.sample(train_rows, bs)
        x, lens, tgt = make_batch(batch, base.stoi, True, base.dev)
        flat = torch.tensor([t for s in tgt for t in s], device=base.dev)
        tl = torch.tensor([len(s) for s in tgt], device=base.dev)
        loss = ctc(model(x).float().log_softmax(-1), flat, torch.clamp(lens // 4, min=1), tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save({"key": key, "step": step + 1, "model": model.state_dict(), "opt": opt.state_dict()}, tmp)
            tmp.replace(ckpt_path)
    return model


def testbed_units(a, panel_rows: list):
    """Yields (unit, pool rows, test rows or None = the unit's remaining lines, Ks)."""
    rng = random.Random(a.seed)
    if a.testbed == "rasam2":
        by = defaultdict(list)
        for r in panel_rows:
            if r["dataset"] == "rasam2":
                by[r["meta"]["manuscript_name"]].append(r)
        for u in sorted(by):
            ks = [k for k in a.ks if len(by[u]) - k >= 16]
            if ks:
                yield u, by[u], None, ks
            else:
                print(f"skip {u}: {len(by[u])} lines leave fewer than 16 test lines", flush=True)
    elif a.testbed == "nakba":
        pool = [r for r in pickle.load(open(a.pool_data, "rb"))["rows"] if r["role"] == "train"]
        pool = rng.sample(pool, min(a.pool_size, len(pool)))
        test = []
        for tag in ("nakba_lines", "nakba_blind"):
            rs = sorted((r for r in panel_rows if r["dataset"] == tag), key=lambda r: r["id"])
            test += rng.sample(rs, min(a.test_per_tag, len(rs)))
        yield "nakba", pool, test, a.ks
    else:
        pool, test = [], []
        by = defaultdict(list)
        for r in panel_rows:
            if r["dataset"] == "rasam1":
                by[r["meta"]["manuscript_name"]].append(r)
        for u, rs in sorted(by.items()):
            pages = sorted({r["meta"]["image_name"] for r in rs})
            adapt_pages = set(pages[: int(round(0.6 * len(pages)))])
            pool += [r for r in rs if r["meta"]["image_name"] in adapt_pages]
            test += [r for r in rs if r["meta"]["image_name"] not in adapt_pages]
        yield "rasam1", pool, test, [k for k in a.ks if k <= len(pool)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", choices=sorted(KS), required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--bank-data", type=Path, help="standard-split pickle: novelty bank (train) and threshold (heldout)")
    ap.add_argument("--bank-lines", type=int, default=2000)
    ap.add_argument("--pool-data", type=Path, help="nakba: NAKBA train pickle (aligner_data_nakba.pkl)")
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--test-per-tag", type=int, default=600)
    ap.add_argument("--ks", type=int, nargs="+")
    ap.add_argument("--selections", nargs="+", default=["random", "coverage", "novelty"])
    ap.add_argument("--draws", type=int, default=3, help="random draws per K")
    ap.add_argument("--methods", nargs="+", default=["full", "top"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr-full", type=float, default=1e-4)
    ap.add_argument("--lr-top", type=float, default=3e-4)
    ap.add_argument("--max-units", type=int, default=0, help="smoke tests: at most this many units")
    ap.add_argument("--max-test", type=int, default=0, help="smoke tests: at most this many test lines per configuration")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.ks = a.ks or KS[a.testbed]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = a.out.with_name(a.out.stem + "_ckpt.pt")
    res = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {"meta": {}, "results": {}}
    res["meta"] = {"testbed": a.testbed, "run": a.run.name, "ckpt": a.ckpt, "seed": a.seed, "steps": a.steps, "lr_full": a.lr_full, "lr_top": a.lr_top,
                   "selections": a.selections, "draws": a.draws, "methods": a.methods, "ks": a.ks}

    def save():
        tmp = a.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        tmp.replace(a.out)

    base = Base(a.run, a.ckpt, a.device)
    panel_rows = pickle.load(open(a.panel, "rb"))["rows"]
    nov = None
    if "novelty" in a.selections:
        brows = pickle.load(open(a.bank_data, "rb"))["rows"]
        rng = random.Random(a.seed)
        bank = rng.sample([r for r in brows if r["role"] == "train"], min(a.bank_lines, sum(r["role"] == "train" for r in brows)))
        held = rng.sample([r for r in brows if r["role"] == "heldout"], min(300, sum(r["role"] == "heldout" for r in brows)))
        del brows
        nov = Novelty(base, bank, held)
    units = list(testbed_units(a, panel_rows))
    if a.max_units:
        units = units[: a.max_units]
    for unit, pool, fixed_test, ks in units:
        t0 = time.time()
        test_all = fixed_test if fixed_test is not None else pool
        if a.max_test:
            test_all = test_all[: a.max_test]
        key0 = f"{unit}|0|base|0|none"
        if key0 not in res["results"]:
            res["results"][key0] = {"unit": unit, "K": 0, "selection": "base", "draw": 0, "method": "none", "adapt_ids": [], **evaluate(base.model(), test_all, base)}
            save()
        orders = {}
        if "coverage" in a.selections:
            orders["coverage"] = coverage_order(pool, max(ks))
        if nov is not None:
            sc = nov.scores(base, pool)
            orders["novelty"] = sorted(sc, key=lambda i: (-sc[i], i))[: max(ks)]
        by_id = {r["id"]: r for r in pool}
        for k in ks:
            picks = []
            for sel in a.selections:
                if sel == "random":
                    for d in range(a.draws):
                        picks.append(("random", d, random.Random(f"{a.seed}|{unit}|{k}|{d}").sample(sorted(by_id), k)))
                else:
                    picks.append((sel, 0, orders[sel][:k]))
            for sel, d, ids in picks:
                for method in a.methods:
                    key = f"{unit}|{k}|{sel}|{d}|{method}"
                    if key in res["results"]:
                        continue
                    model = adapt(base, [by_id[i] for i in ids], method, a, key, ckpt_path)
                    test = test_all if fixed_test is not None else [r for r in test_all if r["id"] not in set(ids)]
                    ev = evaluate(model, test, base)
                    res["results"][key] = {"unit": unit, "K": k, "selection": sel, "draw": d, "method": method, "adapt_ids": ids, **ev}
                    save()
                    print(f"{key:48s} test {len(test):4d}  CER {100 * ev['cer']:5.1f} %  (base {100 * res['results'][key0]['cer']:5.1f} %)  q->f {ev['q_to_f']}", flush=True)
        print(f"unit {unit} done in {time.time() - t0:.0f}s", flush=True)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print("done", flush=True)


if __name__ == "__main__":
    main()
