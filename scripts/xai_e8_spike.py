"""E8 spike: the frozen CRNN-CTC encoder read by a small LLM through a projector. Runs on the 3090.

    python xai_e8_spike.py --train-data aligner_data.pkl --eval-data eval_data.pkl --run aligner_run --out e8_out \
        [--llm Qwen/Qwen3-0.6B] [--stage1-epochs 1] [--stage2-epochs 2]

Encoder: the trained aligner (xai_train_aligner.CRNN), frozen; its BiLSTM output (512 numbers per 4-pixel column) is
average-pooled by 2 along the line, then a two-layer projector maps each column to the LLM's embedding size.
Sequence: [projected columns] + "\\nالنص:" + label + end-of-text. Loss on the label and the end token only.
Stage 1 trains the projector alone; stage 2 adds LoRA (r = 16) on all linear layers of the LLM. Checkpoints every 500 steps
and at the end of each stage (resumable per stage). Then greedy decoding on Muharaf held-out, NAKBA (test and blind) and KHATT,
next to the CRNN's own greedy reading of the same lines. Writes results.json (CER per set and model, fluent-but-wrong word
share) and per_line.json (both readings per line, for the gallery).
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import CRNN, edit_distance, greedy, normalise  # noqa: E402

PROMPT = "\nالنص:"


class Encoder(nn.Module):
    def __init__(self, crnn: CRNN):
        super().__init__()
        self.m = crnn

    @torch.no_grad()
    def forward(self, x):  # B,1,64,W -> B,T',512
        f = self.m.cnn(x)
        b, c, h, t = f.shape
        f = f.permute(3, 0, 1, 2).reshape(t, b, c * h)
        f, _ = self.m.rnn(F.relu(self.m.proj(f)))  # T,B,512
        f = f.permute(1, 2, 0)  # B,512,T
        return F.avg_pool1d(f, 2, ceil_mode=True).transpose(1, 2)  # B,T/2,512


def images(rows):
    ws = [r["img"].shape[1] for r in rows]
    wmax = int(math.ceil(max(ws) / 8) * 8)
    x = np.ones((len(rows), 1, 64, wmax), np.float32)
    for i, r in enumerate(rows):
        x[i, 0, :, : ws[i]] = normalise(r["img"])
    return torch.from_numpy(x).cuda(), [math.ceil(math.ceil(w / 4) / 2) for w in ws]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--eval-data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--stage1-epochs", type=int, default=1)
    ap.add_argument("--stage2-epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-train", type=int, default=0, help="smoke test: use this many training lines")
    ap.add_argument("--max-eval", type=int, default=0, help="smoke test: this many lines per evaluation set")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    crnn = CRNN(len(chars) + 1).cuda().eval()
    crnn.load_state_dict(torch.load(a.run / a.ckpt, map_location="cuda"))
    enc = Encoder(crnn)
    tok = AutoTokenizer.from_pretrained(a.llm)
    llm = AutoModelForCausalLM.from_pretrained(a.llm, torch_dtype=torch.bfloat16).cuda()
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    hid = llm.config.hidden_size
    proj = nn.Sequential(nn.Linear(512, hid), nn.GELU(), nn.Linear(hid, hid)).cuda()
    eos = tok.eos_token_id
    prompt_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    emb = llm.get_input_embeddings()

    rows = pickle.load(open(a.train_data, "rb"))["rows"]
    train = [r for r in rows if r["role"] in ("train", "panel")]
    held = [r for r in rows if r["role"] == "heldout"]
    del rows
    evrows = held + pickle.load(open(a.eval_data, "rb"))["rows"]
    if a.max_train:
        random.Random(0).shuffle(train)
        train = train[: a.max_train]
    if a.max_eval:
        evrows = [r for t in sorted({r["dataset"] for r in evrows}) for r in [x for x in evrows if x["dataset"] == t][: a.max_eval]]
    vocab = {w for r in train for w in r["label"].split()}
    print(f"train {len(train)} eval {len(evrows)} llm {a.llm} hidden {hid}", flush=True)

    def build(batch):
        x, tl = images(batch)
        v = proj(enc(x).float()).to(torch.bfloat16)  # B,T,hid
        seqs, labs = [], []
        for i, r in enumerate(batch):
            ids = tok(r["label"], add_special_tokens=False)["input_ids"][:200] + [eos]
            p = torch.tensor(prompt_ids + ids, device="cuda")
            seqs.append(torch.cat([v[i, : tl[i]], emb(p)]))
            labs.append(torch.cat([torch.full((tl[i] + len(prompt_ids),), -100, device="cuda"), torch.tensor(ids, device="cuda")]))
        L = max(s.shape[0] for s in seqs)
        E = torch.zeros(len(seqs), L, hid, device="cuda", dtype=torch.bfloat16)
        Y = torch.full((len(seqs), L), -100, device="cuda")
        M = torch.zeros(len(seqs), L, device="cuda", dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            E[i, : len(s)], Y[i, : len(s)], M[i, : len(s)] = s, l, 1
        return E, Y, M

    def run_stage(name, epochs, params, lr):
        ck = a.out / f"{name}.pt"
        if ck.exists():
            st = torch.load(ck, map_location="cuda")
            proj.load_state_dict(st["proj"])
            if st.get("lora") is not None:
                llm.load_state_dict(st["lora"], strict=False)
            print(name, "loaded from checkpoint", flush=True)
            return
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        order = sorted(range(len(train)), key=lambda i: train[i]["img"].shape[1])
        batches = [order[k : k + a.bs] for k in range(0, len(order), a.bs)]
        total = epochs * len(batches)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
        step, t0 = 0, time.time()
        for ep in range(epochs):
            random.Random(ep).shuffle(batches)
            run = 0.0
            for bi, b in enumerate(batches):
                E, Y, M = build([train[i] for i in b])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = llm(inputs_embeds=E, attention_mask=M, labels=Y)
                out.loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                run += float(out.loss); step += 1
                if step % 100 == 0:
                    print(f"{name} epoch {ep + 1}/{epochs} step {step}/{total} loss {run / 100:.3f} {time.time() - t0:.0f}s", flush=True)
                    run = 0.0
        lora = {k: v for k, v in llm.state_dict().items() if "lora_" in k} if name == "stage2" else None
        torch.save({"proj": proj.state_dict(), "lora": lora}, ck)

    if not a.eval_only:
        for p in llm.parameters():
            p.requires_grad_(False)
        run_stage("stage1", a.stage1_epochs, list(proj.parameters()), 1e-3)
    llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if not a.eval_only:
        run_stage("stage2", a.stage2_epochs, [p for p in llm.parameters() if p.requires_grad] + list(proj.parameters()), 2e-4)
    else:
        st = torch.load(a.out / "stage2.pt", map_location="cuda")
        proj.load_state_dict(st["proj"]); llm.load_state_dict(st["lora"], strict=False)

    # ---------------- evaluation
    llm.eval(); llm.config.use_cache = True
    per, res = {}, {}
    for tag in sorted({r["dataset"] for r in evrows}):
        rs = [r for r in evrows if r["dataset"] == tag]
        for k in range(0, len(rs), 16):
            b = rs[k : k + 16]
            x, tl = images(b)
            with torch.no_grad():
                logits = crnn(x)
                crnn_h = ["".join(itos.get(t, "?") for t in h)[::-1] for h in greedy(logits.float(), torch.tensor([r["img"].shape[1] for r in b]))]
                v = proj(enc(x).float()).to(torch.bfloat16)
                P = emb(torch.tensor(prompt_ids, device="cuda"))
                seqs = [torch.cat([v[i, : tl[i]], P]) for i in range(len(b))]
                L = max(s.shape[0] for s in seqs)
                E = torch.zeros(len(b), L, hid, device="cuda", dtype=torch.bfloat16)
                M = torch.zeros(len(b), L, device="cuda", dtype=torch.long)
                for i, s in enumerate(seqs):  # left padding for generation
                    E[i, L - len(s) :], M[i, L - len(s) :] = s, 1
                gen = llm.generate(inputs_embeds=E, attention_mask=M, max_new_tokens=160, do_sample=False, eos_token_id=eos, pad_token_id=eos)
            for r, ch, g in zip(b, crnn_h, gen):
                e8 = tok.decode(g, skip_special_tokens=True).strip()
                per.setdefault(tag, {})[r["id"]] = {"gt": r["label"], "crnn": ch, "e8": e8}
        d = per[tag]
        n = sum(len(v["gt"]) for v in d.values())
        out = {"lines": len(d)}
        for m in ("crnn", "e8"):
            out[f"cer_{m}"] = sum(edit_distance(list(v[m]), list(v["gt"])) for v in d.values()) / n
            wrong = inv = 0  # fluent-but-wrong: output words absent from the truth at that place but present in the training vocabulary
            for v in d.values():
                g, h = v["gt"].split(), v[m].split()
                for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, g, h, autojunk=False).get_opcodes():
                    if op in ("replace", "insert"):
                        wrong += j2 - j1
                        inv += sum(w in vocab for w in h[j1:j2])
            out[f"wrong_words_{m}"] = wrong
            out[f"fluent_wrong_share_{m}"] = inv / max(wrong, 1)
        res[tag] = out
        print(tag, json.dumps(out), flush=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (a.out / "per_line.json").write_text(json.dumps(per, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
