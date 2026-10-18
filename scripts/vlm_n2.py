"""GPU night 2 VLM jobs (checklist 7.2-7.6) on the 3090, through the patched benchmark client (0.2b: token cap + logprobs).

    python vlm_n2.py gen       --model qwen3vl4b-difflr --lines vlm_n2_lines.pkl --select kitab_khatt:stored,inkcrop,shear10 --out out\\gen\\M.jsonl
    python vlm_n2.py reliance  --model qwen3vl8b --lines vlm_n2_lines.pkl --select kitab_khatt:stored:200 ... --out out\\reliance\\M.jsonl
    python vlm_n2.py icl       --model qwen3vl4b-difflr --lines vlm_n2_lines.pkl --trials icl_trials.json --out out\\icl\\M.jsonl
    python vlm_n2.py selftest  (laptop or box; tiny random Qwen3-VL, no download beyond the processor files)

`--pkg DIR` puts the patched `arabic_ocr_bench` (src of the main checkout, branch research-characterization with
docs/research/patches/0002 applied) on sys.path; the box copy lives in xai\\vlm_n2\\pkg.

gen       one reading per line with arabic_ocr_bench's HFLocalClient exactly as the benchmark runs it (prompt line-v2, image
          capped at 1024, greedy with repetition_penalty 1.15 and no_repeat_ngram_size 3, max_new_tokens 160), plus
          finish_reason / hit_token_cap and per-token log-probabilities from the raw logits. An adapter is merged in bf16
          after loading, as the suite's merge phase did. Serves 7.3 (geometry), 7.5 (diff-LR bench) and 7.6 (resolution).
reliance  7.4 / X-A3: teacher-forced log-probability of the ground-truth tokens under three inputs: the line image, a blank
          page of the same size, and no image at all (text-only prompt). Per token, so letters can be grouped later.
icl       7.2: reference line(s) with their labels as earlier chat turns, then the test line (design writer-adaptation.md B).
          Conditions come from icl_trials.json (vlm_n2_export.py): cov, label (label only), other (other writer), rand, cov3.

Every loop appends one JSON line per sample and flushes it, and skips keys already in --out, so a rerun resumes.
--deadline-file holds a Unix time written when the queue starts (start + 8 h); --deadline-offset-h moves this job's own
stop earlier. A job past its stop finishes the current sample and exits 0, so the queue moves on.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path

MODELS = {  # name: (hub id, adapter dir or None)
    "qwen3vl4b-difflr": ("Qwen/Qwen3-VL-4B-Instruct", r"C:\Users\User\a-experiment\xai\adapter"),
    "qwen3vl4b": ("Qwen/Qwen3-VL-4B-Instruct", None),
    "qwen3vl8b": ("Qwen/Qwen3-VL-8B-Instruct", None),
}
REF_TEXT = "This line is by the same writer as the line you will be asked to read next.\n"
REF_LABEL_ONLY = ("Below is the transcription of a line by the same writer as the line you will be asked to read next "
                  "(the image of that line is not shown).")
STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")


# ---------------------------------------------------------------- helpers
def norm(t: str) -> str:
    return " ".join(STRIP.sub("", unicodedata.normalize("NFC", t or "")).split())


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def stop_time(a) -> float:
    if not a.deadline_file or not Path(a.deadline_file).exists():
        return float("inf")
    return float(Path(a.deadline_file).read_text().strip()) + 3600 * a.deadline_offset_h


def done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut by a crash; that sample is redone
    return keys


class Writer:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def put(self, row: dict):
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())


def load_lines(path: Path) -> dict:
    return {r["key"]: r for r in pickle.load(open(path, "rb"))["rows"]}


def select(lines: dict, specs: list[str]) -> list[dict]:
    """'dataset:variant1,variant2[:n]' -> rows, the first n line ids of the dataset (by id) in each variant, variant-major."""
    out = []
    for s in specs:
        parts = s.split(":")
        ds, variants = parts[0], parts[1].split(",")
        n = int(parts[2]) if len(parts) > 2 else None
        ids = sorted({r["id"] for r in lines.values() if r["dataset"] == ds})[:n]
        for v in variants:
            out += [lines[f"{i}|{v}"] for i in ids if f"{i}|{v}" in lines]
    return out


def image(row: dict):
    from PIL import Image

    return Image.open(io.BytesIO(row["png"])).convert("RGB")


def make_client(a):
    if a.pkg:
        sys.path.insert(0, a.pkg)
    from arabic_ocr_bench.models.hf_local_client import HFLocalClient

    hf_id, adapter = MODELS.get(a.model, (None, None))
    if a.hf_id:
        hf_id, adapter = a.hf_id, a.adapter
    elif a.adapter:
        adapter = a.adapter
    c = HFLocalClient(a.model, hf_id, arch="qwen3_vl", dtype=a.dtype, adapter_id=adapter, max_new_tokens=a.max_new_tokens,
                      logprobs=True)
    c._ensure_loaded()
    if c._model is None:
        raise SystemExit(f"model {a.model} did not load (see the [load] lines above)")
    if adapter:
        c._model = c._model.merge_and_unload().eval()
        print(f"[load] adapter merged ({adapter})", flush=True)
    return c


def progress(tag, n, total, t0, errs, chars):
    rate = (time.time() - t0) / max(1, n)
    cer = f"  live CER {errs / max(1, chars):.3f}" if chars else ""
    print(f"[{tag}] {n}/{total}  {rate:.1f} s/line{cer}  {time.strftime('%H:%M:%S')}", flush=True)


def reading_row(c, pred, base: dict) -> dict:
    from arabic_ocr_bench.models.hf_local_client import HFLocalClient  # noqa: F401  (package on sys.path)

    lps = pred.token_logprobs
    return {**base, "model": c.name, "raw_text": pred.raw_text, "cleaned_text": pred.cleaned_text, "error": pred.error,
            "latency_ms": round(pred.latency_ms, 1), "output_tokens": pred.output_tokens, "token_cap": c.max_new_tokens,
            "finish_reason": pred.finish_reason, "hit_token_cap": pred.finish_reason == "length",
            "token_logprobs": [round(x, 4) for x in lps] if lps else None,
            "mean_logprob": round(sum(lps) / len(lps), 4) if lps else None, "logprob_kind": pred.logprob_kind}


# ---------------------------------------------------------------- gen (7.3, 7.5, 7.6)
def cmd_gen(a):
    from arabic_ocr_bench.prompting import LINE_V2  # noqa: E402

    lines = load_lines(a.lines)
    todo = select(lines, a.select)
    out = Path(a.out)
    seen = done_keys(out)
    todo = [r for r in todo if r["key"] not in seen]
    print(f"[gen] {a.model}: {len(todo)} to do, {len(seen)} already in {out}", flush=True)
    if not todo:
        return
    stop = stop_time(a)
    c = make_client(a)
    w, t0, errs, chars = Writer(out), time.time(), 0, 0
    for n, r in enumerate(todo, 1):
        if time.time() > stop:
            print(f"[gen] deadline reached after {n - 1} lines; stopping", flush=True)
            break
        pred = c.transcribe(image(r), LINE_V2)
        w.put(reading_row(c, pred, {"key": r["key"], "id": r["id"], "dataset": r["dataset"], "variant": r["variant"],
                                    "prompt_version": "line-v2", "ground_truth": r["gt"]}))
        g = norm(r["gt"])
        errs, chars = errs + lev(norm(pred.cleaned_text), g), chars + len(g)
        if n % 25 == 0 or n == len(todo):
            progress("gen " + a.model, n, len(todo), t0, errs, chars)


# ---------------------------------------------------------------- reliance (7.4)
def gt_logprobs(c, img, gt: str, use_image: bool):
    import torch
    from arabic_ocr_bench.imaging import cap_image
    from arabic_ocr_bench.prompting import LINE_V2

    proc, model = c._processor, c._model
    content = ([{"type": "image"}] if use_image else []) + [{"type": "text", "text": LINE_V2}]
    prompt = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    kw = {"images": [cap_image(img, 1024)]} if use_image else {}
    p_ids = proc(text=[prompt], return_tensors="pt", **kw)["input_ids"][0]
    full = proc(text=[prompt + gt], return_tensors="pt", **kw).to(c._device)
    ids = full["input_ids"][0]
    n0 = len(p_ids)
    aligned = bool(len(ids) > n0 and torch.equal(ids[:n0].cpu(), p_ids))
    with torch.no_grad():
        logits = model(**full).logits[0, n0 - 1 : -1].float()
    lp = torch.log_softmax(logits, -1).gather(1, ids[n0:, None]).squeeze(1).cpu().tolist()
    toks = [proc.tokenizer.decode([int(t)]) for t in ids[n0:]]
    return lp, toks, aligned


def cmd_reliance(a):
    from PIL import Image

    lines = load_lines(a.lines)
    todo = select(lines, a.select)
    out = Path(a.out)
    seen = done_keys(out)
    todo = [r for r in todo if r["key"] not in seen]
    print(f"[reliance] {a.model}: {len(todo)} to do, {len(seen)} already in {out}", flush=True)
    if not todo:
        return
    stop = stop_time(a)
    c = make_client(a)
    w, t0 = Writer(out), time.time()
    for n, r in enumerate(todo, 1):
        if time.time() > stop:
            print(f"[reliance] deadline reached after {n - 1} lines; stopping", flush=True)
            break
        img, gt = image(r), (r["gt"] or "").strip()
        row = {"key": r["key"], "id": r["id"], "dataset": r["dataset"], "variant": r["variant"], "model": a.model,
               "ground_truth": gt, "error": None}
        try:
            for cond in ("image", "blank", "none"):
                im = Image.new("RGB", img.size, "white") if cond == "blank" else img
                lp, toks, ok = gt_logprobs(c, im, gt, cond != "none")
                row[cond] = {"logprobs": [round(x, 4) for x in lp], "sum": round(sum(lp), 3), "n": len(lp), "prefix_ok": ok}
            row["tokens"] = toks
        except Exception as e:  # noqa: BLE001  (an OOM on one long line must not end the job)
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            import torch

            torch.cuda.empty_cache()
        w.put(row)
        if n % 25 == 0 or n == len(todo):
            progress("reliance " + a.model, n, len(todo), t0, 0, 0)


# ---------------------------------------------------------------- icl (7.2)
def icl_generate(c, refs: list[tuple], test_img, label_only: bool):
    """refs: [(image, label)]. Returns the GenOut of the final turn."""
    import torch
    from arabic_ocr_bench.imaging import cap_image
    from arabic_ocr_bench.prompting import LINE_V2

    msgs, imgs = [], []
    for im, lab in refs:
        if label_only:
            msgs.append({"role": "user", "content": [{"type": "text", "text": REF_LABEL_ONLY}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": REF_TEXT + LINE_V2}]})
            imgs.append(cap_image(im, 1024))
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": lab}]})
    msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": LINE_V2}]})
    imgs.append(cap_image(test_img, 1024))
    chat = c._processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = c._processor(text=[chat], images=imgs, return_tensors="pt").to(c._device)
    with torch.no_grad():
        out = c._model.generate(**inputs, **c._gen_kwargs())
    n = inputs["input_ids"].shape[1]
    gen = getattr(out, "sequences", out)[0][n:]
    return c._gen_out(out, n, c._processor.decode(gen, skip_special_tokens=True).strip())


def cmd_icl(a):
    from arabic_ocr_bench.models.base import Prediction
    from arabic_ocr_bench.models.litellm_client import clean_output

    lines = load_lines(a.lines)
    trials = json.loads(Path(a.trials).read_text(encoding="utf-8"))
    if a.units:
        trials = [t for t in trials if t["unit"].split(":")[0] in a.units]
    out = Path(a.out)
    seen = done_keys(out)
    todo = [t for t in trials if t["key"] not in seen]
    print(f"[icl] {a.model}: {len(todo)} to do, {len(seen)} already in {out}", flush=True)
    if not todo:
        return
    stop = stop_time(a)
    c = make_client(a)
    w, t0, errs, chars = Writer(out), time.time(), 0, 0
    for n, t in enumerate(todo, 1):
        if time.time() > stop:
            print(f"[icl] deadline reached after {n - 1} trials; stopping", flush=True)
            break
        test = lines[t["test"] + "|stored"]
        refs = [(image(lines[i + "|stored"]), (lines[i + "|stored"]["gt"] or "").strip()) for i in t["refs"]]
        t1 = time.time()
        try:
            g = icl_generate(c, refs, image(test), t["label_only"])
            pred = Prediction(raw_text=g.text, cleaned_text=clean_output(g.text), output_tokens=g.n_tokens or 0,
                              finish_reason=g.finish_reason, token_logprobs=g.token_logprobs, logprob_kind=g.logprob_kind,
                              latency_ms=(time.time() - t1) * 1000)
        except Exception as e:  # noqa: BLE001
            import torch

            torch.cuda.empty_cache()
            pred = Prediction(error=f"{type(e).__name__}: {e}"[:300], latency_ms=(time.time() - t1) * 1000)
        w.put(reading_row(c, pred, {"key": t["key"], "id": t["test"], "unit": t["unit"], "cond": t["cond"], "refs": t["refs"],
                                    "label_only": t["label_only"], "prompt_version": "line-v2+icl",
                                    "ground_truth": test["gt"]}))
        gt = norm(test["gt"])
        errs, chars = errs + lev(norm(pred.cleaned_text), gt), chars + len(gt)
        if n % 25 == 0 or n == len(todo):
            progress("icl " + a.model, n, len(todo), t0, errs, chars)


# ---------------------------------------------------------------- selftest (laptop / box, tiny random model)
def cmd_selftest(a):
    """Builds a tiny random Qwen3-VL with the real Qwen3-VL processor, plus a LoRA on the same target modules as the
    diff-LR adapter, and runs gen / reliance / icl on --lines through the real code paths (CPU)."""
    import tempfile

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    src = a.processor or "Qwen/Qwen3-VL-4B-Instruct"
    tmp = Path(a.workdir or tempfile.mkdtemp(prefix="vlm_n2_selftest_"))
    cfg = AutoConfig.from_pretrained(src)
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, 2
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 2, 1, 32
    tc.rope_scaling = {**(tc.rope_scaling or {}), "mrope_section": [8, 4, 4]} if getattr(tc, "rope_scaling", None) else tc.rope_scaling
    vc.depth, vc.hidden_size, vc.intermediate_size, vc.num_heads, vc.out_hidden_size = 2, 64, 128, 2, 64
    vc.deepstack_visual_indexes = [0]
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg).to(torch.bfloat16)
    mdir = tmp / "tiny"
    model.save_pretrained(mdir)
    AutoProcessor.from_pretrained(src).save_pretrained(mdir)
    lora = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["proj", "q_proj", "down_proj", "k_proj", "up_proj",
                                                                               "o_proj", "qkv", "v_proj", "gate_proj"]))
    lora.save_pretrained(tmp / "adapter")
    print("tiny model + adapter in", tmp, flush=True)
    base = dict(pkg=a.pkg, lines=a.lines, dtype="float32", max_new_tokens=12, deadline_file=None, deadline_offset_h=0.0,
                hf_id=str(mdir), units=None)
    sel = [f"kitab_khatt:stored,h60_n8:2", "rasam2:stored:2"]
    for name, adapter in (("tiny-adapter", str(tmp / "adapter")), ("tiny", None)):
        ns = argparse.Namespace(**base, model=name, adapter=adapter)
        cmd_gen(argparse.Namespace(**vars(ns), select=sel, out=str(tmp / f"gen_{name}.jsonl")))
        cmd_gen(argparse.Namespace(**vars(ns), select=sel, out=str(tmp / f"gen_{name}.jsonl")))  # resume: nothing to do
    ns = argparse.Namespace(**base, model="tiny", adapter=None)
    cmd_reliance(argparse.Namespace(**vars(ns), select=["kitab_khatt:stored:2"], out=str(tmp / "rel.jsonl")))
    trials = json.loads(Path(a.trials).read_text(encoding="utf-8"))
    pick = [t for t in trials if t["cond"] == "label"][:1] + [t for t in trials if t["cond"] == "cov3"][:1]
    (tmp / "trials.json").write_text(json.dumps(pick))
    cmd_icl(argparse.Namespace(**vars(ns), trials=str(tmp / "trials.json"), out=str(tmp / "icl.jsonl")))
    for f in ("gen_tiny-adapter.jsonl", "rel.jsonl", "icl.jsonl"):
        rows = [json.loads(x) for x in (tmp / f).read_text(encoding="utf-8").splitlines()]
        r = rows[0]
        print(f, len(rows), "rows; first:", {k: (v if not isinstance(v, (list, dict)) else type(v).__name__) for k, v in r.items()},
              flush=True)
        assert all(not x.get("error") for x in rows), [x.get("error") for x in rows]
    rel = json.loads((tmp / "rel.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rel["image"]["prefix_ok"] and rel["none"]["prefix_ok"] and rel["image"]["n"] == rel["none"]["n"] > 0
    print("SELFTEST OK", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "reliance", "icl", "selftest"])
    ap.add_argument("--model", default="qwen3vl4b-difflr")
    ap.add_argument("--hf-id", help="override the hub id / local dir (selftest)")
    ap.add_argument("--adapter", help="override the adapter dir")
    ap.add_argument("--pkg", help="directory that holds the patched arabic_ocr_bench package")
    ap.add_argument("--lines", type=Path)
    ap.add_argument("--select", nargs="+", default=[])
    ap.add_argument("--trials")
    ap.add_argument("--units", nargs="+", help="icl: only these units (rasam2, nakba)")
    ap.add_argument("--out")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--deadline-file")
    ap.add_argument("--deadline-offset-h", type=float, default=0.0)
    ap.add_argument("--processor", help="selftest: processor source")
    ap.add_argument("--workdir", help="selftest: where the tiny model goes")
    a = ap.parse_args()
    if a.pkg:
        sys.path.insert(0, a.pkg)
    {"gen": cmd_gen, "reliance": cmd_reliance, "icl": cmd_icl, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
