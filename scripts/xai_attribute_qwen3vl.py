"""Per-token pixel attribution for Qwen3-VL-4B (+ our LoRA adapter) on OCR lines.

Throwaway research spike (runs on the 3090 box; needs torch + transformers>=4.57 + peft).

    python xai_attribute_qwen3vl.py --lines LINES_DIR --adapter ADAPTER_DIR --out OUT_DIR

For each line image: greedy-decode the transcription, then for a handful of the
generated tokens (those containing the target letters first) compute a map over the
image with four methods:

  attn      raw attention to image tokens, averaged over heads and layers
  attn_rel  the same divided by the mean over all generated tokens (removes sinks / borders)
  gxi       gradient x (input - blank paper) w.r.t. the *pixel values* (true pixel attribution)

and two sanity checks: a deletion test (mask the top-k % patches by each method vs
random patches; the token's log-prob should drop more for a faithful map) and an RTL
test (the attribution centroid should move right-to-left along the generated tokens).

Writes OUT/<line>/{orig.png, ov_<method>_<t>.png, tokens.json} and OUT/metrics.json.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# arabic_ocr_bench.prompting.LINE_V2, inlined so the script is self-contained
PROMPT = (
    "This image contains a single line of handwritten Arabic.\n"
    "Transcribe exactly that one line, as plain text.\n"
    "Preserve harakat that are actually written; do not invent any.\n"
    "Do NOT add HTML tags, Markdown, bullet points, or formatting.\n"
    "Do NOT repeat words or phrases. If a word is illegible, give your best single guess.\n"
    "Output only the transcription — nothing before or after it."
)
TARGETS = {"alef": set("اآأإ"), "lam": set("ل"), "jim": set("ج"), "ha": set("ح"), "kha": set("خ")}
ALL_TARGET = set().union(*TARGETS.values())
METHODS = ["attn", "attn_rel", "gxi"]
PIXEL_METHODS = {"gxi"}
P, M = 16, 2  # vision patch size, spatial merge


# ---------------------------------------------------------------- geometry
def rows_to_grid(rows: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """(N, ...) per-patch values in processor row order -> (gh, gw, ...) spatial grid."""
    x = rows.reshape(gh // M, gw // M, M, M, *rows.shape[1:])
    x = x.permute(0, 2, 1, 3, *range(4, x.ndim))
    return x.reshape(gh, gw, *rows.shape[1:])


def grid_to_rows(grid: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    x = grid.reshape(gh // M, M, gw // M, M, *grid.shape[2:])
    x = x.permute(0, 2, 1, 3, *range(4, x.ndim))
    return x.reshape(gh * gw, *grid.shape[2:])


def pixel_map(attr_rows: torch.Tensor, gh: int, gw: int) -> np.ndarray:
    """(N, C*T*P*P) per-row attribution -> (gh*P, gw*P) pixel map (positive part of channel sum)."""
    a = attr_rows.reshape(-1, 3, 2, P, P).sum((1, 2))  # (N, P, P)
    g = rows_to_grid(a, gh, gw)  # (gh, gw, P, P)
    return g.permute(0, 2, 1, 3).reshape(gh * P, gw * P).clamp(min=0).float().cpu().numpy()


def patch_scores(pix: np.ndarray, gh: int, gw: int) -> np.ndarray:
    return pix.reshape(gh, P, gw, P).sum((1, 3))


def merged_to_patch(m: np.ndarray) -> np.ndarray:
    return np.kron(m, np.ones((M, M), dtype=m.dtype))


# ---------------------------------------------------------------- rendering
def overlay(img: Image.Image, amap: np.ndarray) -> Image.Image:
    from matplotlib import cm

    h, w = img.size[1], img.size[0]
    amap = np.asarray(Image.fromarray(amap.astype(np.float32)).resize((w, h), Image.BILINEAR))
    hi = np.percentile(amap, 99.5) or 1.0
    a = np.clip(amap / hi, 0, 1)
    heat = (cm.inferno(a)[..., :3] * 255).astype(np.float32)
    base = np.asarray(img.convert("L").convert("RGB"), dtype=np.float32) * 0.6 + 60
    alpha = (a ** 0.7 * 0.85)[..., None]
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8))


# ---------------------------------------------------------------- model
def load(base: str, adapter: str | None):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(base)
    proc.image_processor.max_pixels = 1_003_520
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def build_inputs(proc, img: Image.Image):
    conv = [
        {"role": "system", "content": [{"type": "text", "text": PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    return proc(text=[text], images=[img], return_tensors="pt").to("cuda")


def token_pieces(tok, ids: list[int]) -> list[str]:
    out, prev = [], ""
    for i in range(len(ids)):
        cur = tok.decode(ids[: i + 1])
        stable = cur[:-1] if cur.endswith("�") else cur
        out.append(stable[len(prev):] if stable.startswith(prev) else "")
        prev = stable if stable.startswith(prev) else prev
    return out


# ---------------------------------------------------------------- attribution methods
class Ctx:
    def __init__(self, model, proc, inputs, full_ids, n_prompt):
        self.model, self.proc, self.inputs = model, proc, inputs
        self.ids = full_ids  # (1, S) prompt + generation
        self.n_prompt = n_prompt
        self.gh, self.gw = [int(v) for v in inputs["image_grid_thw"][0][1:]]
        self.img_mask = (full_ids[0] == model.config.image_token_id)
        self.px = inputs["pixel_values"].float()
        self.mask = torch.ones_like(full_ids)

    def forward(self, px: torch.Tensor, attn=False):
        kw = dict(input_ids=self.ids, attention_mask=self.mask, pixel_values=px,
                  image_grid_thw=self.inputs["image_grid_thw"], mm_token_type_ids=self.img_mask.long()[None])
        return self.model(**kw, output_attentions=attn, use_cache=False)

    def logprob(self, logits: torch.Tensor, p: int) -> torch.Tensor:
        return torch.log_softmax(logits[0, p - 1].float(), -1)[self.ids[0, p]]


def attention_maps(ctx: Ctx, positions: list[int]):
    """Raw and relative attention from each generated token to the image tokens.

    Needs eager attention to return weights; everything else runs on sdpa, so the model is
    switched for this one pass only. `_rel` divides by the mean over all generated tokens,
    which removes sinks and border patches that every token attends to.
    """
    ctx.model.set_attn_implementation("eager")
    try:
        with torch.no_grad():
            out = ctx.forward(ctx.px, attn=True)
    finally:
        ctx.model.set_attn_implementation("sdpa")
    avg = torch.stack([a[0].float().mean(0) for a in out.attentions]).mean(0)  # (S, S), head+layer mean
    del out
    S = avg.shape[0]
    gen_rows = torch.arange(ctx.n_prompt - 1, S - 1)
    base = avg[gen_rows][:, ctx.img_mask].mean(0)
    base = base + 0.05 * base.mean()
    maps = {"attn": {}, "attn_rel": {}}
    for p in positions:
        v = avg[p - 1][ctx.img_mask]
        for key, val in (("attn", v), ("attn_rel", v / base)):
            maps[key][p] = merged_to_patch(val.reshape(ctx.gh // M, ctx.gw // M).cpu().numpy())
    return maps


def grad_maps(ctx: Ctx, positions: list[int]):
    """Gradient x (input - paper) per generated token, at pixel resolution.

    Baseline = blank paper (median patch), not zero: normalised pixels are + for paper and - for ink,
    so with a zero baseline the positive part of grad*input kept the paper and dropped the ink.
    """
    base = ctx.px.median(0).values.expand_as(ctx.px)
    diff = ctx.px - base
    x = ctx.px.clone().requires_grad_(True)
    out = {}
    with torch.enable_grad():
        logits = ctx.forward(x).logits
        for p in positions:
            (g,) = torch.autograd.grad(ctx.logprob(logits, p), x, retain_graph=True)
            out[p] = pixel_map(g * diff, ctx.gh, ctx.gw)
    del logits
    return out


# ---------------------------------------------------------------- sanity checks
def deletion(ctx: Ctx, p: int, score_grid: np.ndarray, fracs=(0.1,)) -> dict:
    n = ctx.gh * ctx.gw
    base = ctx.logprob(ctx.forward(ctx.px).logits, p).item()
    order = np.argsort(-score_grid.ravel(), kind="stable")
    fill = ctx.px.median(0).values
    res = {}
    for f in fracs:
        k = max(1, int(round(f * n)))
        sel = np.zeros(n, dtype=bool)
        sel[order[:k]] = True
        grid = torch.from_numpy(sel.reshape(ctx.gh, ctx.gw)).to(ctx.px.device)
        rows = grid_to_rows(grid, ctx.gh, ctx.gw)
        px = ctx.px.clone()
        px[rows] = fill  # median patch = background paper, so the masked patch looks like blank page
        res[f] = base - ctx.logprob(ctx.forward(px).logits, p).item()
    return res


def conc(score_grid: np.ndarray, frac: float = 0.1) -> float:
    """Share of the total attribution held by the top `frac` of patches (1 = pinpoint, ~frac = diffuse)."""
    v = np.sort(score_grid.ravel())[::-1]
    return float(v[: max(1, int(round(frac * len(v))))].sum() / max(v.sum(), 1e-12))


def error_flags(pieces: list[str], gt: str, hyp: str):
    """Per generated token: is any of its characters wrong vs the GT, and the GT text it aligns to."""
    sm = difflib.SequenceMatcher(None, hyp, gt, autojunk=False)
    bad = [False] * len(hyp)
    gt_of = [""] * len(hyp)
    missing = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                bad[i] = True
                gt_of[i] = gt[j1:j2]
        elif tag == "insert":
            missing += j2 - j1
    out, pos = [], 0
    for pc in pieces:
        sl = range(pos, pos + len(pc))
        out.append((any(bad[i] for i in sl), "".join(dict.fromkeys(gt_of[i] for i in sl if gt_of[i]))))
        pos += len(pc)
    return out, missing


def ink_mask(img: Image.Image) -> np.ndarray:
    """Boolean ink mask (Otsu on grey) ignoring the outer 8 % top/bottom bands, where scans carry black borders."""
    g = np.asarray(img.convert("L"))
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    tot, sum_all, w0, s0, best, thr = hist.sum(), (np.arange(256) * hist).sum(), 0.0, 0.0, -1.0, 127
    for t in range(256):
        w0 += hist[t]
        if w0 == 0 or w0 == tot:
            continue
        s0 += t * hist[t]
        b = w0 * (tot - w0) * (s0 / w0 - (sum_all - s0) / (tot - w0)) ** 2
        if b > best:
            best, thr = b, t
    m = g < thr
    k = int(0.08 * g.shape[0])
    m[:k], m[-k:] = False, False
    return m


def ink_enrichment(pix: np.ndarray, ink: np.ndarray) -> float:
    """Attribution share on ink pixels / ink share of the valid area. 1 = no better than chance; >1 = lands on ink."""
    valid = np.ones_like(ink)
    k = int(0.08 * ink.shape[0])
    valid[:k], valid[-k:] = False, False
    mass = pix[valid].sum()
    if mass <= 0 or ink[valid].mean() <= 0:
        return float("nan")
    return float((pix[ink].sum() / mass) / ink[valid].mean())


def centroid_x(score_grid: np.ndarray) -> float:
    col = score_grid.sum(0)
    return float((col * np.arange(len(col))).sum() / max(col.sum(), 1e-12) / len(col))


# ---------------------------------------------------------------- main
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=Path, required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-width", type=int, default=1024, help="downscale wider lines first (patch count drives cost)")
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--del-methods", default=",".join(METHODS), help="comma list; deletion test is the slow part")
    ap.add_argument("--select", choices=["targets", "errors"], default="targets")
    a = ap.parse_args()
    proc, model = load(a.base, a.adapter)
    tok = proc.tokenizer
    rng = np.random.default_rng(0)
    manifest = json.loads((a.lines / "manifest.json").read_text(encoding="utf-8"))
    if a.limit:
        manifest = manifest[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    agg: dict = {}
    by_err: dict = {}
    per_line = []
    for li, item in enumerate(manifest):
        t0 = time.time()
        img = Image.open(a.lines / item["file"]).convert("RGB")
        if img.width > a.max_width:
            img = img.resize((a.max_width, max(32, round(img.height * a.max_width / img.width))), Image.BICUBIC)
        inputs = build_inputs(proc, img)
        n_prompt = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        ids = gen[:, :]
        end = ids.shape[1]
        while end > n_prompt and ids[0, end - 1].item() in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            end -= 1
        ids = ids[:, :end]
        gen_ids = ids[0, n_prompt:].tolist()
        pieces = token_pieces(tok, gen_ids)
        hyp = "".join(pieces)
        ctx = Ctx(model, proc, inputs, ids, n_prompt)
        # choose tokens: target-letter ones first
        cand = [(i, pieces[i]) for i in range(len(gen_ids)) if pieces[i].strip()]
        flags, n_missing = error_flags(pieces, item["gt"], hyp)
        if a.select == "errors":
            err = [i for i, _ in cand if flags[i][0]][: a.max_tokens * 2 // 3]
            ok = [i for i, _ in cand if not flags[i][0]]
            step = max(1, len(ok) // max(1, a.max_tokens - len(err)))
            chosen = sorted(err + ok[::step][: a.max_tokens - len(err)])
        else:
            pri = [i for i, s in cand if ALL_TARGET & set(s)] + [i for i, s in cand if not ALL_TARGET & set(s)]
            chosen = sorted(pri[: a.max_tokens])
        positions = [n_prompt + i for i in chosen]
        t1 = time.time()
        maps = attention_maps(ctx, positions)
        t2 = time.time()
        maps["gxi"] = grad_maps(ctx, positions)
        t3 = time.time()
        print(f"   grid {ctx.gh}x{ctx.gw}  decode {t1 - t0:.0f}s  attn {t2 - t1:.0f}s  grad {t3 - t2:.0f}s", flush=True)
        grids = {m: {p: (patch_scores(v, ctx.gh, ctx.gw) if m in PIXEL_METHODS else v) for p, v in maps[m].items()} for m in METHODS}
        d = a.out / f"{li:03d}"
        d.mkdir(exist_ok=True)
        rimg = img.resize((ctx.gw * P, ctx.gh * P), Image.BICUBIC)
        rimg.save(d / "orig.png")
        toks = []
        ink = ink_mask(rimg)
        with torch.no_grad():
            full = ctx.forward(ctx.px).logits
            lps = {p: ctx.logprob(full, p).item() for p in positions}
            del full
        for i, p in zip(chosen, positions):
            row = {"t": i, "id": gen_ids[i], "text": pieces[i], "targets": sorted(k for k, s in TARGETS.items() if s & set(pieces[i])),
                   "err": flags[i][0], "gt_span": flags[i][1], "lp": lps[p], "conc": {}, "ink": {}}
            row["deletion"] = {}
            n = ctx.gh * ctx.gw
            for m in METHODS:
                grid = grids[m][p]
                pm = maps[m][p] if m in PIXEL_METHODS else np.kron(grid, np.ones((P, P)))
                overlay(rimg, pm).save(d / f"ov_{m}_{i}.png")
                if m in a.del_methods.split(","):
                    row["deletion"][m] = deletion(ctx, p, grid)
                row.setdefault("cx", {})[m] = centroid_x(grid)
                row["conc"][m] = conc(grid)
                row["ink"][m] = ink_enrichment(pm, ink)
            rnd = [deletion(ctx, p, rng.random((ctx.gh, ctx.gw))) for _ in range(2)]
            row["deletion"]["random"] = {f: float(np.mean([r[f] for r in rnd])) for f in rnd[0]}
            toks.append(row)
        # RTL test: Spearman between token order and centroid x (expect negative)
        rtl = {}
        for m in METHODS:
            xs = np.array([r["cx"][m] for r in toks])
            ts = np.array([r["t"] for r in toks])
            if len(xs) >= 4 and xs.std() > 0:
                rx, rt = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ts))
                rtl[m] = float(np.corrcoef(rx, rt)[0, 1])
        (d / "tokens.json").write_text(json.dumps({"id": item["id"], "dataset": item["dataset"], "gt": item["gt"], "hyp": hyp, "stored_hyp": item.get("stored_hyp"), "group": item.get("group"), "cer": item.get("cer"), "missing_gt_chars": n_missing,
                                                    "grid": [ctx.gh, ctx.gw], "tokens": toks, "rtl": rtl}, ensure_ascii=False, indent=1), encoding="utf-8")
        per_line.append({"line": li, "id": item["id"], "dataset": item["dataset"], "rtl": rtl})
        for r in toks:
            for m in [k for k in METHODS if k in r["deletion"]] + ["random"]:
                for f, v in r["deletion"][m].items():
                    agg.setdefault(m, {}).setdefault(str(f), []).append(v)
            g = by_err.setdefault("err" if r["err"] else "ok", {"lp": [], **{f"{m}_{k}": [] for m in METHODS for k in ("conc", "del0.1", "ink")}})
            g["lp"].append(r["lp"])
            for m in METHODS:
                g[f"{m}_conc"].append(r["conc"][m])
                g[f"{m}_ink"].append(r["ink"][m])
                if m in r["deletion"]:
                    g[f"{m}_del0.1"].append(r["deletion"][m][0.1])
        print(f"[{li + 1}/{len(manifest)}] {item['id']} {len(toks)} tokens {time.time() - t0:.0f}s  hyp={hyp[:40]!r}", flush=True)
        torch.cuda.empty_cache()
    summary = {m: {f: {"mean_drop": float(np.mean(v)), "n": len(v)} for f, v in fs.items()} for m, fs in agg.items()}
    (a.out / "metrics.json").write_text(json.dumps({"deletion": summary, "by_err": {g: {k: {"mean": float(np.nanmean(v)), "n": len(v)} for k, v in d.items()} for g, d in by_err.items()}, "per_line": per_line}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
