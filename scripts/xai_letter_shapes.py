"""Letter-shape library and shape novelty from CTC letter spans (X-G10). Runs on the 3090.

    python xai_letter_shapes.py --train-data aligner_data.pkl --eval-data eval_data.pkl --run aligner_run --ckpt model_final.pt --out letters_out [--n-train 2000]

1. Forced-aligns every line (ground-truth label) with the trained CRNN-CTC, as xai_align.py does, and gives each letter
   a feature: the mean of the CNN output columns inside its span (1024 numbers), plus a 40 x 40 crop for display.
   The model's own greedy reading is aligned to the ground truth, so each letter is also marked read correctly or not.
2. The training bank is the letters of n Muharaf training lines. For every evaluation letter, novelty = cosine distance to
   the nearest training letter of the same character and positional form, divided by the median nearest-neighbour
   distance inside the bank for that character and form (1 = as typical as a training letter).
3. Writes summary.json (novelty by dataset, error rate by novelty band, per-letter numbers), allographs.npz (k-means
   shape clusters per common letter in the training bank, with medoid crops) and examples.npz (crops for the gallery).
"""
from __future__ import annotations

import argparse
import difflib
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from xai_align import char_spans, force_align  # noqa: E402
from xai_train_aligner import CRNN, greedy, normalise  # noqa: E402

RIGHT = set("اأإآٱدذرزوؤة")
LETTERS = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")


def form_of(label: str, i: int) -> str:
    ch = label[i]
    prev = label[i - 1] if i else " "
    nxt = label[i + 1] if i + 1 < len(label) else " "
    jp = prev in LETTERS and prev not in RIGHT and prev != "ء" and ch != "ء"
    jn = ch not in RIGHT and ch != "ء" and nxt in LETTERS and nxt != "ء"
    return {(False, False): "iso", (False, True): "ini", (True, True): "med", (True, False): "fin"}[(jp, jn)]


def crop(img: np.ndarray, px0: int, px1: int, size: int = 40) -> np.ndarray:
    w = img.shape[1]
    c = img[:, max(0, px0 - 3) : min(w, px1 + 3)]
    if c.shape[1] < 2:
        return np.full((size, size), 255, np.uint8)
    nw = max(1, round(c.shape[1] * size / 64))
    c = np.asarray(Image.fromarray(c).resize((nw, size), Image.LANCZOS))
    out = np.full((size, size), 255, np.uint8)
    if nw >= size:
        s = (nw - size) // 2
        out[:] = c[:, s : s + size]
    else:
        s = (size - nw) // 2
        out[:, s : s + nw] = c
    return out


def letters_of(model, rows, stoi, itos, tag_of, feat_model=None):
    feats, crops, meta = [], [], []
    for r in rows:
        arr = r["img"]
        w = arr.shape[1]
        x = np.ones((1, 1, 64, int(np.ceil(w / 4) * 4)), dtype=np.float32)
        x[0, 0, :, :w] = normalise(arr)
        xt = torch.from_numpy(x).cuda()
        with torch.no_grad():
            f = (feat_model or model).cnn(xt)  # 1,512,2,T; features may come from a second, separately trained model
            logits = model(xt)
        T = w // 4
        cols = f[0].permute(2, 0, 1).reshape(f.shape[3], -1)[:T].float()  # T,1024
        lp = logits[:T, 0].float().log_softmax(-1).cpu().numpy().astype(np.float64)
        lab = r["label"]
        vis = lab[::-1]
        tgt = [stoi[c] for c in vis if c in stoi]
        if len(tgt) != len(vis):
            continue
        path = force_align(lp, tgt)
        if path is None:
            continue
        spans = char_spans(path, len(tgt), T)
        hyp = "".join(itos.get(t, "?") for t in greedy(logits[:T].float(), torch.tensor([w]))[0])[::-1]
        ok = np.zeros(len(lab), bool)  # letters the greedy reading got right
        for op, i1, i2, _, _ in difflib.SequenceMatcher(None, lab, hyp, autojunk=False).get_opcodes():
            if op == "equal":
                ok[i1:i2] = True
        L = len(lab)
        for i, ch in enumerate(lab):
            if ch not in LETTERS:
                continue
            s0, s1 = (int(v) for v in spans[L - 1 - i])
            s1 = max(s1, s0 + 1)
            feats.append(cols[s0:s1].mean(0).half().cpu())
            crops.append(crop(arr, s0 * 4, s1 * 4))
            meta.append({"tag": tag_of(r), "id": r["id"], "ch": ch, "form": form_of(lab, i), "ok": bool(ok[i]), "i": i, "span": [s0 * 4, s1 * 4]})
    return torch.stack(feats), np.stack(crops), meta


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--eval-data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--feat-run", type=Path, help="take the letter features from this run (model_final.pt) instead; alignment and errors stay with --run")
    ap.add_argument("--dump-letters", action="store_true", help="also write letters.json: one record per evaluated letter "
                    "[id, tag, index in the cleaned label, ch, form, span x0, x1 (px at 64 px height), ok, novelty, pixel novelty] for the store (plan B7)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    model = CRNN(len(chars) + 1).cuda().eval()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cuda"))
    fm = None
    if a.feat_run:
        fchars = json.loads((a.feat_run / "charset.json").read_text(encoding="utf-8"))["chars"]
        fm = CRNN(len(fchars) + 1).cuda().eval()
        fm.load_state_dict(torch.load(a.feat_run / "model_final.pt", map_location="cuda"))
    tr = pickle.load(open(a.train_data, "rb"))["rows"]
    bank_rows = [r for r in tr if r["role"] == "train" and r["dataset"] == "muharaf_lines"]
    random.Random(0).shuffle(bank_rows)
    bank_rows = bank_rows[: a.n_train]
    ev_rows = [r for r in tr if r["role"] == "heldout" and r["dataset"] == "muharaf_lines"] + pickle.load(open(a.eval_data, "rb"))["rows"]
    del tr
    Fb, Cb, Mb = letters_of(model, bank_rows, stoi, itos, lambda r: "train", fm)
    print("bank letters", len(Mb), flush=True)
    Fe, Ce, Me = letters_of(model, ev_rows, stoi, itos, lambda r: r["dataset"], fm)
    print("eval letters", len(Me), flush=True)
    Fb = torch.nn.functional.normalize(Fb.float().cuda(), dim=1)
    Fe = torch.nn.functional.normalize(Fe.float().cuda(), dim=1)

    def pix(C):  # model-independent feature: the crop itself, contrast-normalised, blurred to 20 x 20, centred
        X = torch.from_numpy(C).float().cuda()
        lo = X.flatten(1).quantile(0.05, dim=1)[:, None, None]
        hi = X.flatten(1).quantile(0.95, dim=1)[:, None, None]
        X = ((X - lo) / (hi - lo + 1)).clamp(0, 1)[:, None]
        X = (X < 0.5).float()  # ink mask: removes paper tone, texture and contrast, which differ by scan
        X = torch.nn.functional.avg_pool2d(X, 2).flatten(1)
        return torch.nn.functional.normalize(X - X.mean(1, keepdim=True), dim=1)

    Pb = torch.cat([pix(Cb[k : k + 20000]) for k in range(0, len(Cb), 20000)])
    Pe = torch.cat([pix(Ce[k : k + 20000]) for k in range(0, len(Ce), 20000)])
    # group the bank by (character, form); fall back to the character when a form is rare
    groups = defaultdict(list)
    for i, m in enumerate(Mb):
        groups[(m["ch"], m["form"])].append(i)
        groups[(m["ch"], "*")].append(i)

    def novelty(B, E):
        scale = {}
        for g, idx in groups.items():
            if len(idx) < 20:
                continue
            X = B[idx[:8000]]
            d = 1 - X @ X.T
            d.fill_diagonal_(9)
            scale[g] = max(float(d.min(1).values.median()), 0.02)  # floor: near-duplicate crops would make the scale vanish
        out = np.full(len(Me), np.nan)
        for i, m in enumerate(Me):
            g = (m["ch"], m["form"]) if (m["ch"], m["form"]) in scale else (m["ch"], "*")
            if g in scale:
                out[i] = float((1 - B[groups[g]] @ E[i]).min()) / max(scale[g], 1e-6)
        return out, scale

    nov, scale = novelty(Fb, Fe)
    nov_pix, _ = novelty(Pb, Pe)
    if a.dump_letters:
        rec = [[m["id"], m["tag"], m["i"], m["ch"], m["form"], m["span"][0], m["span"][1], m["ok"],
                None if not np.isfinite(nov[k]) else round(float(nov[k]), 4), None if not np.isfinite(nov_pix[k]) else round(float(nov_pix[k]), 4)]
               for k, m in enumerate(Me)]
        (a.out / "letters.json").write_text(json.dumps({"fields": ["id", "tag", "i", "ch", "form", "x0", "x1", "ok", "novelty", "novelty_pix"], "rows": rec},
                                                       ensure_ascii=False), encoding="utf-8")
        print("dumped", len(rec), "letters", flush=True)
    tags = sorted({m["tag"] for m in Me})
    ok = np.array([m["ok"] for m in Me])
    tg = np.array([m["tag"] for m in Me])
    fin = np.isfinite(nov)
    edges = np.nanpercentile(nov[fin & (tg == "muharaf_lines")], [20, 40, 60, 80, 95])  # bands from the in-domain test letters
    band = np.digitize(nov, edges)
    summ = {"bank_letters": len(Mb), "bank_lines": len(bank_rows), "band_edges": edges.tolist(), "datasets": {}, "per_letter": {}}
    for t in tags:
        s = fin & (tg == t)
        summ["datasets"][t] = {
            "letters": int(s.sum()), "median_novelty": float(np.median(nov[s])), "share_above_p95_of_muharaf": float((nov[s] > edges[-1]).mean()),
            "letter_error_rate": float(1 - ok[s].mean()),
            "error_rate_by_band": [float(1 - ok[s & (band == b)].mean()) if (s & (band == b)).sum() >= 20 else None for b in range(6)],
            "share_by_band": [float((s & (band == b)).sum() / max(s.sum(), 1)) for b in range(6)]}
    finp = np.isfinite(nov_pix)
    edges_p = np.nanpercentile(nov_pix[finp & (tg == "muharaf_lines")], [20, 40, 60, 80, 95])
    band_p = np.digitize(nov_pix, edges_p)
    for t in tags:
        s = finp & (tg == t)
        summ["datasets"][t]["pixel"] = {"median_novelty": float(np.median(nov_pix[s])), "share_above_p95_of_muharaf": float((nov_pix[s] > edges_p[-1]).mean()),
                                        "error_rate_by_band": [float(1 - ok[s & (band_p == b)].mean()) if (s & (band_p == b)).sum() >= 20 else None for b in range(6)]}
    both = fin & finp
    summ["spearman_cnn_vs_pixel_novelty"] = float(np.corrcoef(np.argsort(np.argsort(nov[both])), np.argsort(np.argsort(nov_pix[both])))[0, 1])
    for (ch, f), idx in groups.items():
        if f != "*" or len(idx) < 50:
            continue
        row = {"bank": len(idx)}
        for t in tags:
            s = fin & (tg == t) & np.array([m["ch"] == ch for m in Me])
            if s.sum() >= 15:
                row[t] = {"n": int(s.sum()), "median_novelty": float(np.median(nov[s])), "error_rate": float(1 - ok[s].mean())}
        summ["per_letter"][ch] = row
    # allographs: k-means (k = 6) per common letter in the bank, medoid crops
    allo = {}
    rng = torch.Generator(device="cuda").manual_seed(0)
    for ch in ["ه", "ك", "ي", "ع", "ح", "س", "م", "ل", "ف", "ر", "ن", "ب"]:
        for form in ["iso", "ini", "med", "fin"]:
            idx = groups.get((ch, form), [])
            if len(idx) < 120:
                continue
            X = Fb[idx]
            C = X[torch.randperm(len(idx), generator=rng, device="cuda")[:6]].clone()
            for _ in range(25):
                lab = (X @ C.T).argmax(1)
                for k in range(6):
                    if (lab == k).any():
                        C[k] = torch.nn.functional.normalize(X[lab == k].mean(0), dim=0)
            lab = (X @ C.T).argmax(1)
            meds, sizes = [], []
            for k in range(6):
                mem = (lab == k).nonzero().flatten()
                if len(mem) == 0:
                    continue
                best = mem[(X[mem] @ C[k]).argmax()]
                meds.append(Cb[idx[int(best)]])
                sizes.append(int(len(mem)))
            order = np.argsort(sizes)[::-1]
            allo[f"{ch}|{form}"] = (np.stack([meds[o] for o in order]), [sizes[o] for o in order])
    np.savez_compressed(a.out / "allographs.npz", **{k: v[0] for k, v in allo.items()})
    summ["allograph_sizes"] = {k: v[1] for k, v in allo.items()}
    # examples for the gallery: per dataset the most novel letters, and random ones, with their crops
    ex_idx, ex_meta = [], []
    rs = np.random.default_rng(0)
    for t in tags:
        s = np.where(fin & (tg == t))[0]
        top = s[np.argsort(nov[s])[::-1][:24]]
        rnd = rs.choice(s, min(24, len(s)), replace=False)
        for kind, sel in (("most_novel", top), ("random", rnd)):
            for i in sel:
                ex_idx.append(int(i))
                ex_meta.append({**Me[i], "novelty": float(nov[i]), "novelty_pix": float(nov_pix[i]), "kind": kind})
    # nearest training letter for each example, to show next to it
    nn_crops = []
    for i in ex_idx:
        m = Me[i]
        g = (m["ch"], m["form"]) if (m["ch"], m["form"]) in scale else (m["ch"], "*")
        j = groups[g][int((Fb[groups[g]] @ Fe[i]).argmax())]
        nn_crops.append(Cb[j])
    np.savez_compressed(a.out / "examples.npz", crops=Ce[ex_idx], nn=np.stack(nn_crops))
    (a.out / "examples.json").write_text(json.dumps(ex_meta, ensure_ascii=False), encoding="utf-8")
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    for t, v in summ["datasets"].items():
        print(t, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k not in ("share_by_band", "pixel")}, "pixel:", v["pixel"], flush=True)
    print("spearman cnn vs pixel novelty", summ["spearman_cnn_vs_pixel_novelty"], flush=True)


if __name__ == "__main__":
    main()
