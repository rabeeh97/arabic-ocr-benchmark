"""Push experiment (2026-10-29): stack the tricks we have on the official RASAM-1 / RASAM-2 / TariMa test sets and see how far they go.

    PYTHONPATH=src python scripts/push_combo.py [--stage 1|2|3] [--val-n 300] [--tune-n 200]

Models: E60 = the 60-epoch baseline (reptm_h96_w05_e60_ema), PT = pretrained on ~40k extra lines + fine-tuned (pre_B, §22).
Stage 1  per-line log-probs of every checkpoint variant (best, ema_best, final, ema_final; E60 also its last-10 average), cached;
         greedy CER of each variant and of logit ensembles across the two models (frame-wise mean of log-probs, renormalised).
         Everything is chosen on VALIDATION (pooled over the three sets); test numbers are reported, never used to choose.
Stage 2  for the chosen ensemble, tune the in-domain + OpenITI char LM mix (lambda, alpha, beta) per set on validation.
Stage 3  10-best lists (test + validation) for E60 / PT / ENS with their own LM settings (E60 and PT read from OI_E60_* / PTOI_*),
         as results/rasam/lm/NB_PX_<cand>_<set>__<set>__{test,val_nbest}.json, for AraGPT2 rescoring (colab_remote_lmscore.py,
         rasam_nbest_rescore.py).
Summary: results/rasam/push/push_summary.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arabic_ocr_bench.metrics import aggregate, score_pair  # noqa: E402
import rasam_lm_decode as D  # noqa: E402
from big_char_lm import BigCharLM, MixLM  # noqa: E402

PACK = "/Volumes/PortableSSD/models/rasam-replication/crnn-data/rasam_tm_h96.pkl"
BIG = "/Volumes/PortableSSD/models/rasam-replication/lm/openiti_o8.npz"
E60 = Path("/Users/rabeehjouni/colab_staging/jobs/job-reptm-long-ema-t4-10261511/content/crnn_out/reptm_h96_w05_e60_ema")
PT = Path("/Users/rabeehjouni/colab_staging/jobs/job-pretrain-resume-t4-10291211/content/crnn_out/pre_B")
VARIANTS = {f"E60:{c}": (E60, f"model_{c}.pt") for c in ("best", "ema_best", "final", "ema_final", "avg10")}
VARIANTS |= {f"PT:{c}": (PT, f"model_{c}.pt") for c in ("best", "ema_best", "final", "ema_final")}
SETS = ("r1", "r2", "tm")
CACHE = Path("/Users/rabeehjouni/.claude/jobs/abb5a0e7/tmp/push_cache")
OUT = Path("results/rasam/push")


def load_T():
    spec = importlib.util.spec_from_file_location("xai_train_aligner", HERE / "xai_train_aligner.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    return T


def greedy(lp, chars):
    out, prev = [], 0
    for k in lp.argmax(-1):
        if k != prev and k != 0:
            out.append(chars[k - 1])
        prev = k
    return "".join(out)


def cer(pairs, prof="raw"):
    sc = [score_pair(ref, hyp, (prof,)) for ref, hyp in pairs]
    return round(aggregate(sc, (prof,))["micro_cer"][prof] * 100, 3)


def ens(lps):
    m = np.mean([x.astype(np.float32) for x in lps], axis=0)
    return (m - np.logaddexp.reduce(m, axis=-1, keepdims=True)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, help="0 = all stages")
    ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--tune-n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    chars = json.loads((E60 / "charset.json").read_text(encoding="utf-8"))["chars"]
    assert chars == json.loads((PT / "charset.json").read_text(encoding="utf-8"))["chars"]
    blob = pickle.load(open(PACK, "rb"))
    rows = blob["rows"]
    T = load_T()
    T.H = int(blob["H"])
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    sel = {}
    for t in SETS:
        val = [r for r in rows if r["role"] == "heldout" and r["dataset"] == t]
        sel[t, "val"] = val[:: max(1, len(val) // a.val_n)][: a.val_n]
        sel[t, "test"] = [r for r in rows if r["role"] == "test" and r["dataset"] == t]
    summ_p = OUT / "push_summary.json"
    summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}

    # ---- stage 1: cached log-probs, greedy CER of variants and ensembles ----
    lps = {}
    for v, (run, ck) in VARIANTS.items():
        f = CACHE / f"{v.replace(':', '_')}.pkl"
        if f.exists():
            lps[v] = pickle.load(open(f, "rb"))
            continue
        cfg = json.loads((run / "config.json").read_text())
        model = T.build_model(len(chars) + 1, cfg)
        model.load_state_dict(torch.load(run / ck, map_location="cpu"))
        model.eval()
        t0 = time.time()
        d = {}
        with torch.no_grad():
            for key, rs in sel.items():
                for r in rs:
                    x, lens, _ = T.make_batch([r], stoi, False, "cpu")
                    lp = model(x).float().log_softmax(-1)[:, 0, :]
                    d[r["id"]] = lp[: max(1, int(lens[0]) // 4)].numpy().astype(np.float16)
        pickle.dump(d, open(f, "wb"))
        lps[v] = d
        print(f"log-probs {v}: {len(d)} lines, {time.time() - t0:.0f}s", flush=True)

    def greedy_table(get):
        out = {}
        for split in ("val", "test"):
            allp = []
            for t in SETS:
                p = [(r["label"], greedy(get(r["id"]), chars)[::-1]) for r in sel[t, split]]
                out[f"{split}_{t}"] = cer(p)
                allp += p
            out[f"{split}_all"] = cer(allp)
        return out

    if a.stage in (0, 1) or "stage1" not in summ:
        tab = {v: greedy_table(lambda i, v=v: lps[v][i]) for v in VARIANTS}
        for v in VARIANTS:
            print(f"{v:14} greedy val {tab[v]['val_all']:.2f}  test r1 {tab[v]['test_r1']:.2f} r2 {tab[v]['test_r2']:.2f} tm {tab[v]['test_tm']:.2f}", flush=True)
        top = {m: sorted((v for v in VARIANTS if v.startswith(m)), key=lambda v: tab[v]["val_all"])[:2] for m in ("E60", "PT")}
        combos = [list(c) for c in itertools.product(top["E60"], top["PT"])] + [top["E60"] + top["PT"]]
        combos += [list(c) for c in itertools.product(top["E60"][:1], top["PT"][:1], [v for v in VARIANTS if v.startswith("E60") and v not in top["E60"]][:1])]
        for c in combos:
            name = "ENS[" + "+".join(c) + "]"
            tab[name] = greedy_table(lambda i, c=c: ens([lps[v][i] for v in c]))
            print(f"{name:40} greedy val {tab[name]['val_all']:.2f}  test r1 {tab[name]['test_r1']:.2f} r2 {tab[name]['test_r2']:.2f} tm {tab[name]['test_tm']:.2f}", flush=True)
        best_ens = min((k for k in tab if k.startswith("ENS")), key=lambda k: tab[k]["val_all"])
        summ["stage1"] = {"greedy": tab, "best_single": {m: top[m][0] for m in top}, "best_ens": best_ens}
        summ_p.write_text(json.dumps(summ, indent=1))
        print("chosen on validation:", summ["stage1"]["best_single"], best_ens, flush=True)
    s1 = summ["stage1"]
    # Frame-wise log-prob ensembles of the two models are WORSE than either model alone (stage 1: separately trained CTC models put
    # their character spikes on different frames), so the ensemble is built at the hypothesis level instead (stage 4).
    cands = {"E60": lambda i: lps["E60:best"][i].astype(np.float32),  # the baseline as published (model_best)
             "PT": lambda i, v=s1["best_single"]["PT"]: lps[v][i].astype(np.float32)}

    # ---- stage 2: LM mix for the ensemble, tuned per set on validation ----
    big = BigCharLM(BIG)
    train_txt = [r["label"][::-1] for r in rows if r["role"] == "train" and r["dataset"] in SETS]
    small_lm = D.CharLM(train_txt, 8)
    if a.stage in (0, 2) or "stage2" not in summ:
        st2 = summ.get("stage2", {})
        for cand in ("PT",):
            for t in SETS:
                if f"{cand}_{t}" in st2:
                    continue
                tune = sel[t, "val"][:: max(1, len(sel[t, "val"]) // a.tune_n)][: a.tune_n]
                res = []
                for lam in (0.3, 0.5, 0.7, 1.0):
                    mlm = MixLM(small_lm, big, lam) if lam < 1 else small_lm
                    for al, be in itertools.product((0.3, 0.4, 0.6), (1.0, 1.5)):
                        c = cer([(r["label"], D.prefix_beam(cands[cand](r["id"]), chars, mlm, al, be)[::-1]) for r in tune])
                        res.append((c, lam, al, be))
                        print(f"  {cand} {t} lambda {lam} alpha {al} beta {be}: val CER {c:.2f}", flush=True)
                c, lam, al, be = min(res)
                st2[f"{cand}_{t}"] = {"lambda": lam, "alpha": al, "beta": be, "val_cer": c}
                print(f"{cand} {t}: chosen lambda {lam} alpha {al} beta {be} (val {c:.2f})", flush=True)
                summ["stage2"] = st2
                summ_p.write_text(json.dumps(summ, indent=1))
        summ["stage2"] = st2
        summ_p.write_text(json.dumps(summ, indent=1))

    # ---- stage 3: 10-best lists for E60 / PT / ENS ----
    if a.stage in (0, 3):
        lmdir = Path("results/rasam/lm")
        res3 = summ.get("stage3", {})
        for cand, get in cands.items():
            for t in SETS:
                if cand == "E60":  # the settings chosen for the baseline in §21
                    j = json.loads((lmdir / f"OI_E60_{t}__{t}__test.json").read_text())
                    s = {"lambda": j["lambda"], "alpha": j["alpha"], "beta": j["beta"]}
                else:
                    s = {k: summ["stage2"][f"{cand}_{t}"][k] for k in ("lambda", "alpha", "beta")}
                lam = s["lambda"]
                mlm = MixLM(small_lm, big, lam) if lam is not None and lam < 1 else small_lm
                name = f"NB_PX_{cand}_{t}"
                t0 = time.time()
                lines = []
                for r in sel[t, "test"]:
                    lp = get(r["id"])
                    nb = D.prefix_beam(lp, chars, mlm, s["alpha"], s["beta"], 12, nbest=10)
                    lines.append({"id": r["id"], "manuscript": r["manuscript"], "page": r["page"], "region": r["region"], "ref": r["label"],
                                  "hyp": nb[0][0][::-1], "greedy": greedy(lp, chars)[::-1], "nbest": [[p[::-1], round(sc, 4)] for p, sc in nb]})
                # rescoring weights are tuned on the cached validation lines (for ENS they overlap the LM-tuning subset of stage 2)
                vl = [{"id": r["id"], "ref": r["label"], "nbest": [[p[::-1], round(sc, 4)] for p, sc in
                                                                  D.prefix_beam(get(r["id"]), chars, mlm, s["alpha"], s["beta"], 12, nbest=10)]}
                      for r in sel[t, "val"]]
                (lmdir / f"{name}__{t}__val_nbest.json").write_text(json.dumps({"run": name, "alpha": s["alpha"], "beta": s["beta"], "lines": vl}, ensure_ascii=False))
                g = cer([(l["ref"], l["greedy"]) for l in lines])
                h = cer([(l["ref"], l["hyp"]) for l in lines])
                hd = cer([(l["ref"], l["hyp"]) for l in lines], "diac_insensitive")
                (lmdir / f"{name}__{t}__test.json").write_text(json.dumps({"run": name, "dataset": t, "split": "test", "n": len(lines), **s,
                                                                           "micro_cer": {"raw": h / 100, "diac_insensitive": hd / 100},
                                                                           "greedy_micro_cer": {"raw": g / 100}, "lines": lines}, ensure_ascii=False, indent=1))
                res3[f"{cand}_{t}"] = {"greedy": g, "lm": h, "lm_diac_insensitive": hd, **s, "val_nbest_lines": len(vl)}
                print(f"{name}: test greedy {g:.2f}  LM {h:.2f}  ({len(vl)} val N-best lists, {time.time() - t0:.0f}s)", flush=True)
                summ["stage3"] = res3
                summ_p.write_text(json.dumps(summ, indent=1))

    # ---- stage 4: hypothesis-level ensemble (N-best union, both models' CTC likelihoods + char LM) ----
    if a.stage in (0, 3, 4):
        lmdir = Path("results/rasam/lm")
        ctc = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        mix = MixLM(small_lm, big, 0.5)

        def ctc_ll(lp: np.ndarray, text: str) -> float:  # log P(text | image) under one model, summed over all alignments
            tgt = [stoi[c] for c in text[::-1] if c in stoi]  # visual order, as in training
            x = torch.from_numpy(lp).unsqueeze(1)
            return -float(ctc(x, torch.tensor([tgt], dtype=torch.long).reshape(1, -1) if tgt else torch.zeros(1, 0, dtype=torch.long),
                              torch.tensor([lp.shape[0]]), torch.tensor([len(tgt)])))

        def lm_ll(text: str) -> float:
            v = text[::-1]
            return sum(mix.logp(c, v[:k]) for k, c in enumerate(v)) + mix.logp(D.EOS, v)

        def feats(rid: str, nb_lists: list) -> list:
            pool = list(dict.fromkeys(h for nb in nb_lists for h, _ in nb))  # union, order kept
            le, lpt = cands["E60"](rid), cands["PT"](rid)
            return [(h, ctc_ll(le, h), ctc_ll(lpt, h), lm_ll(h), len(h)) for h in pool]

        grid = [(w, al, be) for w in (0.0, 0.25, 0.5, 0.75, 1.0) for al in (0.2, 0.3, 0.4, 0.6) for be in (0.0, 0.5, 1.0, 1.5)]
        res4 = summ.get("stage4", {})
        for t in SETS:
            t0 = time.time()
            nbv = {c: {l["id"]: l["nbest"] for l in json.loads((lmdir / f"NB_PX_{c}_{t}__{t}__val_nbest.json").read_text())["lines"]} for c in ("E60", "PT")}
            nbt = {c: json.loads((lmdir / f"NB_PX_{c}_{t}__{t}__test.json").read_text())["lines"] for c in ("E60", "PT")}
            vref = {r["id"]: r["label"] for r in sel[t, "val"]}
            fv = {i: feats(i, [nbv["E60"][i], nbv["PT"][i]]) for i in nbv["E60"]}

            def pick(F, w, al, be):
                return {i: max(fs, key=lambda f: w * f[1] + (1 - w) * f[2] + al * f[3] + be * f[4])[0] for i, fs in F.items()}

            res = sorted((cer([(vref[i], h) for i, h in pick(fv, *g).items()]), *g) for g in grid)
            vc, w, al, be = res[0]
            print(f"ENS {t}: chosen on validation w(E60)={w} alpha={al} beta={be} (val {vc:.2f})", flush=True)
            tl = []
            for le, lpt in zip(nbt["E60"], nbt["PT"]):
                fs = feats(le["id"], [le["nbest"], lpt["nbest"]])
                sc = sorted(((w * f[1] + (1 - w) * f[2] + al * f[3] + be * f[4], f[0]) for f in fs), reverse=True)
                tl.append({**{k: le[k] for k in ("id", "manuscript", "page", "region", "ref")}, "hyp": sc[0][1], "greedy": le["greedy"],
                           "nbest": [[h, round(s, 4)] for s, h in sc[:10]]})
            vl = []
            for i, fs in fv.items():
                sc = sorted(((w * f[1] + (1 - w) * f[2] + al * f[3] + be * f[4], f[0]) for f in fs), reverse=True)
                vl.append({"id": i, "ref": vref[i], "nbest": [[h, round(s, 4)] for s, h in sc[:10]]})
            name = f"NB_PX_ENS_{t}"
            (lmdir / f"{name}__{t}__val_nbest.json").write_text(json.dumps({"run": name, "w": w, "alpha": al, "beta": be, "lines": vl}, ensure_ascii=False))
            h = cer([(l["ref"], l["hyp"]) for l in tl])
            hd = cer([(l["ref"], l["hyp"]) for l in tl], "diac_insensitive")
            orc = cer([(l["ref"], min((x for x, _ in l["nbest"]), key=lambda x: line_err(l["ref"], x))) for l in tl])
            (lmdir / f"{name}__{t}__test.json").write_text(json.dumps({"run": name, "dataset": t, "split": "test", "n": len(tl), "w_e60": w, "alpha": al,
                                                                       "beta": be, "lambda": 0.5, "micro_cer": {"raw": h / 100, "diac_insensitive": hd / 100},
                                                                       "lines": tl}, ensure_ascii=False, indent=1))
            res4[t] = {"w_e60": w, "alpha": al, "beta": be, "val_cer": vc, "test_cer": h, "test_cer_diac_insensitive": hd, "test_oracle_union_top10": orc}
            print(f"{name}: test {h:.2f} (diac-insensitive {hd:.2f}; oracle in the pooled top 10 {orc:.2f})  {time.time() - t0:.0f}s", flush=True)
            summ["stage4"] = res4
            summ_p.write_text(json.dumps(summ, indent=1))


def line_err(ref: str, hyp: str) -> int:
    from arabic_ocr_bench.metrics import char_edit_counts
    from arabic_ocr_bench.normalize import normalize
    c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp, "raw").strip())
    return c["sub"] + c["del"] + c["ins"]


if __name__ == "__main__":
    main()
