"""Re-read the 25 official test lines whose published crop is broken (re-cut from the page images by rasam_recut_broken.py --crop-check), so every
system can be scored on the clean test view (scripts/clean_bench.py). Everything that runs on the Mac CPU is here; HAFITH, AMIDDA and AraGPT2
run on a Colab CPU session (scripts/colab_recut_rerun.py, configs/colab/recut_rerun_cpu.json) on the files this script exports.

    PYTHONPATH=src:scripts python scripts/clean_rerun_local.py --stage export|ours|hatformer|kraken|apply

export     results/rasam/clean/rerun/recut_test.json: key, set, test index, reference, the re-cut image (PNG, base64, page resolution)
ours       our CRNNs on the 96 px re-cut rows of rasam_tm_h96_rep2.pkl with the settings chosen on validation for the official test
           (push_summary.json, OI_E60_*): greedy, char-LM 10-best for E60 and PT, and the hypothesis-level ensemble (ENS). Writes
           rerun/ours_greedy.json and rerun/NB_RC_<cand>_<set>.json (N-best lists for AraGPT2 scoring on Colab)
hatformer  HATFormer (card preprocessing, beam 3) -> rerun/hatformer.json
kraken     Phoenix and the TariMa manuscript model (binarised input, as in results/rasam/replication) -> rerun/phoenix.json, rerun/tarima-manuscript.json
apply      AraGPT2 rescoring of the NB_RC lists with the (w, v) chosen on validation for the official test (NB_PX_*__rescored.json)
           -> rerun/ours_lm.json {cand: {key: hyp}} with cand in E60_lm, PT_lm, ENS_lm (char LM) and E60, PT, ENS (+ AraGPT2)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
REP = Path("results/rasam/repairs/rasam_recut_v2.pkl")
PACK = "/Volumes/PortableSSD/models/rasam-replication/crnn-data/rasam_tm_h96_rep2.pkl"
OUT = Path("results/rasam/clean/rerun")
TAG = {"RASAM-1": "r1", "RASAM-2": "r2", "tarima": "tm"}
LMSCORES = OUT / "lmscore"  # preds_lmscore_NB_RC_<cand>_<set>.json, copied back from the Colab job


def recut_keys():
    reps = pickle.load(open(REP, "rb"))
    return {f"{n}/{s}/{i:06d}": v for (n, s, i), v in reps.items() if s == "test"}


def export():
    reps = recut_keys()
    rows = {r["key"]: r for r in pickle.load(open(PACK, "rb"))["rows"] if r["key"] in reps}
    out = [{"key": k, "set": TAG[k.split("/")[0]], "idx": int(k.split("/")[2]), "ref": rows[k]["label"],
            "png": base64.b64encode(v["bytes"]).decode()} for k, v in sorted(reps.items())]
    (OUT / "recut_test.json").write_text(json.dumps(out, ensure_ascii=False))
    print(len(out), "re-cut test lines ->", OUT / "recut_test.json", (OUT / "recut_test.json").stat().st_size // 1024, "KB")


def images():
    return [(d, Image.open(io.BytesIO(base64.b64decode(d["png"])))) for d in json.loads((OUT / "recut_test.json").read_text())]


def ours():
    import torch
    import push_combo as P
    import rasam_lm_decode as D
    from big_char_lm import BigCharLM, MixLM

    reps = recut_keys()
    blob = pickle.load(open(PACK, "rb"))
    rows = blob["rows"]
    T = P.load_T()
    T.H = int(blob["H"])
    chars = json.loads((P.E60 / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    summ = json.loads(Path("results/rasam/push/push_summary.json").read_text())
    test = [r for r in rows if r["key"] in reps]
    lps = {}
    for cand, (run, ck) in {"E60": P.VARIANTS[summ["stage1"]["best_single"]["E60"]], "PT": P.VARIANTS[summ["stage1"]["best_single"]["PT"]]}.items():
        cfg = json.loads((run / "config.json").read_text())
        model = T.build_model(len(chars) + 1, cfg)
        model.load_state_dict(torch.load(run / ck, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            for r in test:
                x, lens, _ = T.make_batch([r], stoi, False, "cpu")
                lps[cand, r["key"]] = model(x).float().log_softmax(-1)[:, 0, :][: max(1, int(lens[0]) // 4)].numpy()
    big = BigCharLM(P.BIG)
    small = D.CharLM([r["label"][::-1] for r in rows if r["role"] == "train" and r["dataset"] in P.SETS], 8)
    mix = MixLM(small, big, 0.5)
    ctc = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)

    def ctc_ll(lp, text):
        tgt = [stoi[c] for c in text[::-1] if c in stoi]
        return -float(ctc(torch.from_numpy(lp).unsqueeze(1), torch.tensor([tgt], dtype=torch.long).reshape(1, -1) if tgt else torch.zeros(1, 0, dtype=torch.long),
                          torch.tensor([lp.shape[0]]), torch.tensor([len(tgt)])))

    def lm_ll(text):
        v = text[::-1]
        return sum(mix.logp(c, v[:k]) for k, c in enumerate(v)) + mix.logp(D.EOS, v)

    greedy = {c: {r["key"]: P.greedy(lps[c, r["key"]], chars)[::-1] for r in test} for c in ("E60", "PT")}
    (OUT / "ours_greedy.json").write_text(json.dumps(greedy, ensure_ascii=False, indent=1))
    for t in sorted({r["dataset"] for r in test}):
        sub = [r for r in test if r["dataset"] == t]
        nb = {}
        for cand in ("E60", "PT"):
            if cand == "E60":
                j = json.loads(Path(f"results/rasam/lm/OI_E60_{t}__{t}__test.json").read_text())
                s = {k: j[k] for k in ("lambda", "alpha", "beta")}
            else:
                s = {k: summ["stage2"][f"PT_{t}"][k] for k in ("lambda", "alpha", "beta")}
            mlm = MixLM(small, big, s["lambda"]) if s["lambda"] is not None and s["lambda"] < 1 else small
            nb[cand] = {r["key"]: [[p[::-1], round(sc, 4)] for p, sc in D.prefix_beam(lps[cand, r["key"]], chars, mlm, s["alpha"], s["beta"], 12, nbest=10)]
                        for r in sub}
        p4 = summ["stage4"][t]
        w, al, be = p4["w_e60"], p4["alpha"], p4["beta"]
        nb["ENS"] = {}
        for r in sub:
            k = r["key"]
            pool = list(dict.fromkeys(h for c in ("E60", "PT") for h, _ in nb[c][k]))
            sc = sorted(((w * ctc_ll(lps["E60", k], h) + (1 - w) * ctc_ll(lps["PT", k], h) + al * lm_ll(h) + be * len(h), h) for h in pool), reverse=True)
            nb["ENS"][k] = [[h, round(s, 4)] for s, h in sc[:10]]
        for cand, d in nb.items():
            lines = [{"id": r["key"], "ref": r["label"], "hyp": d[r["key"]][0][0], "nbest": d[r["key"]]} for r in sub]
            (OUT / f"NB_RC_{cand}_{t}.json").write_text(json.dumps({"run": f"NB_RC_{cand}_{t}", "lines": lines}, ensure_ascii=False))
        print(t, len(sub), "lines: N-best lists written for E60, PT, ENS", flush=True)


def hatformer():
    import torch
    from transformers import PreTrainedTokenizerFast, VisionEncoderDecoderModel
    from hatformer_rasam_eval import pixel_values

    ck = "/Volumes/PortableSSD/models/hatformer-arams28k"
    model = VisionEncoderDecoderModel.from_pretrained(ck).eval()
    tok = PreTrainedTokenizerFast.from_pretrained(ck)
    tok.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>", "cls_token": "<s>", "bos_token": "<s>"})
    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    out = {}
    for d, im in images():
        with torch.no_grad():
            ids = model.generate(pixel_values=pixel_values(im)[None], num_beams=3, length_penalty=0, max_new_tokens=64)
        out[d["key"]] = tok.batch_decode(ids, skip_special_tokens=True)[0]
    (OUT / "hatformer.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("hatformer", len(out))


def kraken():
    import unicodedata
    from kraken.lib import models
    from rasam_kraken_score import prep, read_line

    for name, path in (("phoenix", "/Volumes/PortableSSD/models/rasam-replication/phoenix/model.mlmodel"),
                       ("tarima-manuscript", "/Volumes/PortableSSD/models/rasam-replication/tarima/model-manuscript_best.mlmodel")):
        net = models.load_any(path, device="cpu")
        out = {d["key"]: unicodedata.normalize("NFC", read_line(net, prep(im.convert("RGB"), "bin"), "horizontal-rl", d["key"]))
               for d, im in images() if d["set"] in ("r1", "r2")}
        (OUT / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(name, len(out))


def apply():
    res = {}
    for cand in ("E60", "PT", "ENS"):
        res[cand + "_lm"], res[cand] = {}, {}
        for f in sorted(OUT.glob(f"NB_RC_{cand}_*.json")):
            t = f.stem.rsplit("_", 1)[1]
            lines = json.loads(f.read_text())["lines"]
            sc = json.loads((LMSCORES / f"preds_lmscore_{f.stem}.json").read_text())["scores"]
            r = json.loads(Path(f"results/rasam/lm/NB_PX_{cand}_{t}__{t}__rescored.json").read_text())
            w, v = r["w"], r["v"]
            for l, s in zip(lines, sc):
                c = l["nbest"]
                best = max(range(len(c)), key=lambda k: c[k][1] + (w * s[k][0] + v * s[k][1] if k < len(s) else -1e9))
                res[cand + "_lm"][l["id"]] = l["hyp"]
                res[cand][l["id"]] = c[best][0]
    (OUT / "ours_lm.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: len(v) for k, v in res.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["export", "ours", "hatformer", "kraken", "apply"])
    OUT.mkdir(parents=True, exist_ok=True)
    {"export": export, "ours": ours, "hatformer": hatformer, "kraken": kraken, "apply": apply}[ap.parse_args().stage]()
