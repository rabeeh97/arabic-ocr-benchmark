"""Muharaf dirty-vs-clean training experiment. Runs INSIDE a Colab VM (configs/colab/muharaf_clean_*.json); every step is
idempotent, so a new VM continues where the last one stopped:

  1. restore: pull this experiment's earlier arm folders (ckpt_last.pt, readings) from the HF runs repo
  2. pack:    Muharaf line images at --height, published splits (train / validation→heldout / test), key '<repo>/<split>/<index>'
              (same keys as the toolkit's flags; checked against probe_gt.json, the run stops on a mismatch)
  3. train:   xai_train_aligner.py, the same recipe for every arm; arms differ only by --exclude (lists from muharaf_clean_lists.py)
              The trainer checkpoints every 200 steps and resumes from ckpt_last.pt.
  4. read:    greedy readings of every test line with the EMA weights selected on the held-out set → <arm>/test_lines.json,
              and CER on the full test set and on the test set without CER-HV's error lines → <arm>/test_summary.json

    python -u muharaf_clean_exp.py --lists /content/lists --arms D HV TK HVTK RND --seeds 0 --epochs 30 --restore job-muh-clean
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import io
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = "aamijar/muharaf-public"
NAME = "muharaf-public"
ROLES = {"train": "train", "validation": "heldout", "test": "test"}
OUT = Path("/content/crnn_out")
HF_RUNS = os.environ.get("HF_RUNS", "SubSpring/arabic-ocr-colab-runs")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def restore(prefix: str, arms: list[str]):
    """Copy each arm's newest saved state from the HF runs repo (jobs named <prefix>-*) into /content/crnn_out/<arm>."""
    from huggingface_hub import HfApi, hf_hub_download
    tok = open("/content/.hf_token").read().strip()
    files = [f for f in HfApi(token=tok).list_repo_files(HF_RUNS) if f.startswith(f"runs/{prefix}-")]
    for arm in arms:
        jobs = sorted({f.split("/")[1] for f in files if f"/crnn_out/{arm}/ckpt_last.pt" in f or f"/crnn_out/{arm}/test_lines.json" in f})
        if not jobs or (OUT / arm / "test_lines.json").exists():
            continue
        def rank(j):  # prefer a finished arm, then the newest job (names end in MMDDHHMM)
            return (any(f == f"runs/{j}/crnn_out/{arm}/test_lines.json" for f in files), j.rsplit("-", 1)[-1])
        j = max(jobs, key=rank)
        got = [f for f in files if f.startswith(f"runs/{j}/crnn_out/{arm}/") and "/model_ep" not in f]
        for f in got:
            p = hf_hub_download(HF_RUNS, f, token=tok, local_dir="/content/restore")
            dst = OUT / arm / Path(f).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, dst)
        print(f"restore {arm}: {len(got)} files from {j}", flush=True)


def build_pack(pack: Path, hub: str, height: int, probe: dict):
    P = load("rasam_prep", HERE / "rasam_prepare_crnn_data.py")
    P.H, P.MAX_W = height, 24 * height
    root = glob.glob(f"{hub}/datasets--{REPO.replace('/', '--')}/snapshots/*/data")[0]
    rows, bad, skipped = [], 0, 0
    for split, role in ROLES.items():
        k = -1
        for f in sorted(glob.glob(f"{root}/{split}-*.parquet")):
            t = pq.read_table(f)
            for i in range(0, t.num_rows, 512):
                for rec in t.slice(i, 512).to_pylist():
                    k += 1
                    key = f"{NAME}/{split}/{k:06d}"
                    if key in probe and " ".join((probe[key] or "").split()) != " ".join((rec["text"] or "").split()):
                        bad += 1
                    label = P.clean(rec["text"], "raw")
                    arr = P.prep(Image.open(io.BytesIO(rec["image"]["bytes"])))
                    if role != "test" and (not label or arr.shape[1] // 4 < len(label) + 2):
                        skipped += 1
                        continue
                    rows.append({"id": key, "key": key, "dataset": "mh", "role": role, "label": label, "img": arr,
                                 "manuscript": "", "page": "", "region": "", "orig_split": split})
        print(split, k + 1, flush=True)
    if bad > 2:
        raise SystemExit(f"row keys do not match the toolkit's: {bad} of {len(probe)} probe texts differ")
    print(f"pack: {len(rows)} rows, {skipped} train/heldout lines unusable for CTC, probe mismatches {bad}", flush=True)
    tmp = pack.with_suffix(".tmp")
    pickle.dump({"rows": rows, "H": height, "split": f"muharaf-published-raw-h{height}"}, open(tmp, "wb"), protocol=4)
    tmp.replace(pack)


def read_test(arm_dir: Path, pack_rows, test_err: set):
    import torch
    from rapidfuzz.distance import Levenshtein
    T = load("xai_train_aligner", HERE / "xai_train_aligner.py")
    cfg = json.loads((arm_dir / "config.json").read_text())
    chars = json.loads((arm_dir / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi, itos = {c: i + 1 for i, c in enumerate(chars)}, {i + 1: c for i, c in enumerate(chars)}
    T.H = int(cfg.get("height", 64))
    sel = arm_dir / "select.json"  # written by the trainer when it compared EMA-best and averaged weights on the held-out set
    ck = json.loads(sel.read_text())["ckpt"] if sel.exists() else next(
        p for p in ("model_ema_best.pt", "model_best.pt", "model_final.pt") if (arm_dir / p).exists())
    model = T.build_model(len(chars) + 1, cfg).to("cuda")
    model.load_state_dict(torch.load(arm_dir / ck, map_location="cuda"))
    model.eval()
    lines = []
    with torch.no_grad():
        for r in pack_rows:
            x, lens, _ = T.make_batch([r], stoi, False, "cuda")
            seq = T.greedy(model(x).float(), lens)[0]
            lines.append({"id": r["id"], "ref": r["label"], "hyp": "".join(itos[k] for k in seq)[::-1]})
    def cer(sel):
        e = sum(Levenshtein.distance(x["ref"], x["hyp"]) for x in sel)
        return round(e / max(1, sum(len(x["ref"]) for x in sel)), 5)
    clean = [x for x in lines if x["id"] not in test_err]
    summ = {"ckpt": ck, "n_full": len(lines), "cer_full": cer(lines), "n_clean": len(clean), "cer_clean": cer(clean),
            "n_cerhv_err": len(lines) - len(clean), "cer_on_cerhv_err": cer([x for x in lines if x["id"] in test_err])}
    (arm_dir / "test_lines.json").write_text(json.dumps(lines, ensure_ascii=False) + "\n")
    (arm_dir / "test_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(arm_dir.name, "TEST", json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["D", "HV", "TK", "HVTK", "RND"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--hub", default="/content/hf/hub")
    ap.add_argument("--pack", type=Path, default=Path("/content/pack_muharaf.pkl"))
    ap.add_argument("--restore", help="job-name prefix of earlier runs of this experiment on the HF runs repo")
    ap.add_argument("--recipe", default="--bs 16 --width 0.5 --eval-every 2 --geom-aug --ema 0.999")
    ap.add_argument("--parallel", type=int, default=1, help="train this many arms at the same time on the one GPU (each gets 1/N of its memory)")
    ap.add_argument("--tag", default="", help="suffix of the arm folders (recipe name), e.g. _bp -> D_s0_bp")
    a = ap.parse_args()
    runs = [(arm, s, f"{arm}_s{s}{a.tag}") for s in a.seeds for arm in a.arms]
    if a.restore:
        restore(a.restore, [d for _, _, d in runs])
    if all((OUT / d / "test_lines.json").exists() for _, _, d in runs):
        print("all arms done", flush=True)
        return
    if not a.pack.exists():
        build_pack(a.pack, a.hub, a.height, json.loads((a.lists / "probe_gt.json").read_text()))
    test_err = set(json.loads((a.lists / "test_cerhv_errors.json").read_text()))
    todo = [(arm, seed, d) for arm, seed, d in runs if not (OUT / d / "test_lines.json").exists()]
    env = dict(os.environ)
    if a.parallel > 1:
        env["AOCR_CUDA_MEM_FRACTION"] = f"{0.95 / a.parallel:.3f}"

    def pump(proc, d):
        for line in proc.stdout:
            print(f"[{d}] {line}", end="", flush=True)

    def start(arm, seed, d):
        arm_dir = OUT / d
        arm_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(HERE / "xai_train_aligner.py"), "--data", str(a.pack), "--out", str(arm_dir),
               "--epochs", str(a.epochs), "--device", "cuda", "--seed", str(seed), *a.recipe.split()]
        if arm != "D":
            cmd += ["--exclude", str(a.lists / f"exclude_{arm}.json")]
        print("TRAIN", d, " ".join(cmd[2:]), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        th = threading.Thread(target=pump, args=(proc, d), daemon=True)
        th.start()
        return proc, th, time.time()

    test_rows = None
    pending = [r for r in todo if not (OUT / r[2] / "model_final.pt").exists()]
    trained = [r for r in todo if (OUT / r[2] / "model_final.pt").exists()]
    live: dict = {}
    failed = []
    while pending or live:
        while pending and len(live) < max(1, a.parallel):
            r = pending.pop(0)
            live[r] = start(*r)
        time.sleep(5)
        for r, (proc, th, t0) in list(live.items()):
            if proc.poll() is None:
                continue
            th.join(timeout=10)
            del live[r]
            print(f"TIMING {r[2]} train_s={time.time() - t0:.0f} rc={proc.returncode}", flush=True)
            (failed if proc.returncode else trained).append(r)
    for arm, seed, d in sorted(trained, key=lambda r: [x[2] for x in todo].index(r[2])):
        if test_rows is None:
            test_rows = [r for r in pickle.load(open(a.pack, "rb"))["rows"] if r["role"] == "test"]
        read_test(OUT / d, test_rows, test_err)
    if failed:
        raise SystemExit(f"training failed: {[r[2] for r in failed]}")


if __name__ == "__main__":
    main()
