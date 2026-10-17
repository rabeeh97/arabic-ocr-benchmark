"""Per-sample gallery for detectors D4, D21, D23 and D24 (checklist 1.4 / 1.5): for each, the flagged lines the reference
reading gets most right, most wrong, and a random draw, with the full-width image, the ground truth and the readings.

    .geomvenv/bin/python scripts/build_detector_gallery.py [--store results/store] [--out results/analysis/D/gallery_new_detectors.html]

One self-contained HTML file (images inlined), so the exFAT SSD gets one file, not hundreds.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
from change_gallery import esc, marked, page  # noqa: E402
from detectors_extra import lam_alef, quirk_hits, ws_quirks  # noqa: E402
import seed_instability  # noqa: E402

PANEL = "panel-2026-10-13"
K = (4, 4, 4)  # best, worst, random


def b64(img: np.ndarray) -> str:  # full width (evaluation-standards item 8)
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def lc(r):
    return r["e_no_diacritics"] / max(1, r["n_no_diacritics"])


def card(k, S, img, readings, note=""):
    gt = S[k]["gt_raw"]
    rows = [f'<div class="r"><span class="k">truth</span><bdi dir="rtl" lang="ar">{esc(gt)}</bdi></div>']
    for label, r in readings:
        if r is None:
            continue
        rows.append(f'<div class="r"><span class="k">{html.escape(label)}</span><span class="e">{100 * lc(r):.0f} %</span>'
                    f'<bdi dir="rtl" lang="ar">{marked(gt, r["hyp"])}</bdi></div>')
    im = f'<img alt="line image" src="{b64(img)}">' if img is not None else ""
    return (f'<div class="gcard">{im}<div class="gtxt">{"".join(rows)}'
            f'<div class="r id"><span class="k">line</span>{html.escape(k)} · {html.escape(note)}</div></div></div>')


def section(title, note, flagged, score, render, n_total):
    flagged = [k for k in flagged if score(k) is not None]
    d = sorted(flagged, key=score)
    best, worst = d[:K[0]], d[::-1][:K[1]]
    rest = [k for k in flagged if k not in best and k not in worst]
    rnd = random.Random(0).sample(rest, min(K[2], len(rest)))
    sec = lambda name, ks: (f'<h4>{name}</h4><div class="gl">' + "".join(render(k) for k in ks) + "</div>") if ks else ""  # noqa: E731
    return (f'<section><h3>{html.escape(title)}</h3><p>{note}</p><p class="gsum">{len(flagged)} flagged of {n_total} lines</p>'
            + sec("Best read (lowest CER of the reference reading)", best) + sec("Worst read", worst) + sec("Random flagged lines", rnd) + "</section>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/D/gallery_new_detectors.html")
    a = ap.parse_args()
    S = {s["sample_key"]: s for s in pq.read_table(a.store / "samples.parquet").to_pylist()}
    IMG = {r["id"]: r["img"] for r in pickle.load(open(W / "results/panel/panel.pkl", "rb"))["rows"] if "__" not in r["dataset"]}
    want = {"crnn/base", "crnn/rep2", "crnn/base_s3", "vlm/suite-v1-qwen3vl8b-ft-bf16", "vlm/ain-7b-bf16", "vlm/qwen3-vl-8b-bf16"}
    R = defaultdict(dict)  # (run, pack) -> {key: row}, transform none
    for r in pq.read_table(a.store / "lines.parquet", filters=[("transform", "=", "none")]).to_pylist():
        if r["run_id"] in want and r.get("in_train") != "yes":
            R[(r["run_id"], r["pack"])][r["sample_key"]] = r
    base, vlm = R[("crnn/base", PANEL)], R[("vlm/suite-v1-qwen3vl8b-ft-bf16", "bench")]
    body = ['<p>Gallery for the detectors added by checklist items 1.4 and 1.5 (run_detectors.py, detectors_extra.py, seed_instability.py). '
            'Highlighted characters differ from the ground truth; ▾ marks missing ones. The percentage is the line CER without diacritics.</p>']

    # D4
    intr = {f["sample_key"]: f for f in pq.read_table(a.store / "intrusion.parquet").to_pylist()}
    f4 = [k for k, f in intr.items() if f["d4"] and k in base]
    body.append(section("D4 neighbour-line intrusion", "Ink from the line above or below at the image edge, or two baselines. Reference reading: CRNN base.",
                        f4, lambda k: lc(base[k]), lambda k: card(k, S, IMG.get(k), [("CRNN base", base.get(k)), ("Qwen3-VL-8B ft", vlm.get(k))],
                                                                     f"edge components {intr[k]['n_edge']}, cover {100 * intr[k]['edge_cov']:.0f} %, two baselines {intr[k]['multi']}"),
                        sum(k in base for k in intr)))
    # D21
    f21 = [k for k, r in vlm.items() if lc(r) <= 0.8 and lam_alef(S[k]["gt_raw"], r["hyp"])[1] >= 1]
    body.append(section("D21 lam-alef ligature misread", "A lam-alef (لا لأ لإ لآ) of the ground truth is not read as itself. Reference reading: Qwen3-VL-8B fine-tune (suite v1).",
                        f21, lambda k: lc(vlm[k]), lambda k: card(k, S, IMG.get(k), [("Qwen3-VL-8B ft", vlm.get(k)), ("CRNN base", base.get(k))],
                                                                  "ligatures misread {1} of {0}".format(*lam_alef(S[k]["gt_raw"], vlm[k]["hyp"]))), len(vlm)))
    # D23
    ain, peer = R[("vlm/ain-7b-bf16", "bench")], R[("vlm/qwen3-vl-8b-bf16", "bench")]
    f23 = [k for k, r in ain.items() if S[k]["dataset"] == "kitab_khatt" and quirk_hits(ws_quirks(S[k]["gt_raw"]), r["hyp"])[1] >= 1]

    def r23(k):
        q = ws_quirks(S[k]["gt_raw"])
        raw = esc(S[k]["gt_raw"]).replace(" ", '<span class="sp">·</span>')
        return card(k, S, IMG.get(k), [("AIN-7B", ain.get(k)), ("Qwen3-VL-8B", peer.get(k))],
                    f"quirks: {', '.join(x for x, _ in q)}; AIN reproduces {quirk_hits(q, ain[k]['hyp'])[1]}, Qwen3-VL-8B {quirk_hits(q, peer[k]['hyp'])[1] if k in peer else '—'}"
                    ).replace('<div class="r id">', f'<div class="r"><span class="k">truth, spaces</span><bdi dir="rtl" lang="ar">{raw}</bdi></div><div class="r id">', 1)
    body.append(section("D23 memorisation signature (C6 check, unverified)", "KITAB-KHATT lines where AIN-7B reproduces a whitespace quirk of the transcription "
                        "(double space, space before punctuation, detached و) that the image cannot show. Dots in the second truth row are spaces.",
                        f23, lambda k: lc(ain[k]), r23, sum(S[k]["dataset"] == "kitab_khatt" for k in ain)))
    # D24
    inst = json.loads((W / "results/analysis/D/seed_instability.json").read_text(encoding="utf-8"))["unstable_ids"]
    rep2, s3, fp16 = R[("crnn/rep2", PANEL)], R[("crnn/base_s3", PANEL)], R[("crnn/base", seed_instability.FP16)]
    f24 = [k for k in set(inst["seed"]) | set(inst["numeric"]) if k in base and k in rep2 and k in s3]
    body.append(section("D24 seed or numeric instability", "Lines whose reading changes between identical trainings (seeds of one recipe) or between fp32 and fp16. "
                        "Reference reading: CRNN base, seed 1; the other seeds and the fp16 reading follow.",
                        f24, lambda k: lc(base[k]), lambda k: card(k, S, IMG.get(k), [("base seed 1", base.get(k)), ("base seed 2", rep2.get(k)),
                                                                                       ("base seed 3", s3.get(k)), ("seed 1 fp16", fp16.get(k))],
                                                                    ("seed-unstable" if k in inst["seed"] else "") + (" numeric-unstable" if k in inst["numeric"] else "")),
                        len(base)))
    css = '<style>.gtxt .sp{color:var(--muted);font-size:11px}</style>'
    a.out.write_text(page("New detectors: D4, D21, D23, D24", css + "".join(body)), encoding="utf-8")
    print("wrote", a.out, f"{a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
