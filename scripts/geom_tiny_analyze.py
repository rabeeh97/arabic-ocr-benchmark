"""Analyse the tiny-CRNN runs: transfer test (X-T2), augmentation effect, per-line geometry against CER.

    .geomvenv/bin/python scripts/geom_tiny_analyze.py results/geometry/tiny OUT.json

Needs RUN/eval_per_line.json from `geom_tiny.py eval` for the runs present (base, geom).
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from geom_tiny import img, load  # noqa: E402

out_dir = Path(sys.argv[1])
rng = np.random.default_rng(0)
BASES = ["kitab_khatt", "nakba_lines", "muharaf_lines"]


def boot_change(a, b, n, reps=2000):
    """Paired change in pooled CER (b - a) over lines, with a 95 % bootstrap interval."""
    a, b, n = map(np.asarray, (a, b, n))
    d = (b.sum() - a.sum()) / n.sum()
    bs = []
    for _ in range(reps):
        s = rng.integers(0, len(a), len(a))
        bs.append((b[s].sum() - a[s].sum()) / n[s].sum())
    return float(d), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def paired(per, base_tag, tag):
    ids = [i for i in per[tag] if i in per[base_tag]]
    a = [per[base_tag][i][0] for i in ids]
    b = [per[tag][i][0] for i in ids]
    n = [per[base_tag][i][1] for i in ids]
    d, ci = boot_change(a, b, n)
    return {"lines": len(ids), "cer_original": sum(a) / sum(n), "cer_transformed": sum(b) / sum(n), "change": d, "ci95": ci,
            "share_lines_better": float(np.mean(np.array(b) < np.array(a))), "share_lines_worse": float(np.mean(np.array(b) > np.array(a)))}


def _feat(args):
    from geom_transfer import line_feats
    g, lab = args
    r = line_feats(g, lab)
    return None if r is None else {k: v for k, v in r.items() if not k.startswith("_")}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return float("nan"), int(ok.sum())
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def main():
    res = {"runs": {}}
    runs = [r for r in ("base", "geom") if (out_dir / r / "eval_per_line.json").exists()]
    pers = {r: json.loads((out_dir / r / "eval_per_line.json").read_text()) for r in runs}
    for r in runs:
        per = pers[r]
        R = {"cer": {t: sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for t, d in per.items()}, "paired": {}}
        for t in per:
            if "__" in t:
                R["paired"][t] = paired(per, t.split("__")[0], t)
        res["runs"][r] = R
        print(r, {t: round(100 * c, 1) for t, c in sorted(R["cer"].items())})
    if "base" in pers and "geom" in pers:  # augmentation effect on the same, untouched lines
        res["aug_effect"] = {}
        for t in BASES:
            a, b = pers["base"][t], pers["geom"][t]
            ids = [i for i in a if i in b]
            d, ci = boot_change([a[i][0] for i in ids], [b[i][0] for i in ids], [a[i][1] for i in ids])
            res["aug_effect"][t] = {"lines": len(ids), "cer_base": sum(a[i][0] for i in ids) / sum(a[i][1] for i in ids), "cer_geom": sum(b[i][0] for i in ids) / sum(a[i][1] for i in ids), "change": d, "ci95": ci}
        print("aug effect", {t: (round(100 * v["cer_base"], 1), round(100 * v["cer_geom"], 1), round(100 * v["change"], 1)) for t, v in res["aug_effect"].items()})
    # per-line geometry of the untouched evaluation lines against the base model's per-line CER
    fcache = out_dir / "eval_feats.json"
    if not fcache.exists():
        mm, meta = load(out_dir, "eval")
        ms = [m for m in meta if m["dataset"] in BASES]
        with Pool(4) as p:
            fs = p.map(_feat, [(img(mm, m), m["label"]) for m in ms], chunksize=8)
        fcache.write_text(json.dumps({f'{m["dataset"]}|{m["id"]}': f for m, f in zip(ms, fs)}))
    feats = json.loads(fcache.read_text())
    if "base" in pers:
        per = pers["base"]
        keys = ["slant_skel_deg", "sw_abs", "pen_contrast", "dot_diam_over_sw", "ink_level", "paper_level", "paper_noise", "sharpness", "turn_mean_rad", "ink_density"]
        res["line_corr"] = {}
        for t in BASES + ["pooled"]:
            tags = BASES if t == "pooled" else [t]
            cer, F = [], {k: [] for k in keys}
            for tg in tags:
                for i, (e, n) in per[tg].items():
                    f = feats.get(f"{tg}|{i}")
                    if not f:
                        continue
                    cer.append(e / n)
                    for k in keys:
                        F[k].append(f.get(k, np.nan))
            res["line_corr"][t] = {k: spearman(F[k], cer) for k in keys}
            print("corr", t, {k: round(v[0], 2) for k, v in res["line_corr"][t].items()})
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
