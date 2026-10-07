"""Where does the end-of-line error excess come from? (task R2.8)

Over every complete block (characterize/samples.py), each model x line pair is
aligned at the letters view (canonical, harakat and tatweel stripped). For
each tenth of the GT line it counts substituted, deleted and inserted
characters per GT character. Per pair it also measures:

- **tail cut**: the GT suffix after the last aligned match, when every edit
  there is a deletion: the model stopped early;
- **head cut**: the same at the start of the line;
- **overrun**: predicted text after the GT's last aligned character.

If the excess in the last tenth is deletions concentrated in tail cuts, it is
early stopping or a crop/resize artefact rather than a property of the script.
If it is substitutions spread over many lines, the model reads line ends
worse. Results are split by line length and by model.

    PYTHONPATH=src .venv/bin/python scripts/line_end_check.py [--images]

``--images`` adds image width / aspect / left-edge ink from the local HF copy
(characterize/images.py) and relates the tail-cut rate to them.
Writes results/characterize/<stamp>/line_end.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rapidfuzz.distance import Levenshtein

from arabic_ocr_bench.characterize import samples as S
from arabic_ocr_bench.characterize.runs import load_canonical

ROOT = Path(__file__).resolve().parents[1]
MIN_TAIL = 0.10   # a cut must remove at least this share of the GT line


def pair_profile(gt: str, pred: str) -> dict:
    g, p = S._letters_view(gt), S._letters_view(pred)
    n = len(g)
    sub, dele, ins = [0] * n, [0] * n, [0] * (n + 1)
    first_eq = last_eq = None
    for op in Levenshtein.opcodes(g, p):
        if op.tag == "equal":
            first_eq = op.src_start if first_eq is None else first_eq
            last_eq = op.src_end
        elif op.tag == "replace":
            k = min(op.src_end - op.src_start, op.dest_end - op.dest_start)
            for i in range(op.src_start, op.src_start + k):
                sub[i] = 1
            for i in range(op.src_start + k, op.src_end):
                dele[i] = 1
            ins[op.src_end] += max(0, (op.dest_end - op.dest_start) - k)
        elif op.tag == "delete":
            for i in range(op.src_start, op.src_end):
                dele[i] = 1
        elif op.tag == "insert":
            ins[op.src_start] += op.dest_end - op.dest_start
    tail = n - last_eq if last_eq is not None else n
    head = first_eq if first_eq is not None else n
    tail_all_del = tail > 0 and all(dele[i] for i in range(n - tail, n))
    head_all_del = head > 0 and all(dele[i] for i in range(head))
    overrun = 0
    if last_eq is not None:
        # predicted characters aligned after the GT's last match
        ops = [op for op in Levenshtein.opcodes(g, p) if op.src_start >= last_eq]
        overrun = sum(op.dest_end - op.dest_start for op in ops if op.tag in ("insert", "replace"))
    dec = {k: [0, 0, 0, 0] for k in range(10)}  # sub, del, ins, chars
    for i in range(n):
        d = min(int(10 * i / n), 9)
        dec[d][0] += sub[i]
        dec[d][1] += dele[i]
        dec[d][2] += ins[i]
        dec[d][3] += 1
    dec[9][2] += ins[n]
    return {
        "n": n, "dec": dec,
        "tail_cut": tail_all_del and tail / max(n, 1) >= MIN_TAIL,
        "tail_cut_share": tail / max(n, 1) if tail_all_del else 0.0,
        "head_cut": head_all_del and head / max(n, 1) >= MIN_TAIL,
        "overrun_share": overrun / max(n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true")
    args = ap.parse_args()

    canon = load_canonical(ROOT)
    by_ds: dict[str, dict] = defaultdict(dict)
    for (m, ds), c in canon.items():
        by_ds[ds][m] = c["readings"]
    cfgs = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text(encoding="utf-8"))
    blocks = []
    for ds in sorted(by_ds):
        models, sids = S.complete_block(by_ds[ds], 5)
        if not models:
            continue
        dec = {k: [0, 0, 0, 0] for k in range(10)}
        per_model = {}
        per_line_cut = Counter()
        length_cut = defaultdict(lambda: [0, 0])
        for m in models:
            cuts = heads = 0
            over = []
            for s in sids:
                r = by_ds[ds][m][s]
                pp = pair_profile(r.gt, r.pred)
                for k in range(10):
                    for j in range(4):
                        dec[k][j] += pp["dec"][k][j]
                cuts += pp["tail_cut"]
                heads += pp["head_cut"]
                over.append(pp["overrun_share"])
                per_line_cut[s] += pp["tail_cut"]
                band = "short(<=20)" if pp["n"] <= 20 else "mid(21-60)" if pp["n"] <= 60 else "long(>60)"
                length_cut[band][0] += pp["tail_cut"]
                length_cut[band][1] += 1
            per_model[m] = {"tail_cut_rate": cuts / len(sids), "head_cut_rate": heads / len(sids),
                            "mean_overrun": statistics.fmean(over)}
        rate = lambda k, j: dec[k][j] / dec[k][3] if dec[k][3] else 0.0  # noqa: E731
        mid = range(2, 8)
        mid_rate = {name: statistics.fmean(rate(k, j) for k in mid)
                    for j, name in enumerate(("sub", "del", "ins"))}
        last = {name: rate(9, j) for j, name in enumerate(("sub", "del", "ins"))}
        b = {
            "dataset": ds, "models": models, "n": len(sids),
            "decile_rates": [{"sub": rate(k, 0), "del": rate(k, 1), "ins": rate(k, 2)} for k in range(10)],
            "last_vs_mid": {k: last[k] - mid_rate[k] for k in last},
            "tail_cut_rate": statistics.fmean(v["tail_cut_rate"] for v in per_model.values()),
            "head_cut_rate": statistics.fmean(v["head_cut_rate"] for v in per_model.values()),
            "tail_cut_by_length": {k: v[0] / v[1] for k, v in length_cut.items() if v[1]},
            "lines_cut_by_majority": sum(1 for s in sids if per_line_cut[s] > len(models) / 2),
            "per_model": per_model,
        }
        if args.images:
            from arabic_ocr_bench.characterize.images import image_features, load_images
            try:
                ims = load_images(ds, sids, cfgs[ds])
                feats = {s: image_features(img) for s, (img, _) in ims.items()}
                cut_share = [per_line_cut[s] / len(models) for s in sids]
                b["image_rho"] = {k: S.spearman(cut_share, [feats[s][k] for s in sids])
                                  for k in ("width", "aspect", "ink_left_edge", "ink_width_frac")}
                b["image_median"] = {k: S.median([feats[s][k] for s in sids])
                                     for k in ("width", "height", "aspect", "ink_left_edge")}
            except Exception as e:  # missing local copy: text-side results still stand
                b["image_error"] = repr(e)[:200]
        blocks.append(b)
        print(f"{ds:20s} last-vs-mid sub {b['last_vs_mid']['sub']:+.3f} del {b['last_vs_mid']['del']:+.3f} "
              f"ins {b['last_vs_mid']['ins']:+.3f} | tail-cut {b['tail_cut_rate']:.3f} head-cut {b['head_cut_rate']:.3f}",
              file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results/characterize" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "line_end.json").write_text(json.dumps({"generated": stamp, "min_tail": MIN_TAIL, "blocks": blocks},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
