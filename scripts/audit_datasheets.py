"""Checklist 2.11b / R1.4: one datasheet card per panel dataset, built from the audit outputs.

    .geomvenv/bin/python scripts/audit_datasheets.py [--out results/audit]

Each card: size (test lines, training pool), source, conventions (conventions_table.json), diacritic policy (diacritics.json),
leakage (leakage.json), image profile (results/store/features.parquet medians), difficulty (best and median system CER at
no_diacritics, IRT reliability, challenge-subset shares, fragment share) and three random line images.
Run after audit_leakage, audit_diacritics, audit_spelling and audit_difficulty. Outputs: OUT/datasheets.json and
OUT/datasheets.html (the HTML block is meant for the Atlas).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import HTML_HEAD, OUT, W, load_samples, load_systems, panel_images, png_b64  # noqa: E402

FEATS = ("h0", "w0", "stroke_px64", "dot_px64", "slant_deg", "ink_px_per_char", "ink_density", "pen_contrast")
FLAGS = ("low_res", "ruled", "slant_outlier", "stroke_outlier")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    J = lambda n: json.load(open(a.out / n))
    leak, dia, conv, diff, lb = J("leakage.json"), J("diacritics.json"), J("conventions_table.json"), J("difficulty.json"), J("length_bands.json")
    subsets = J("challenge_subsets.json")
    samples = load_samples()
    feats = defaultdict(list)
    for r in pq.read_table(W / "results/store/features.parquet").to_pylist():
        if r["ok"]:
            feats[r["dataset"]].append(r)
    systems = load_systems(("e_no_diacritics", "n_no_diacritics"))
    by_ds = defaultdict(list)
    for k, s in samples.items():
        by_ds[s["dataset"]].append(k)
    cards = {}
    for ds in sorted(by_ds):
        c = {"lines": len(by_ds[ds]), "source": conv["table"].get(ds, {}).get("source"), "conventions": conv["table"].get(ds),
             "diacritics": dia["R1.3_profile"].get(ds), "diacritic_rule": dia["R2.7"].get(ds, {}).get("headline_rung"),
             "leakage": {k: v for k, v in leak["table"].get(ds, {}).items() if k not in ("image_dup_examples", "muharaf_6pct_check")},
             "leak_effect": {k: v for k, v in leak["did"].get(ds, {}).items() if k not in ("in_domain_runs",)},
             "home_train_pool": leak["table"].get(ds, {}).get("home_pool"),
             "home_train_lines": leak["pools"].get(leak["table"].get(ds, {}).get("home_pool") or "", None)}
        F = feats.get(ds, [])
        if F:
            c["image_profile"] = {f: float(np.nanmedian([x[f] for x in F if x[f] is not None])) for f in FEATS}
            c["image_flags"] = {f: float(np.mean([bool(x[f]) for x in F])) for f in FLAGS}
        runs = systems.get(ds, {})
        cer = sorted((sum(x["e_no_diacritics"] for x in m.values()) / max(1, sum(max(1, x["n_no_diacritics"]) for x in m.values())), r) for r, m in runs.items() if len(m) >= 0.8 * len(by_ds[ds]))
        c["difficulty"] = {"systems_full_coverage": len(cer), "best": cer[0] if cer else None, "median_system_cer": float(np.median([x for x, _ in cer])) if cer else None,
                           "irt": {k: v for k, v in diff["R3.2"].get(ds, {}).items() if k not in ("theta", "negative_discrimination_keys")},
                           "subset_shares": {n: v["per_set"].get(ds, {}).get("share", 0.0) for n, v in subsets.items()},
                           "fragments_separate_subset": lb["decision"].get(ds, {}).get("separate_subset")}
        cards[ds] = c
    json.dump(cards, open(a.out / "datasheets.json", "w"), ensure_ascii=False, indent=1)

    rnd = random.Random(0)
    pick = {ds: rnd.sample(by_ds[ds], 3) for ds in cards}
    imgs = panel_images({k for v in pick.values() for k in v})
    pct = lambda v: "–" if v is None else f"{v:.1%}"
    html = [HTML_HEAD.format(title="Dataset datasheets"), "<h1>Datasheet cards (checklist 2.11, R1.4)</h1>",
            "<p class=mut>Built by scripts/audit_datasheets.py from results/audit/*.json and the per-line store. CER at the no_diacritics rung; "
            "systems = every run reading ≥ 80 % of the set, lines they trained on excluded.</p>"]
    for ds, c in cards.items():
        cv, d, lk, df = c["conventions"] or {}, c["diacritics"] or {}, c["leakage"], c["difficulty"]
        ip = c.get("image_profile", {})
        best = df["best"]
        html.append(f"<div class=card><h2>{ds}</h2><p>{c['source']}</p><table>"
                    f"<tr><th>size</th><td>{c['lines']} test lines; home training pool {c['home_train_pool']} ({c['home_train_lines']} lines)</td></tr>"
                    f"<tr><th>conventions</th><td>hamza on alif {pct(cv.get('hamza_on_alif'))}; tāʾ marbūṭa share {pct(cv.get('ta_marbuta_share'))}; alif maqṣūra share {pct(cv.get('alif_maqsura_share'))}; "
                    f"digits {json.dumps(cv.get('digits'))}; double spaces {pct(cv.get('double_space_lines'))}; space before punctuation {pct(cv.get('space_before_punct'))}; "
                    f"minority spellings {pct(cv.get('spelling_minority_rate'))} of tokens; documented: {cv.get('documented_conventions')}</td></tr>"
                    f"<tr><th>diacritics</th><td>{pct(d.get('share_lines_with_marks'))} of lines carry marks ({d.get('harakat_per_100_letters', 0):.2f} per 100 letters) → policy <b>{d.get('policy')}</b>; headline rung: {c['diacritic_rule']}</td></tr>"
                    f"<tr><th>leakage</th><td>exact text in train {pct(lk.get('exact_in_home_train'))}; near {pct(lk.get('near_in_home_train'))}; image {pct(lk.get('image_in_home_train'))}; "
                    f"within-set text duplicates {pct(lk.get('text_dup_within_set'))}; image duplicates in the panel {pct(lk.get('image_dup_within_panel'))}; "
                    f"memorisation effect: {c['leak_effect'].get('decision', '–')}</td></tr>"
                    f"<tr><th>image profile</th><td>{'; '.join(f'{k} {v:.2f}' for k, v in ip.items())}<br>{'; '.join(f'{k} {v:.1%}' for k, v in c.get('image_flags', {}).items())}</td></tr>"
                    f"<tr><th>difficulty</th><td>best {best[1] if best else '–'} {best[0]*100:.1f} % CER; median system {df['median_system_cer']*100:.1f} % ({df['systems_full_coverage']} systems); "
                    if best else f"<tr><th>difficulty</th><td>no system reads ≥ 80 % of the set; ")
        html.append(f"IRT split-half ρ(b) {df['irt'].get('split_half_spearman_b', ['–'])[0] if df['irt'].get('split_half_spearman_b') else '–'}; "
                    f"{'; '.join(f'{n} {v:.1%}' for n, v in df['subset_shares'].items() if v)}; fragments a separate subset: {df['fragments_separate_subset']}</td></tr></table>")
        for k in pick[ds]:
            if k in imgs:
                html.append(f"<img src='{png_b64(imgs[k], 1000)}'><div class=ar>{samples[k]['gt_raw']}</div>")
        html.append("</div>")
    html.append("</body></html>")
    (a.out / "datasheets.html").write_text("\n".join(html))
    print(json.dumps({ds: {"lines": c["lines"], "policy": (c["diacritics"] or {}).get("policy"), "leak": c["leakage"].get("leaked_any")} for ds, c in cards.items()}))


if __name__ == "__main__":
    main()
