"""Extra descriptive numbers from the geometry run: slant coverage against the training pool, rule contamination, pen contrast."""
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
rows = [r for r in json.loads((run / "per_line.json").read_text()) if r.get("ok")]
O = ["muharaf_train", "muharaf", "rasam1", "rasam2", "nakba", "nakba_blind", "khatt"]


def g(t, k):
    return np.array([r.get(k, np.nan) for r in rows if r["tag"] == t], float)


def pen_contrast(t):
    hs, vs = g(t, "sw_horizontal_stroke"), g(t, "sw_vertical_stroke")
    return vs / hs


tr = g("muharaf_train", "slant_skel_deg")
tr = tr[np.isfinite(tr)]
lo, hi = np.percentile(tr, [5, 95])
out = {"train_slant_p5_p95": [float(lo), float(hi)], "train_slant_median": float(np.median(tr)), "slant": {}, "rules": {}, "other": {}}
for t in O:
    s = g(t, "slant_skel_deg")
    s = s[np.isfinite(s)]
    out["slant"][t] = {"p10_p50_p90": [float(x) for x in np.percentile(s, [10, 50, 90])],
                       "share_outside_train_p5_p95": float(((s < lo) | (s > hi)).mean()),
                       "share_further_than_10deg_from_train_median": float((np.abs(s - np.median(tr)) > 10).mean())}
    dens = np.array([(np.abs(tr - v) < 4).mean() for v in s])  # share of training lines within ±4° of this line's slant
    out["slant"][t]["train_density_within_4deg_median"] = float(np.median(dens))
    out["slant"][t]["share_lines_where_train_density_below_5pct"] = float((dens < 0.05).mean())
    rs = g(t, "rule_share")
    out["rules"][t] = {"lines_with_rule_ink": float((rs > 0.02).mean()), "mean_rule_share_of_ink": float(np.nanmean(rs))}
    out["other"][t] = {"pen_contrast_median": float(np.nanmedian(pen_contrast(t))), "sw_cv_median": float(np.nanmedian(g(t, "sw_cv"))),
                       "ink_density_median": float(np.nanmedian(g(t, "ink_density"))),
                       "slope_abs_p90_deg": float(np.nanpercentile(np.abs(g(t, "slope_proj_deg")), 90))}
(run / "extra.json").write_text(json.dumps(out, indent=1))
for t in O:
    print(t, [round(x, 1) for x in out["slant"][t]["p10_p50_p90"]], round(out["slant"][t]["share_outside_train_p5_p95"], 2),
          round(out["slant"][t]["share_further_than_10deg_from_train_median"], 2), out["rules"][t], {a: round(b, 2) for a, b in out["other"][t].items()})
print(out["train_slant_p5_p95"], out["train_slant_median"])
