"""Gate slant TTA on measured slant (plan 2026-10-13, P1.3), offline from stored readings.

    .geomvenv/bin/python scripts/analyze_tta_gate.py [--store results/store]

For each line the gated reading is the confidence-selected shear (transform …+tta_conf) when the line's measured slant
(store_features.py) is at least T degrees from the Muharaf-training median, else the upright reading. T is chosen on a
selection half (KHATT half A from shape/halves.json, the even-indexed half of the other sets) to minimise pooled CER,
and reported on the other half. Keep the gate if the held-out NAKBA cost is ≤ +0.1 points and KHATT keeps ≥ 70 % of the
ungated TTA gain (plan's decision rule).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

W = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=W / "results/store")
    ap.add_argument("--out", type=Path, default=W / "results/analysis/V/tta_gate.json")
    ap.add_argument("--pack", default="panel/fp16-bs1", help="evaluation pack of the TTA readings (inkcrop_eval/fp16-bs1 = the old 183 / 600-line packs)")
    a = ap.parse_args()
    F = {f["sample_key"]: f for f in pq.read_table(a.store / "features.parquet").to_pylist()}
    med = json.loads((a.store / "features_reference.json").read_text())["slant_deg"]["p50"]
    halves = json.loads((W / "results/geometry/shape/halves.json").read_text())
    half_a = {k.replace("kitab_khatt/", "kitab_khatt/train/") for k in halves["half_a"]}
    R = defaultdict(dict)
    for r in pq.read_table(a.store / "lines.parquet").to_pylist():
        if r["pack"] == a.pack:
            R[(r["run_id"], r["transform"])][r["sample_key"]] = (r["e_crnn_native"], max(1, r["n_crnn_native"]))
    out = {}
    for run in ("crnn/base", "crnn/combo"):
        # cropped pair where the pack has it, else the as-stored pair (the panel TTA ran NAKBA blind as stored only)
        up = {**R[(run, "none")], **R[(run, "inkcrop")]}
        tta = {**R[(run, "tta_conf")], **R[(run, "inkcrop+tta_conf")]}
        up = {k: v for k, v in up.items() if k in tta and (k in R[(run, "inkcrop")]) == (k in R[(run, "inkcrop+tta_conf")])}
        keys = [k for k in up if k in tta and F.get(k, {}).get("ok") and F[k]["slant_deg"] is not None]
        dist = {k: abs(F[k]["slant_deg"] - med) for k in keys}
        sel = {k for k in keys if (k in half_a) if k.startswith("kitab")} | {k for k in keys if not k.startswith("kitab") and int(k[-6:]) % 2 == 0}

        def cer(ks, T):
            e = sum((tta[k] if dist[k] >= T else up[k])[0] for k in ks)
            return e / sum(up[k][1] for k in ks)

        grid = np.arange(0, 31, 1.0)
        pooled = [cer([k for k in keys if k in sel], T) for T in grid]
        T = float(grid[int(np.argmin(pooled))])
        rep = {"threshold_deg": T, "training_median_slant": med}
        for ds in ("kitab_khatt", "nakba_lines", "nakba_blind"):
            test = [k for k in keys if k.startswith(ds) and k not in sel]
            rep[ds] = {"lines": len(test), "upright": cer(test, 999), "tta_all": cer(test, 0), "gated": cer(test, T),
                       "share_gated_on": float(np.mean([dist[k] >= T for k in test]))}
        k_gain = rep["kitab_khatt"]["upright"] - rep["kitab_khatt"]["tta_all"]
        k_keep = (rep["kitab_khatt"]["upright"] - rep["kitab_khatt"]["gated"]) / k_gain if k_gain > 0 else float("nan")
        nak_cost = max(rep[d]["gated"] - rep[d]["upright"] for d in ("nakba_lines", "nakba_blind"))
        rep["keep_gate"] = bool(nak_cost <= 0.001 and k_keep >= 0.7)
        rep["khatt_gain_kept"], rep["nakba_cost"] = k_keep, nak_cost
        out[run] = rep
        print(f"{run}: threshold {T:.0f}° from the training median ({med:+.1f}°)")
        for ds in ("kitab_khatt", "nakba_lines", "nakba_blind"):
            v = rep[ds]
            print(f"   {ds:12s} held-out {v['lines']:4d}  upright {100 * v['upright']:.2f}  TTA everywhere {100 * v['tta_all']:.2f}  gated {100 * v['gated']:.2f}  (TTA on {100 * v['share_gated_on']:.0f}% of lines)")
        print(f"   KHATT gain kept {100 * k_keep:.0f}%, worst NAKBA change {100 * nak_cost:+.2f} points -> keep gate: {rep['keep_gate']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
