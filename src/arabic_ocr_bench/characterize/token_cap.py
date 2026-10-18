"""Which stored readings stopped at the output-token cap (NEXT_STEPS X6).

A reading that ends at the cap was cut by the pipeline, not by the model, so
its end-of-line deletions are an artefact (R2.8). Rows written after the X6
runner change carry ``hit_token_cap`` / ``finish_reason`` themselves; for older
rows the cap is inferred from ``output_tokens`` against the cap the run used
(the model's configs/models.yaml entry, or the run's ``max_tokens_override``).
Providers count a few tokens differently (gemini-flash stops at 252 of 256), so
an older row counts as at-cap within ``slack`` tokens of it.

At-cap rows are split by what the cut did:
- ``truncated``: the cleaned reading is shorter than 90 % of the GT (text lost);
- ``runaway``: it is not (a loop or a preamble ran into the cap; raising the
  cap would not help, the reading is wrong anyway).

Older ``hf_local`` rows store ``output_tokens = 0`` and are counted as unknown.
CPU only; reads predictions files, never calls a model.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml

from ..models.registry import output_token_cap
from .runs import PRED_GLOBS

SLACK = 8
DONE_SHARE = 0.05  # X6 "done when": no model has more than 5 % of lines at its cap


def _override(pred_file: str) -> int | None:
    snap = Path(pred_file).parent / "run_config.snapshot.yaml"
    if not snap.exists():
        return None
    try:
        cfg = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    v = cfg.get("max_tokens_override")
    return int(v) if v is not None else None


def row_at_cap(row: dict, cap: int | None, slack: int = SLACK) -> bool | None:
    """True / False, or None when it cannot be told (no cap, no token count)."""
    if "hit_token_cap" in row:
        return bool(row["hit_token_cap"])
    cap = row.get("token_cap") or cap
    out = int(row.get("output_tokens") or 0)
    if not cap or out <= 0:
        return None
    return out >= int(cap) - slack


def cut_kind(row: dict) -> str:
    gt = (row.get("ground_truth") or "").strip()
    hyp = (row.get("cleaned_text") or "").strip()
    return "truncated" if len(hyp) < 0.9 * len(gt) else "runaway"


def scan(root: str | Path, model_cfgs: dict[str, dict], globs=PRED_GLOBS,
         slack: int = SLACK) -> dict[tuple[str, str], dict]:
    """{(model, dataset): {n, at_cap, truncated, runaway, unknown, cap}} over
    every non-errored, non-cached reading (cached rows repeat earlier ones)."""
    root = Path(root)
    acc: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "at_cap": 0, "truncated": 0, "runaway": 0, "unknown": 0, "caps": set()})
    files = sorted({f for g in globs for f in glob.glob(str(root / g), recursive=True)})
    for f in files:
        if "smoke" in Path(os.path.relpath(f, root)).parts:
            continue
        override = _override(f)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") or r.get("cached"):
                    continue
                model = r.get("model") or ""
                cfg = model_cfgs.get(model) or {}
                cap = output_token_cap(cfg)
                if cap is not None and override is not None:
                    cap = override
                ds = r.get("dataset") or str(r.get("sample_id", "")).split("/")[0]
                a = acc[(model, ds)]
                a["n"] += 1
                hit = row_at_cap(r, cap, slack)
                if hit is None:
                    a["unknown"] += 1
                    continue
                if cap:
                    a["caps"].add(int(r.get("token_cap") or cap))
                if hit:
                    a["at_cap"] += 1
                    a[cut_kind(r)] += 1
    return {k: {**v, "caps": sorted(v["caps"])} for k, v in acc.items()}


def by_model(pairs: dict[tuple[str, str], dict]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "at_cap": 0, "truncated": 0, "runaway": 0, "unknown": 0, "caps": set()})
    for (model, _), v in pairs.items():
        o = out[model]
        for k in ("n", "at_cap", "truncated", "runaway", "unknown"):
            o[k] += v[k]
        o["caps"] |= set(v["caps"])
    for o in out.values():
        known = o["n"] - o["unknown"]
        o["share_at_cap"] = o["at_cap"] / known if known else None
        o["caps"] = sorted(o["caps"])
    return dict(out)


def flagged(models: dict[str, dict], share: float = DONE_SHARE) -> list[str]:
    """Models with more than ``share`` of their known readings at the cap."""
    return sorted(m for m, v in models.items()
                  if v["share_at_cap"] is not None and v["share_at_cap"] > share)
