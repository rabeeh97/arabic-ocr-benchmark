"""Canonical predictions per (model, dataset) (characterisation P0, task R0.3).

The leaderboard already merges every run of a (model, dataset) pair into one
row over the union of its samples, so "one canonical run" is defined per
*sample*, the same way:

1. Only the pair's **standard prompt** counts. Prompt-study ablations
   (``line-v2-<variant>``) are excluded; if a pair was scored under more than
   one standard prompt (``line-v2`` / ``page-v2`` / ``nakba-contest``), the
   one covering the most samples wins, so a row never mixes prompts.
2. Per sample, the **latest successful** reading wins (a later re-run
   supersedes an earlier one, e.g. after the hindawi token-cap fix). A sample
   whose every reading errored keeps its latest error, so failures still count.

Run time comes from the ``<YYYYmmddTHHMMSSZ>`` directory in the file path,
else the file's mtime. Files under a ``smoke/`` directory are skipped: they are
pipeline checks, and at least one (``qwen3vl4b_visft_difflr``) wrote a
different checkpoint under an existing model name.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PRED_GLOBS = (
    "results/**/predictions*.jsonl",
    "exp3090-results/**/predictions*.jsonl",
    "suite-v1-artifacts/**/predictions*.jsonl",
)
STANDARD_PROMPTS = ("line-v2", "page-v2", "nakba-contest")
# runs that wrote their output under another experiment's model name:
# (path prefix, name in the file) -> the name the run should have had
MODEL_RENAMES = {
    # E1 condition (c) trained under the `qwen3vl4b` BASES entry (2026-09-24)
    ("suite-v1-artifacts/qwen3vl4b_visft_difflr/", "exp3090-qwen3vl4b-ft"):
        "suite-v1-qwen3vl4b-visft-difflr-ft",
}
_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")


@dataclass(frozen=True)
class Reading:
    model: str
    dataset: str
    sample_id: str
    gt: str
    pred: str
    error: str | None
    prompt_version: str
    file: str
    stamp: str


def _stamp(path: str) -> str:
    m = _STAMP_RE.findall(path)
    if m:
        return m[-1]
    from datetime import datetime, timezone

    return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iter_readings(root: str | Path = ".", globs=PRED_GLOBS):
    root = Path(root)
    files = sorted({f for g in globs for f in glob.glob(str(root / g), recursive=True)})
    for f in files:
        rel = os.path.relpath(f, root)
        if "smoke" in Path(rel).parts:
            continue
        stamp = _stamp(f)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model, sid = r.get("model") or "", r.get("sample_id") or ""
                for (prefix, old), new in MODEL_RENAMES.items():
                    if model == old and rel.startswith(prefix):
                        model = new
                ds = r.get("dataset") or sid.split("/")[0]
                if not model or not sid or model.startswith("mock") or ds == "synthetic":
                    continue
                yield Reading(
                    model=model, dataset=ds, sample_id=sid,
                    gt=r.get("ground_truth") or "",
                    pred=r.get("cleaned_text") or "",
                    error=r.get("error"),
                    prompt_version=r.get("prompt_version") or "?",
                    file=rel, stamp=stamp,
                )


def canonical(readings) -> dict[tuple[str, str], dict]:
    """{(model, dataset) -> {"prompt_version", "readings": {sample_id: Reading},
    "files": Counter(file -> samples taken), "excluded_prompts": Counter}}."""
    by_pair: dict[tuple[str, str], list[Reading]] = defaultdict(list)
    for r in readings:
        by_pair[(r.model, r.dataset)].append(r)

    out = {}
    for pair, rs in by_pair.items():
        std = [r for r in rs if r.prompt_version in STANDARD_PROMPTS]
        excluded = Counter(r.prompt_version for r in rs if r.prompt_version not in STANDARD_PROMPTS)
        if not std:
            continue
        cover = Counter()
        for pv in STANDARD_PROMPTS:
            cover[pv] = len({r.sample_id for r in std if r.prompt_version == pv})
        pv = cover.most_common(1)[0][0]
        excluded.update(r.prompt_version for r in std if r.prompt_version != pv)
        best: dict[str, Reading] = {}
        for r in sorted((r for r in std if r.prompt_version == pv), key=lambda r: (r.stamp, r.file)):
            prev = best.get(r.sample_id)
            # later wins, except an error never displaces a success
            if prev is None or not r.error or prev.error:
                best[r.sample_id] = r
        out[pair] = {
            "prompt_version": pv,
            "readings": best,
            "files": Counter(r.file for r in best.values()),
            "excluded_prompts": excluded,
        }
    return out


def manifest(canon: dict[tuple[str, str], dict]) -> list[dict]:
    """JSON-ready rows for results/characterize/canonical_runs.json."""
    rows = []
    for (model, ds), c in sorted(canon.items()):
        rd = c["readings"].values()
        rows.append({
            "model": model, "dataset": ds, "prompt_version": c["prompt_version"],
            "n": len(c["readings"]),
            "n_error": sum(1 for r in rd if r.error),
            "files": dict(c["files"].most_common()),
            "excluded_prompts": dict(c["excluded_prompts"]),
        })
    return rows


def load_canonical(root: str | Path = ".") -> dict[tuple[str, str], dict]:
    return canonical(iter_readings(root))
