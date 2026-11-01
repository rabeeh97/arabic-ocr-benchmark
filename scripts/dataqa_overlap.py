"""Cross-dataset overlap (leakage) check by ground-truth text, offline.

  PYTHONPATH=src python scripts/dataqa_overlap.py calfa-ai/RASAM-1:test calfa-ai/AMIDDA:train [...more]

The first dataset is the probe (usually a test split); every line of it whose
normalised GT (letters and digits only, >= 20 of them) also occurs in any of
the other datasets is counted. Reads only the text column of the cached
parquet shards. Writes results/dataqa/overlap_<probe>__<others>.json.

A shared GT is strong evidence the same line image is in both sets (a whole
line of manuscript text rarely recurs verbatim); short or formulaic lines
(basmala, dates) are excluded by the length floor and listed separately.
"""

import glob
import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HUB = "/Volumes/PortableSSD/huggingface/hub"
GT_KEYS = ("text", "answer", "label", "ground_truth", "transcription", "gt", "target")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    return "".join(c for c in t if unicodedata.category(c)[0] in "LN" and unicodedata.category(c) != "Mn")


def texts(spec: str) -> list[str]:
    repo, split = spec.split(":")
    snap = Path(HUB) / f"datasets--{repo.replace('/', '--')}" / "snapshots"
    files = sorted(f for f in glob.glob(str(snap / "*" / "**" / f"{split}-*.parquet"), recursive=True)
                   if not os.path.basename(f).startswith("._"))
    if not files:
        raise SystemExit(f"no cached parquet for {spec}")
    newest = max({Path(f).relative_to(snap).parts[0] for f in files}, key=lambda s: os.path.getmtime(snap / s))
    out = []
    for f in files:
        if Path(f).relative_to(snap).parts[0] != newest:
            continue
        cols = pq.read_schema(f).names
        k = next(c for c in GT_KEYS if c in cols)
        out += pq.read_table(f, columns=[k]).column(k).to_pylist()
    return out


def main():
    probe, others = sys.argv[1], sys.argv[2:]
    P = texts(probe)
    pool = Counter()
    for o in others:
        pool.update(norm(t) for t in texts(o))
    hits, short = [], 0
    for i, t in enumerate(P):
        n = norm(t)
        if n in pool:
            if len(n) >= 20:
                hits.append(i)
            else:
                short += 1
    eligible = sum(len(norm(t)) >= 20 for t in P)
    res = {"probe": probe, "against": others, "probe_lines": len(P), "eligible_ge20": eligible,
           "shared_gt": len(hits), "shared_frac": round(len(hits) / max(eligible, 1), 4),
           "short_shared_excluded": short, "indices": hits}
    name = "overlap_" + probe.replace("/", "_").replace(":", "-") + "__" + "+".join(
        o.replace("/", "_").replace(":", "-") for o in others)
    Path("results/dataqa").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(f"results/dataqa/{name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "indices"}))


if __name__ == "__main__":
    main()
