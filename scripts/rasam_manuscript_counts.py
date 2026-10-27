"""Lines per manuscript and split for RASAM-1, RASAM-2 and TariMa (HF parquet, text columns only), to plan leave-one-manuscript-out runs.

    python scripts/rasam_manuscript_counts.py   -> results/rasam/manuscript_counts.json + a printed table
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

HUB = "/Volumes/PortableSSD/huggingface/hub"


def main() -> None:
    out = {}
    for repo, name in (("calfa-ai--RASAM-1", "RASAM-1"), ("calfa-ai--RASAM-2", "RASAM-2"), ("calfa-ai--tarima", "TariMa")):
        per = defaultdict(Counter)
        chars = defaultdict(int)
        for split in ("train", "validation", "test"):
            for f in sorted(x for x in glob.glob(f"{HUB}/datasets--{repo}/snapshots/*/data/{split}-*.parquet") if "/._" not in x):
                d = pq.read_table(f, columns=["manuscript_name", "transcription"]).to_pydict()
                for m, t in zip(d["manuscript_name"], d["transcription"]):
                    per[m][split] += 1
                    chars[m] += len(t or "")
        out[name] = {m: {**dict(c), "chars": chars[m]} for m, c in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))}
        print(f"\n{name}: {len(per)} manuscripts")
        for m, c in out[name].items():
            print(f"  {m:28s} train {c.get('train', 0):5d}  val {c.get('validation', 0):4d}  test {c.get('test', 0):4d}  chars {c['chars']}")
    json.dump(out, open("results/rasam/manuscript_counts.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
