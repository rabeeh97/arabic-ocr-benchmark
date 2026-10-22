"""RS1: verify RASAM-1/2 split counts and write results/rasam/rasam_split_check.json."""
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = "/Volumes/PortableSSD/huggingface/hub/datasets--calfa-ai--"
out = {}
for name in ("RASAM-1", "RASAM-2"):
    splits = {}
    for f in sorted(glob.glob(f"{ROOT}{name}/snapshots/*/data/*.parquet")):
        sp = f.rsplit("/", 1)[1].split("-")[0]
        pf = pq.ParquetFile(f)
        s = splits.setdefault(sp, {"rows": 0, "files": 0, "columns": list(pf.schema_arrow.names)})
        s["rows"] += pf.metadata.num_rows
        s["files"] += 1
    out[name] = splits
    print(name, {k: (v["rows"], v["files"]) for k, v in splits.items()}, list(splits.values())[0]["columns"])
Path("results/rasam").mkdir(parents=True, exist_ok=True)
json.dump(out, open("results/rasam/rasam_split_check.json", "w"), indent=1)
