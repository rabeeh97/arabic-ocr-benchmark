"""Pack a full rescan (tadqiq/scripts/rescan_all.py) of the nine datasets for Colab CPU and write its job spec.

    python3 scripts/make_rescan_spec.py      → configs/colab/rescan_cpu.json

Same pack as the synthetic grid (code, lexicon caches, readers, stored flags / measurements); resumable per dataset
(finished datasets are kept by the sync and skipped on a new VM only if their files are restored by hand).
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synth_grid_spec import DATASETS, members  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T = REPO / "tadqiq"


def main():
    names = list(DATASETS)
    ms = [m for m in members(names, None) if not m[1].startswith("tadqiq/scripts/")]
    ms.append((T / "scripts/rescan_all.py", "tadqiq/scripts/rescan_all.py"))
    packs = [REPO / f"results/rasam/colab_runs/_uploads/rescan_{part}.tgz" for part in ("code", "data")]
    for pk, sel in zip(packs, (lambda arc: not arc.startswith("tadqiq/results/"), lambda arc: arc.startswith("tadqiq/results/"))):
        with tarfile.open(pk, "w:gz") as tf:
            for p, arc in ms:
                if sel(arc):
                    tf.add(p, arcname=arc)
        mb = pk.stat().st_size / 1e6
        print(f"{pk.name}: {mb:.0f} MB")
        if mb > 66:
            raise SystemExit(f"{pk.name} is {mb:.0f} MB: over the ~64 MB upload limit")
    repos = [DATASETS[d] for d in names]
    dl = ("python -c \"import os; from huggingface_hub import snapshot_download; "
          "t = open('/content/.hf_token').read().strip() if os.path.exists('/content/.hf_token') else None; "
          f"[snapshot_download(r, repo_type='dataset', cache_dir='/content/hf/hub', token=t) for r in {repos!r}]\"")
    spec = {
        "name": "rescan",
        "gpu": "CPU",
        "max_units": 1.0,
        "max_minutes": 600,
        "poll_s": 60,
        "sync_every_s": 600,
        "final_sync_grace_min": 10,
        "uploads": [[str(pk.relative_to(REPO)), f"/content/{pk.name}"] for pk in packs],
        "setup": ["mkdir -p /content/work && " + " && ".join(f"tar -xzf /content/{pk.name} -C /content/work" for pk in packs)
                  + " && pip install -q rapidfuzz && " + dl],
        "runs": [{"label": "rescan",
                  "cmd": "cd /content/work/tadqiq && RESCAN_OUT=/content/crnn_out/rescan TADQIQ_HUB=/content/hf/hub TADQIQ_MAIN=/content/work PYTHONPATH=src "
                         f"python -u scripts/rescan_all.py --datasets {' '.join(names)}"}],
        "artifacts": ["/content/crnn_out/rescan"],
        "keep": [".json", ".jsonl"],
        "local_out": "results/rasam/colab_runs",
    }
    out = REPO / "configs/colab/rescan_cpu.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"spec -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
