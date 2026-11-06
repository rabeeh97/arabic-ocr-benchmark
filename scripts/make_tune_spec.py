"""Pack the detector tuning sweep (tadqiq/scripts/tune_detectors.py) for Colab CPU and write its job spec.

    python3 scripts/make_tune_spec.py        → configs/colab/tune_detectors_cpu.json

Input: the newest synced grid v2 run (results/rasam/colab_runs/job-synth-grid-cpu-*/.../results/synthetic/grid/<name>__{base,text,image}),
compacted locally (flag-level check names per line + injected rows; light). The VM downloads the datasets for the GTs.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synth_grid_spec import DATASETS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T = REPO / "tadqiq"


def main():
    runs = sorted(glob.glob(str(REPO / "results/rasam/colab_runs/job-synth-grid-cpu-*/content/work/tadqiq/results/synthetic/grid")),
                  key=os.path.getmtime)
    if not runs:
        raise SystemExit("no synced grid run")
    grid = Path(runs[-1])
    need = [grid / f"{n}__{c}" / f for n in DATASETS for c in ("base", "text", "image")
            for f in (["flags.jsonl"] + ([] if c == "base" else ["rows.jsonl"]))]
    miss = [str(p.relative_to(grid)) for p in need if not p.exists()]
    if miss:
        raise SystemExit(f"grid {grid} incomplete: " + ", ".join(miss[:8]))
    comp = REPO / "results/rasam/colab_runs/_uploads/tune_grid"
    subprocess.run([sys.executable, str(T / "scripts/tune_detectors.py"), "compact", str(grid), str(comp)], check=True,
                   env=dict(os.environ, PYTHONPATH=str(T / "src")))
    code = [(p, f"tadqiq/src/tadqiq/{p.name}") for p in sorted((T / "src" / "tadqiq").glob("*.py"))]
    code += [(T / "scripts/tune_detectors.py", "tadqiq/scripts/tune_detectors.py"), (T / "config/scope.json", "tadqiq/config/scope.json")]
    data = [(T / "results/known_errors/muharaf-public.jsonl", "tadqiq/results/known_errors/muharaf-public.jsonl")]
    for d in DATASETS:
        for p in [T / "results" / d / "summary.json", *sorted((T / "results" / d).glob("imgstats_*.jsonl"))]:
            data.append((p, f"tadqiq/results/{d}/{p.name}"))
    data += [(p, f"tune_grid/{p.name}") for p in sorted(comp.glob("*.json.gz"))]
    packs = []
    for part, ms in (("code", code), ("data", data)):
        pk = REPO / f"results/rasam/colab_runs/_uploads/tune_{part}.tgz"
        with tarfile.open(pk, "w:gz") as tf:
            for p, arc in ms:
                tf.add(p, arcname=arc)
        mb = pk.stat().st_size / 1e6
        print(f"{pk.name}: {mb:.0f} MB")
        if mb > 62:
            raise SystemExit(f"{pk.name} is {mb:.0f} MB: over the ~64 MB upload limit")
        packs.append(pk)
    repos = list(DATASETS.values())
    # text columns only would do, but the datasets are image parquet; snapshot_download is what the grid job used
    dl = ("python -c \"import os; from huggingface_hub import snapshot_download; "
          "t = open('/content/.hf_token').read().strip() if os.path.exists('/content/.hf_token') else None; "
          f"[snapshot_download(r, repo_type='dataset', cache_dir='/content/hf/hub', token=t) for r in {repos!r}]\"")
    spec = {
        "name": "tune-detectors",
        "gpu": "CPU",
        "max_units": 0.5,
        "max_minutes": 300,
        "poll_s": 60,
        "sync_every_s": 600,
        "final_sync_grace_min": 10,
        "uploads": [[str(pk.relative_to(REPO)), f"/content/{pk.name}"] for pk in packs],
        "setup": ["mkdir -p /content/work && " + " && ".join(f"tar -xzf /content/{pk.name} -C /content/work" for pk in packs) + " && " + dl],
        "runs": [{"label": "tune",
                  "cmd": "cd /content/work/tadqiq && TADQIQ_HUB=/content/hf/hub PYTHONPATH=src python -u scripts/tune_detectors.py run "
                         "--grid /content/work/tune_grid"}],
        "artifacts": ["/content/work/tadqiq/results/tuning"],
        "keep": [".json"],
        "local_out": "results/rasam/colab_runs",
    }
    out = REPO / "configs/colab/tune_detectors_cpu.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"grid {grid.relative_to(REPO)}\nspec -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
