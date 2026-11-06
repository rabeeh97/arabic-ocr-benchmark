"""Pack the tiny orientation detector (tadqiq/scripts/orient_model.py) for Colab CPU and write its job spec.

    python3 scripts/make_orient_spec.py      → configs/colab/orient2_cpu.json

Resumable: checkpoints (every 300 steps and each epoch) go to /content/crnn_out/orient/<run>/, which the job pushes to the HF
runs repo; a new VM restores them (--restore job-orient2-cpu) and skips finished runs. The line pack is rebuilt on each VM.
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synth_grid_spec import DATASETS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T = REPO / "tadqiq"


def main():
    ms = [(p, f"tadqiq/src/tadqiq/{p.name}") for p in sorted((T / "src" / "tadqiq").glob("*.py"))]
    ms += [(T / "scripts/orient_model.py", "tadqiq/scripts/orient_model.py"),
           (T / "results/known_errors/muharaf-public.jsonl", "tadqiq/results/known_errors/muharaf-public.jsonl")]
    pk = REPO / "results/rasam/colab_runs/_uploads/orient_code.tgz"
    pk.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(pk, "w:gz") as tf:
        for p, arc in ms:
            tf.add(p, arcname=arc)
    print(f"{pk.name}: {pk.stat().st_size / 1e6:.1f} MB")
    repos = list(DATASETS.values())
    dl = ("python -c \"import os; from huggingface_hub import snapshot_download; "
          "t = open('/content/.hf_token').read().strip() if os.path.exists('/content/.hf_token') else None; "
          f"[snapshot_download(r, repo_type='dataset', cache_dir='/content/hf/hub', token=t) for r in {repos!r}]\"")
    spec = {
        "name": "orient2",
        "gpu": "CPU",
        "max_units": 1.0,
        "max_minutes": 660,
        "poll_s": 60,
        "sync_every_s": 600,
        "final_sync_grace_min": 10,
        "uploads": [[str(pk.relative_to(REPO)), f"/content/{pk.name}"]],
        "setup": [f"mkdir -p /content/work && tar -xzf /content/{pk.name} -C /content/work && {dl} && nproc && free -g"],
        "runs": [{"label": "orient",
                  "cmd": "cd /content/work/tadqiq && TADQIQ_HUB=/content/hf/hub PYTHONPATH=src python -u scripts/orient_model.py all "
                         "--restore job-orient2-cpu"}],
        "artifacts": ["/content/crnn_out/orient"],
        "keep": [".json", ".gz"],
        "local_out": "results/rasam/colab_runs",
    }
    out = REPO / "configs/colab/orient2_cpu.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"spec -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
