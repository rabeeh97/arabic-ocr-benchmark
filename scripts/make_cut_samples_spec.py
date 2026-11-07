"""Colab CPU spec: cut the audit page's sample thumbnails for datasets that are not fully on the local disk (tadqiq/scripts/cut_samples.py).

    AUDIT_DUMP_WANT=<tmp>/want.json PYTHONPATH=src python tadqiq/scripts/audit_page.py      (in tadqiq/, local)
    python scripts/make_cut_samples_spec.py <tmp>/want.json AMIDDA:calfa-ai/AMIDDA           → configs/colab/cut_samples_cpu.json
Output comes back under results/rasam/colab_runs/job-cut-samples-cpu-*/content/crnn_out/samples/<name>/ (and via HF).
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results" / "rasam" / "colab_runs" / "_uploads" / "cut_samples.tgz"


def main():
    want, pairs = Path(sys.argv[1]), [x.split(":", 1) for x in sys.argv[2:]]
    names, repos = [n for n, _ in pairs], [r for _, r in pairs]
    with tarfile.open(PACK, "w:gz") as t:
        t.add(ROOT / "tadqiq" / "src" / "tadqiq", arcname="tadqiq/src/tadqiq", filter=lambda ti: None if "__pycache__" in ti.name or "/._" in ti.name else ti)
        t.add(ROOT / "tadqiq" / "scripts" / "cut_samples.py", arcname="tadqiq/scripts/cut_samples.py")
        t.add(want, arcname="want.json")
    dl = ("python -c \"from huggingface_hub import snapshot_download; t = open('/content/.hf_token').read().strip(); "
          f"[snapshot_download(r, repo_type='dataset', cache_dir='/content/hf/hub', token=t, max_workers=8) for r in {repos!r}]\"")
    spec = {
        "name": "cut-samples", "gpu": "CPU", "max_units": 0.3, "max_minutes": 120, "poll_s": 60, "sync_every_s": 300, "final_sync_grace_min": 10,
        "uploads": [[str(PACK.relative_to(ROOT)), "/content/cut_samples.tgz"]],
        "setup": ["mkdir -p /content/work && tar -xzf /content/cut_samples.tgz -C /content/work", dl],
        "runs": [{"label": "cut", "cmd": "cd /content/work/tadqiq && TADQIQ_HUB=/content/hf/hub PYTHONPATH=src python -u scripts/cut_samples.py "
                                         f"/content/work/want.json /content/crnn_out/samples --only {' '.join(names)}"}],
        "artifacts": ["/content/crnn_out"], "keep": [".json", ".jpg"], "local_out": "results/rasam/colab_runs",
    }
    out = ROOT / "configs" / "colab" / "cut_samples_cpu.json"
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(out, f"pack {PACK.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
