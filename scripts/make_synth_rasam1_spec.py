"""Pack the tadqiq synthetic RASAM-1 run for Colab CPU and write its job spec.

    python3 scripts/make_synth_rasam1_spec.py
    python3 scripts/colab_job.py configs/colab/synth_rasam1_cpu.json --dry-run

The pack (untracked, results/rasam/colab_runs/_uploads/synth_rasam1_pack.tgz) holds the tadqiq package and scripts, the
OpenITI language-model caches (so OpenITI is not downloaded), the real RASAM-1 scan (flags, summary, image measurements),
the blind re-check verdicts and the RASAM-1 model readings. The VM downloads calfa-ai/RASAM-1 (~2.9 GB) itself.
No model is trained or run on the VM: the scan is CPU-only.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
T = REPO / "tadqiq"
PACK = REPO / "results/rasam/colab_runs/_uploads/synth_rasam1_pack.tgz"
SPEC = REPO / "configs/colab/synth_rasam1_cpu.json"
READERS = ["E60_best", "CALFA_long_best", "TALL_h96_best", "SPEC_all_best", "STD_reptm_best"]


def members() -> list[tuple[Path, str]]:
    out = []
    for p in sorted((T / "src" / "tadqiq").glob("*.py")):
        out.append((p, f"tadqiq/src/tadqiq/{p.name}"))
    for n in ["synthetic_rasam1", "synthetic_inject", "synthetic_eval", "muharaf_figures"]:
        out.append((T / "scripts" / f"{n}.py", f"tadqiq/scripts/{n}.py"))
    out.append((T / "config" / "scope.json", "tadqiq/config/scope.json"))
    for p in sorted((T / "cache").glob("openiti_*.pkl")):
        out.append((p, f"tadqiq/cache/{p.name}"))
    for n in ["flags.jsonl", "summary.json", "imgstats_train.jsonl", "imgstats_validation.jsonl", "imgstats_test.jsonl"]:
        out.append((T / "results" / "RASAM-1" / n, f"tadqiq/results/RASAM-1/{n}"))
    for n in ["verdicts_user.json", "items.json"]:  # verdicts are keyed by card number; items.json maps it to the line id
        out.append((T / "results/adjudication/RASAM-1" / n, f"tadqiq/results/adjudication/RASAM-1/{n}"))
    for n in ["read_f0.json", "read_f1.json"]:
        out.append((REPO / "results/rasam/clean" / n, f"results/rasam/clean/{n}"))
    for r in READERS:
        out.append((REPO / "results/rasam/crnn" / f"{r}__r1__test.json", f"results/rasam/crnn/{r}__r1__test.json"))
    return out


def main():
    ms = members()
    miss = [str(p) for p, _ in ms if not p.exists()]
    if miss:
        raise SystemExit("missing: " + ", ".join(miss))
    PACK.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(PACK, "w:gz") as tf:
        for p, arc in ms:
            tf.add(p, arcname=arc)
    print(f"pack: {len(ms)} files, {PACK.stat().st_size / 1e6:.0f} MB -> {PACK.relative_to(REPO)}")
    dl = ("python -c \"import os; from huggingface_hub import snapshot_download; "
          "t = open('/content/.hf_token').read().strip() if os.path.exists('/content/.hf_token') else None; "
          "snapshot_download('calfa-ai/RASAM-1', repo_type='dataset', cache_dir='/content/hf/hub', token=t)\"")
    spec = {
        "name": "synth-r1",
        "gpu": "CPU",
        "max_units": 0.3,
        "max_minutes": 150,
        "poll_s": 60,
        "sync_every_s": 600,
        "final_sync_grace_min": 10,
        "uploads": [[str(PACK.relative_to(REPO)), "/content/synth_rasam1_pack.tgz"]],
        "setup": ["mkdir -p /content/work /content/synth && tar -xzf /content/synth_rasam1_pack.tgz -C /content/work "
                  "&& pip install -q rapidfuzz && " + dl + " && free -g && df -h /content | tail -1"],
        "runs": [{"label": "synth_rasam1",
                  "cmd": "cd /content/work/tadqiq && TADQIQ_HUB=/content/hf/hub TADQIQ_MAIN=/content/work "
                         "TADQIQ_SYNTH=/content/synth PYTHONPATH=src python -u scripts/synthetic_rasam1.py"}],
        "artifacts": ["/content/work/tadqiq/results/synthetic"],
        "keep": [".json", ".jsonl"],  # rasam1_recall.json + each type's flags.jsonl / summary.json (no model files here)
        "local_out": "results/rasam/colab_runs",
    }
    SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"spec -> {SPEC.relative_to(REPO)}")


if __name__ == "__main__":
    main()
