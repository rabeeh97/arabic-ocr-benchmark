"""Pack the tadqiq synthetic grid (tadqiq/scripts/synthetic_grid.py) for Colab CPU and write its job spec.

    python3 scripts/make_synth_grid_spec.py smoke     → configs/colab/synth_grid_smoke_cpu.json (TariMa, 5 lines per cell)
    python3 scripts/make_synth_grid_spec.py full      → configs/colab/synth_grid_cpu.json (all nine datasets, 50 per cell)

Resuming: the pack includes the newest grid.json from an earlier full run (results/rasam/colab_runs/job-synth-grid-cpu-*), so a
new VM skips the (dataset, copy) pairs already done. The VM downloads the datasets itself; nothing is trained (CPU only).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
T = REPO / "tadqiq"
DATASETS = {"muharaf-public": "aamijar/muharaf-public", "RASAM-1": "calfa-ai/RASAM-1", "RASAM-2": "calfa-ai/RASAM-2",
            "tarima": "calfa-ai/tarima", "KHATT_v1.0": "Teklia/KHATT_v1.0",
            "omar-al-saleh-manuscripts-segments": "U4RASD/omar-al-saleh-manuscripts-segments", "baybars": "calfa-ai/baybars",
            "iskandar": "calfa-ai/iskandar", "AMIDDA": "calfa-ai/AMIDDA"}
READERS = ["E60_best", "CALFA_long_best", "TALL_h96_best", "SPEC_all_best", "STD_reptm_best"]


def members(names, resume):
    out = [(p, f"tadqiq/src/tadqiq/{p.name}") for p in sorted((T / "src" / "tadqiq").glob("*.py"))]
    for n in ["synthetic_grid", "synthetic_inject", "synthetic_eval", "muharaf_figures"]:
        out.append((T / "scripts" / f"{n}.py", f"tadqiq/scripts/{n}.py"))
    out.append((T / "config" / "scope.json", "tadqiq/config/scope.json"))
    out += [(p, f"tadqiq/cache/{p.name}") for p in sorted((T / "cache").glob("openiti_*.pkl"))]
    for d in names:
        for p in [T / "results" / d / "flags.jsonl", T / "results" / d / "summary.json", *sorted((T / "results" / d).glob("imgstats_*.jsonl"))]:
            out.append((p, f"tadqiq/results/{d}/{p.name}"))
    for n in ["read_f0.json", "read_f1.json"]:
        out.append((REPO / "results/rasam/clean" / n, f"results/rasam/clean/{n}"))
    for r in READERS:
        for s in ("r1", "r2", "tm"):
            p = REPO / "results/rasam/crnn" / f"{r}__{s}__test.json"
            if p.exists():
                out.append((p, f"results/rasam/crnn/{p.name}"))
    if resume:
        out.append((resume, "tadqiq/results/synthetic/grid_v2.json"))
    return out


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "full"
    names = ["tarima"] if kind == "smoke" else list(DATASETS)
    n = 5 if kind == "smoke" else 50
    prev = sorted(glob.glob(str(REPO / "results/rasam/colab_runs/job-synth-grid-cpu-*/content/work/tadqiq/results/synthetic/grid_v2.json")),
                  key=os.path.getmtime)
    resume = Path(prev[-1]) if prev and kind == "full" else None
    ms = members(names, resume)
    miss = [str(p) for p, _ in ms if not p.exists()]
    if miss:
        raise SystemExit("missing: " + ", ".join(miss))
    # two tarballs: uploads above ~64 MB fail with the colab CLI (85 MB failed twice, 64 MB worked)
    packs = [REPO / f"results/rasam/colab_runs/_uploads/synth_grid_{kind}_{part}.tgz" for part in ("code", "data")]
    packs[0].parent.mkdir(parents=True, exist_ok=True)
    for pk, sel in zip(packs, (lambda arc: not arc.startswith("tadqiq/results/"), lambda arc: arc.startswith("tadqiq/results/"))):
        with tarfile.open(pk, "w:gz") as tf:
            for p, arc in ms:
                if sel(arc):
                    tf.add(p, arcname=arc)
        mb = pk.stat().st_size / 1e6
        print(f"{pk.name}: {mb:.0f} MB")
        if mb > 66:
            raise SystemExit(f"{pk.name} is {mb:.0f} MB: over the ~64 MB upload limit, split further")
    print(f"{len(ms)} files" + (f", resuming from {resume.relative_to(REPO)}" if resume else ""))
    repos = [DATASETS[d] for d in names]
    dl = ("python -c \"import os; from huggingface_hub import snapshot_download; "
          "t = open('/content/.hf_token').read().strip() if os.path.exists('/content/.hf_token') else None; "
          f"[snapshot_download(r, repo_type='dataset', cache_dir='/content/hf/hub', token=t) for r in {repos!r}]\"")
    spec = {
        "name": "synth-grid-smoke" if kind == "smoke" else "synth-grid",
        "gpu": "CPU",
        "max_units": 0.2 if kind == "smoke" else 1.0,
        "max_minutes": 60 if kind == "smoke" else 660,
        "poll_s": 60,
        "sync_every_s": 600,
        "final_sync_grace_min": 10,
        "uploads": [[str(pk.relative_to(REPO)), f"/content/{pk.name}"] for pk in packs],
        "setup": ["mkdir -p /content/work /content/synth && " + " && ".join(f"tar -xzf /content/{pk.name} -C /content/work" for pk in packs) + " "
                  "&& pip install -q rapidfuzz && " + dl + " && free -g && df -h /content | tail -1"],
        "runs": [{"label": "synth_grid",
                  "cmd": "cd /content/work/tadqiq && TADQIQ_HUB=/content/hf/hub TADQIQ_MAIN=/content/work TADQIQ_SYNTH=/content/synth "
                         f"PYTHONPATH=src python -u scripts/synthetic_grid.py --datasets {' '.join(names)} --n {n}"}],
        "artifacts": ["/content/work/tadqiq/results/synthetic"],
        "keep": [".json"],
        "local_out": "results/rasam/colab_runs",
    }
    out = REPO / f"configs/colab/synth_grid{'_smoke' if kind == 'smoke' else ''}_cpu.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"spec -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
