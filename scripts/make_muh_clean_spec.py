"""Specs for the Muharaf dirty-vs-clean training experiment (scripts/muharaf_clean_exp.py) on Colab T4.

    PYTHONPATH=src python scripts/make_muh_clean_spec.py probe            → configs/colab/muh_clean_probe_t4.json (1 epoch, arm D: timing)
    PYTHONPATH=src python scripts/make_muh_clean_spec.py seed 0 --epochs 30 [--units 8 --minutes 420]
                                                                        → configs/colab/muh_clean_s0_t4.json (all five arms, one seed)
    PYTHONPATH=src python scripts/make_muh_clean_spec.py pilot 0 --epochs 40 --units 6 --minutes 360
                                                                        → configs/colab/muh_bp_pilot_t4.json (arm D, --arch bp, CER-HV-style)
Each seed job restores its arms from the HF runs repo first, so rerunning the same spec after a cut continues the work.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTS = ROOT / "tadqiq" / "results" / "muharaf-public" / "clean_exp"
PACK = ROOT / "results" / "rasam" / "colab_runs" / "_uploads" / "muh_clean_pack.tgz"
SCRIPTS = ["muharaf_clean_exp.py", "xai_train_aligner.py", "rasam_prepare_crnn_data.py"]
# Recipes competing with CER-HV (arXiv 2601.16713) on Muharaf; all on the CER-HV ResNet CRNN front-end, 40 epochs (compute-matched).
RECIPES = {
    # their recogniser and augmentation family (affine + elastic + noise/brightness), our slant/stroke aug, EMA
    "bp": "--arch bp --bs 16 --lr 1e-3 --eval-every 2 --geom-aug --warp-aug 0.5 --ema 0.999",
    # A: modern optimisation: AdamW wd 0.05, warmup + cosine, larger batch, stochastic depth, more LSTM dropout, EMA vs last-4 averaging
    "A": "--arch bp --bs 32 --lr 2e-3 --wd 0.05 --sched cosine --warmup 2 --drop-path 0.1 --rnn-drop 0.4 --keep-last 4 "
         "--eval-every 2 --geom-aug --warp-aug 0.5 --ema 0.999",
    # B: HTR augmentation on top of bp: neighbour ink, crop jitter, small erasing, paper shading + bleed-through, feature masking
    "B": "--arch bp --bs 16 --lr 1e-3 --eval-every 2 --geom-aug --warp-aug 0.5 --neighbor-aug 0.3 --crop-aug 0.5 --erase-aug 0.3 "
         "--paper-aug 0.3 --featmask 0.5 --ema 0.999",
    # C: Conformer encoder instead of the BiLSTM (needs A's warmup/AdamW), bp augmentation
    "C": "--arch bpconf --bs 32 --lr 1e-3 --wd 0.05 --sched cosine --warmup 2 --drop-path 0.1 --keep-last 4 "
         "--eval-every 2 --geom-aug --warp-aug 0.5 --ema 0.999",
}


def make_pack():
    PACK.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(PACK, "w:gz") as t:
        for s in SCRIPTS:
            t.add(ROOT / "scripts" / s, arcname=f"scripts/{s}")
        for f in sorted(LISTS.glob("*.json")):
            t.add(f, arcname=f"lists/{f.name}")


def spec(name, run, units, minutes):
    return {
        "name": name, "gpu": "T4", "max_units": units, "max_minutes": minutes, "poll_s": 60, "sync_every_s": 600,
        "final_sync_grace_min": 15,
        "uploads": [[str(PACK.relative_to(ROOT)), "/content/muh_clean_pack.tgz"]],
        "setup": [
            "mkdir -p /content/work && tar -xzf /content/muh_clean_pack.tgz -C /content/work",
            "pip install -q rapidfuzz",
            "python -c \"from huggingface_hub import snapshot_download; snapshot_download('aamijar/muharaf-public', repo_type='dataset', "
            "cache_dir='/content/hf/hub', token=open('/content/.hf_token').read().strip())\"",
        ],
        "runs": [{"label": name, "cmd": run}],
        "artifacts": ["/content/crnn_out"],
        "keep": [".json"],
        "local_out": "results/rasam/colab_runs",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["probe", "seed", "pilot", "recipe"])
    ap.add_argument("--recipe", choices=sorted(RECIPES), help="kind recipe: which recipe")
    ap.add_argument("--arms", nargs="+", default=["D", "HVTK"])
    ap.add_argument("--parallel", type=int, default=1)  # 2 arms side by side ran out of T4 host RAM (bs 16) or GPU memory (bs 32), 2026-11-07
    ap.add_argument("seed", nargs="?", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--units", type=float, default=8.0)
    ap.add_argument("--minutes", type=int, default=420)
    a = ap.parse_args()
    make_pack()
    base = "cd /content/work/scripts && python -u muharaf_clean_exp.py --lists /content/work/lists"
    if a.kind == "probe":
        s = spec("muh-clean-probe", f"{base} --arms D --seeds 0 --epochs 1", 1.2, 75)
        out = ROOT / "configs" / "colab" / "muh_clean_probe_t4.json"
    elif a.kind == "pilot":  # CER-HV-style ResNet CRNN (--arch bp) on the raw train set: can our trainer reach their ~9.7 % full-test CER?
        name = "muh-bp-pilot"
        rec = "--arch bp --bs 16 --lr 1e-3 --eval-every 2 --geom-aug --warp-aug 0.5 --ema 0.999"
        s = spec(name, f"{base} --arms D --seeds {a.seed} --epochs {a.epochs} --restore job-{name}-t4 --recipe='{rec}'", a.units, a.minutes)
        out = ROOT / "configs" / "colab" / "muh_bp_pilot_t4.json"
    elif a.kind == "recipe":  # one recipe, the given arms and seed, two arms at a time on the T4
        arms = "".join(x.lower() for x in a.arms)
        name = f"muh-r{a.recipe.lower()}-s{a.seed}-{arms}"
        s = spec(name, f"{base} --arms {' '.join(a.arms)} --seeds {a.seed} --epochs {a.epochs} --tag _{a.recipe} --parallel {a.parallel} "
                       f"--restore job-muh-r{a.recipe.lower()}-s{a.seed} --recipe='{RECIPES[a.recipe]}'", a.units, a.minutes)
        out = ROOT / "configs" / "colab" / f"muh_r{a.recipe.lower()}_s{a.seed}_{arms}_t4.json"
    else:
        name = f"muh-clean-s{a.seed}"
        s = spec(name, f"{base} --arms D HV TK HVTK RND --seeds {a.seed} --epochs {a.epochs} --restore job-{name}-t4", a.units, a.minutes)
        out = ROOT / "configs" / "colab" / f"muh_clean_s{a.seed}_t4.json"
    out.write_text(json.dumps(s, indent=1) + "\n")
    print(out, "| pack", PACK, f"{PACK.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
