"""Leave-manuscript-out folds for RASAM-2 (15 manuscripts -> 5 folds of 3, round-robin over the sorted names) and the Colab specs that train them.

    python scripts/make_lomo_specs.py   -> results/rasam/lomo_folds.json, configs/colab/lomo_a_t4.json (folds 1-3), lomo_b_t4.json (folds 4-5)

Each fold trains our standard recipe (96 px, half width, 30 epochs, geometric augmentation, seed 0) on RASAM-1 + the other 12 RASAM-2
manuscripts + TariMa (repaired MS 1936 crops), selecting on the remaining validation lines; ALL lines of the 3 held-out manuscripts become
test lines tagged r2ho, so no page of them is seen. Scoring (Mac): CER on r2ho, and on the held-out manuscripts' official test lines only,
against the in-split model on the same lines (how much the random line split flatters RASAM-2 numbers).
"""
from __future__ import annotations

import json

counts = json.load(open("results/rasam/manuscript_counts.json"))["RASAM-2"]
names = sorted(counts)
folds = [names[i::5] for i in range(5)]
json.dump({"folds": folds, "rule": "sorted manuscript names, round-robin into 5 folds"}, open("results/rasam/lomo_folds.json", "w"), indent=1)
for i, f in enumerate(folds, 1):
    print(f"fold {i}: {f}  ({sum(sum(v for k, v in counts[m].items() if k != 'chars') for m in f)} lines)")

UPLOADS = [["scripts/xai_train_aligner.py", "/content/xai_train_aligner.py"], ["scripts/rasam_prepare_crnn_data.py", "/content/rasam_prepare_crnn_data.py"],
           ["scripts/colab_remote_packs.py", "/content/colab_remote_packs.py"], ["results/rasam/repairs/rasam2_recut.pkl", "/content/rasam2_recut.pkl"]]


def spec(name, idx):
    packs = [{"out": f"/content/pack_lomo{i}.pkl", "sets": ["RASAM-1", "RASAM-2", "tarima"], "height": 96, "repair": "/content/rasam2_recut.pkl",
              "holdout": folds[i - 1]} for i in idx]
    runs = [{"label": f"lomo{i}", "cmd": f"python -u /content/xai_train_aligner.py --data /content/pack_lomo{i}.pkl --out /content/crnn_out/lomo{i} "
             "--width 0.5 --train-sets r1 r2 tm --epochs 30 --bs 16 --device cuda --eval-every 2 --seed 0 --geom-aug --abort-stuck 6 && rm -f /content/pack_lomo" + str(i) + ".pkl"}
            for i in idx]
    return {"name": name, "gpu": "T4", "max_units": round(0.85 * len(idx) + 0.3, 2), "max_minutes": 45 * len(idx) + 15, "poll_s": 60, "sync_every_s": 300,
            "final_sync_grace_min": 15, "uploads": UPLOADS,
            "setup": ["PACKS='" + json.dumps(packs) + "' python /content/colab_remote_packs.py"], "runs": runs,
            "artifacts": ["/content/crnn_out"], "local_out": "results/rasam/colab_runs"}


json.dump(spec("lomo-a", [1, 2, 3]), open("configs/colab/lomo_a_t4.json", "w"), indent=2)
json.dump(spec("lomo-b", [4, 5]), open("configs/colab/lomo_b_t4.json", "w"), indent=2)
print("specs written: lomo_a (folds 1-3), lomo_b (folds 4-5)")
