"""Writes configs/colab/pretrain_t4.json (idea 1, 2026-10-28): one T4 session that
  1. builds the RASAM-1/2 + TariMa pack exactly as for the 60-epoch baseline (repaired crops, 96 px) and adds the in-charset train lines of
     Muharaf, KHATT, Baybars and Iskandar (scripts/pretrain_external_pack.py),
  2. stage A: pretrains our CRNN (width 0.5) on all of it, model selection on RASAM-1/2 + TariMa validation,
  3. stage B: fine-tunes the stage-A best model on RASAM-1/2 + TariMa only (same recipe as the baseline: geom aug, EMA 0.999, seed 0).
Compared with reptm_h96_w05_e60_ema (60 epochs from scratch on RASAM-1/2 + TariMa) by scripts/rasam_lm_decode.py on the Mac.
"""
import json

E60 = "/Users/rabeehjouni/colab_staging/jobs/job-reptm-long-ema-t4-10261511/content/crnn_out/reptm_h96_w05_e60_ema/charset.json"
base = {"out": "/content/pack_reptm_h96.pkl", "sets": ["RASAM-1", "RASAM-2", "tarima"], "height": 96, "repair": "/content/rasam2_recut.pkl"}
T = "python -u /content/xai_train_aligner.py --data /content/pack_pre_h96.pkl --device cuda --width 0.5 --eval-every 2 --seed 0 --geom-aug"
runs = [
    {"label": "packs", "cmd": f"PACKS='{json.dumps([base])}' python /content/colab_remote_packs.py && cd /content && "
                              "python -u /content/pretrain_external_pack.py /content/pack_reptm_h96.pkl /content/pack_pre_h96.pkl "
                              "--charset /content/e60_charset.json --workers 4 | tee /content/crnn_out/pretrain_pack_stats.txt "
                              "&& rm -f /content/pack_reptm_h96.pkl"},
    {"label": "stageA", "cmd": f"{T} --out /content/crnn_out/pre_A --epochs 12 --bs 32 --train-sets r1 r2 tm mu kh bb is"},
    {"label": "stageB", "cmd": f"{T} --out /content/crnn_out/pre_B --epochs 30 --bs 16 --lr 1e-3 --train-sets r1 r2 tm --ema 0.999 "
                               "--init /content/crnn_out/pre_A/model_best.pt"},
]
spec = {"name": "pretrain", "gpu": "T4", "max_units": 3.5, "max_minutes": 210, "poll_s": 60, "sync_every_s": 600, "final_sync_grace_min": 15,
        "uploads": [[f"scripts/{n}", f"/content/{n}"] for n in ("xai_train_aligner.py", "rasam_prepare_crnn_data.py", "colab_remote_packs.py",
                                                                "pretrain_external_pack.py")]
        + [["results/rasam/repairs/rasam2_recut.pkl", "/content/rasam2_recut.pkl"], [E60, "/content/e60_charset.json"]],
        "setup": ["mkdir -p /content/crnn_out"], "runs": runs, "artifacts": ["/content/crnn_out"], "local_out": "results/rasam/colab_runs"}
json.dump(spec, open("configs/colab/pretrain_t4.json", "w"), indent=2)

# Resume (2026-10-29): the first session was lost after stage-A epoch 4 (the Mac's SSD disconnected); the trainer resumes from
# ckpt_last.pt in its output dir, so upload the partial stage A (copied from HF) and run the same steps.
PART = "/Users/rabeehjouni/colab_staging/jobs/job-pretrain-t4-10282207/content/crnn_out/pre_A"
spec["name"] = "pretrain-resume"
spec["max_units"] = 2.5
spec["uploads"] = spec["uploads"] + [[f"{PART}/{n}", f"/content/preA_{n}"] for n in ("ckpt_last.pt", "model_best.pt", "config.json", "charset.json")]
spec["setup"] = ["mkdir -p /content/crnn_out/pre_A && for n in ckpt_last.pt model_best.pt config.json charset.json; do mv /content/preA_$n /content/crnn_out/pre_A/$n; done && ls -la /content/crnn_out/pre_A"]
json.dump(spec, open("configs/colab/pretrain_resume_t4.json", "w"), indent=2)
print(json.dumps(spec, indent=1)[:1500])
