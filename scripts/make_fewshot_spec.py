import json
folds = json.load(open("results/rasam/lomo_folds.json"))["folds"]
jobs = {1: "job-lomo-a-t4-10271738", 2: "job-lomo-a-t4-10271738", 3: "job-lomo-a-t4-10271738", 4: "job-lomo-b-t4-10271738", 5: "job-lomo-b-t4-10271738"}
dl = ("python -c \"from huggingface_hub import hf_hub_download as d; import shutil, os; t=open('/content/.hf_token').read().strip()\n"
      "for f, j in {F}.items():\n"
      "    o=f'/content/folds/lomo{f}'; os.makedirs(o, exist_ok=True)\n"
      "    for n in ('model_best.pt', 'config.json', 'charset.json'): shutil.copy(d('SubSpring/arabic-ocr-colab-runs', f'runs/{j}/crnn_out/lomo{f}/'+n, token=t), o+'/'+n)\n"
      "print('fold models downloaded')\"").replace("{F}", repr(jobs))
runs = []
for i, ms in enumerate(folds, 1):
    pack = {"out": f"/content/pack_lomo{i}.pkl", "sets": ["RASAM-1", "RASAM-2", "tarima"], "height": 96, "repair": "/content/rasam2_recut.pkl", "holdout": ms}
    runs.append({"label": f"fewshot{i}",
                 "cmd": f"PACKS='{json.dumps([pack])}' python /content/colab_remote_packs.py && python -u /content/rasam_fewshot_adapt.py "
                        f"--pack /content/pack_lomo{i}.pkl --run /content/folds/lomo{i} --out /content/crnn_out/fewshot/fewshot_lomo{i}.json "
                        f"--ks 0 10 25 50 100 -1 --seeds 0 1 --epochs 15 --device cuda && rm -f /content/pack_lomo{i}.pkl"})
spec = {"name": "fewshot", "gpu": "T4", "max_units": 3.0, "max_minutes": 170, "poll_s": 60, "sync_every_s": 300, "final_sync_grace_min": 15,
        "uploads": [[f"scripts/{n}", f"/content/{n}"] for n in ("xai_train_aligner.py", "rasam_lm_decode.py", "rasam_fewshot_adapt.py",
                                                                "rasam_prepare_crnn_data.py", "colab_remote_packs.py")]
        + [["results/rasam/repairs/rasam2_recut.pkl", "/content/rasam2_recut.pkl"]],
        "setup": [dl], "runs": runs, "artifacts": ["/content/crnn_out"], "local_out": "results/rasam/colab_runs"}
json.dump(spec, open("configs/colab/fewshot_t4.json", "w"), indent=2)
print(spec["runs"][0]["cmd"][:300])
print(dl)
