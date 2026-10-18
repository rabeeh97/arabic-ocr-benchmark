"""Fetch only the tokenizer files of the benchmark VLMs into HF_HOME (checklist 2.14, E2). No weights.

    HF_HOME=/Volumes/PortableSSD/huggingface /Volumes/PortableSSD/venvs/decoding/bin/python scripts/dec_fetch_tokenizers.py
"""
from huggingface_hub import hf_hub_download

IDS = {
    "qwen3vl": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2vl": "Qwen/Qwen2-VL-2B-Instruct",
    "ain": "MBZUAI/AIN",
    "fanar_oryx": "QCRI/Fanar-2-Oryx-IVU",
    "baseer": "Misraj/Baseer__Nakba",
    "paligemma2": "google/paligemma2-3b-pt-448",
    "paligemma": "google/paligemma-3b-pt-448",
    "gemma3": "google/gemma-3-27b-it",
    # ungated mirror of the Gemma-2 tokenizer; PaliGemma2 = Gemma-2 vocabulary + <loc>/<seg> tokens (text identical)
    "gemma2_mirror": "unsloth/gemma-2-2b-it",
}

if __name__ == "__main__":
    for k, r in IDS.items():
        got = []
        for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model"):
            try:
                hf_hub_download(r, f)
                got.append(f)
            except Exception as e:  # gated / missing file
                err = type(e).__name__
        print(k, r, got or err, flush=True)
