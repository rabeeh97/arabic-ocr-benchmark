"""Generate the Colab notebooks under notebooks/ from plain cell lists.

The .ipynb files are committed (Colab needs them), but this script is the
source of truth — edit here and re-run:  uv run python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(lines),
    }


def _lines(lines):
    text = "\n".join(lines)
    out = text.splitlines(keepends=True)
    return out or [""]


def notebook(cells: list[dict], *, gpu: bool) -> dict:
    meta = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}


def write(name: str, nb: dict) -> None:
    NB_DIR.mkdir(exist_ok=True)
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(NB_DIR.parent)}  ({len(nb['cells'])} cells)")


# --------------------------------------------------------------------------- #
# shared setup cells
# --------------------------------------------------------------------------- #
CLONE_CELL = code(
    "# Clone the (private) benchmark repo using a fine-grained PAT stored as a",
    "# Colab secret named GH_TOKEN.  See notebooks/README.md to create one.",
    "import os, subprocess",
    "GH_USER, REPO = 'rabeeh97', 'arabic-ocr-benchmark'",
    "token = None",
    "try:",
    "    from google.colab import userdata",
    "    token = userdata.get('GH_TOKEN')",
    "except Exception as e:",
    "    print('no GH_TOKEN secret:', e)",
    "if not os.path.isdir(REPO):",
    "    auth = f'{token}@' if token else ''",
    "    url = f'https://{auth}github.com/{GH_USER}/{REPO}.git'",
    "    subprocess.run(['git', 'clone', '--depth', '1', url], check=True)",
    "os.chdir(REPO)",
    "print(subprocess.run(['git','log','--oneline','-1'], capture_output=True, text=True).stdout)",
)

HF_CACHE_CELL = code(
    "# Keep all downloads on Colab's own scratch disk (ephemeral, ~78 GB) --",
    "# never the local machine. Nothing here touches your computer.",
    "import os",
    "os.environ['HF_HOME'] = '/content/hf_cache'",
    "os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'",
    "os.environ['TOKENIZERS_PARALLELISM'] = 'false'",
)

DRIVE_SAVE_CELL = code(
    "# Persist this run's results to Google Drive (they vanish when Colab recycles).",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "import shutil, pathlib",
    "dst = pathlib.Path('/content/drive/MyDrive/arabic-ocr-bench/results') / run_dir.name",
    "dst.parent.mkdir(parents=True, exist_ok=True)",
    "shutil.copytree(run_dir, dst, dirs_exist_ok=True)",
    "print('saved ->', dst)",
)

PEEK_CELL = code(
    "import json, pandas as pd, pathlib",
    "rows = [json.loads(l) for l in (run_dir / 'predictions.jsonl').read_text().splitlines()]",
    "df = pd.DataFrame(rows)[['model','sample_id','ground_truth','cleaned_text','latency_ms','error']]",
    "pd.set_option('display.max_colwidth', 60)",
    "df.head(20)",
)


# --------------------------------------------------------------------------- #
# 01 - local models
# --------------------------------------------------------------------------- #
def build_local():
    cells = [
        md(
            "# Arabic Cursive OCR Benchmark — Local Models on Colab",
            "",
            "Runs open-weight / Arabic-specific VLMs (Qari-OCR, Qwen2.5-VL, PaliGemma-Arabic)",
            "against the **Muharaf** handwritten-Arabic test set and reports CER / WER /",
            "diacritic-F1 / latency.",
            "",
            "**Before running:** `Runtime -> Change runtime type -> T4 GPU`, and add a",
            "`GH_TOKEN` secret (see `notebooks/README.md`). ~10-20 min end to end.",
        ),
        md("## 1. Check the GPU"),
        code(
            "import torch",
            "assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> T4 GPU'",
            "print(torch.cuda.get_device_name(0))",
            "!nvidia-smi --query-gpu=memory.total,memory.used --format=csv",
        ),
        md("## 2. Clone the benchmark repo"),
        CLONE_CELL,
        md("## 3. Install the harness (with local-model deps)"),
        HF_CACHE_CELL,
        code(
            "!pip install -q -e '.[local]'",
            "# Colab ships a recent torch already; this pulls transformers>=4.49, peft, qwen-vl-utils, ...",
        ),
        md(
            "## 4. Pick a model and run",
            "",
            "Model names come from `configs/models.yaml`. Good first picks on a T4:",
            "`qari-ocr-0.2` (2B, Arabic-specific), `qwen2.5-vl-3b`, `qwen2.5-vl-7b`.",
            "`paligemma-arabic-hw` needs HF access to the gated base `google/paligemma-3b-pt-448`",
            "(accept the license on HF and add an `HF_TOKEN` secret).",
        ),
        code(
            "MODEL = 'qari-ocr-0.2'   # <- change me",
            "LIMIT = 50               # lines from the Muharaf test split; set None for all 1,330",
            "",
            "run_cfg = {",
            "    'models': [MODEL],",
            "    'datasets': ['muharaf_lines'],",
            "    'limit': LIMIT,",
            "    'prompt_version': 'line-v1',",
            "    'normalization_profiles': ['raw', 'light', 'diac_insensitive'],",
            "    'concurrency': 1,",
            "    'results_dir': 'results',",
            "}",
        ),
        code(
            "from arabic_ocr_bench.runner import run",
            "from arabic_ocr_bench.report import score_run",
            "",
            "run_dir = run(run_cfg)",
            "summary_md = score_run(run_dir, profiles=tuple(run_cfg['normalization_profiles']))",
            "print(open(summary_md).read())",
        ),
        md("## 5. Inspect predictions"),
        PEEK_CELL,
        md("## 6. Save results to Google Drive"),
        DRIVE_SAVE_CELL,
        md(
            "## Next",
            "",
            "- Re-run with a different `MODEL`; results accumulate under `results/` and on Drive.",
            "- Set `LIMIT = None` for the full test split once a small run looks right.",
            "- Open `03_analysis.ipynb` to compare models and browse the worst errors.",
        ),
    ]
    write("01_colab_local_models.ipynb", notebook(cells, gpu=True))


# --------------------------------------------------------------------------- #
# 02 - hosted APIs
# --------------------------------------------------------------------------- #
def build_hosted():
    cells = [
        md(
            "# Arabic Cursive OCR Benchmark — Hosted APIs on Colab",
            "",
            "Runs Gemini / OpenAI / Claude against the **Muharaf** test set via their APIs.",
            "No GPU needed. Add whichever of these Colab secrets you have keys for:",
            "`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`",
            "(a free Gemini key: https://aistudio.google.com/apikey).",
        ),
        md("## 1. Clone + install"),
        CLONE_CELL,
        code("!pip install -q -e '.'  # core only — no torch/transformers"),
        md("## 2. Load API keys from Colab secrets"),
        code(
            "import os",
            "from google.colab import userdata",
            "KEYMAP = {",
            "    'GEMINI_API_KEY': 'gemini-2.5-flash',",
            "    'OPENAI_API_KEY': 'gpt-4o',",
            "    'ANTHROPIC_API_KEY': 'claude-sonnet',",
            "}",
            "MODELS = []",
            "for env_key, model_name in KEYMAP.items():",
            "    try:",
            "        val = userdata.get(env_key)",
            "    except Exception:",
            "        val = None",
            "    if val:",
            "        os.environ[env_key] = val",
            "        MODELS.append(model_name)",
            "print('will run:', MODELS or 'NOTHING — add at least one API key secret')",
        ),
        md("## 3. Run + report"),
        code(
            "run_cfg = {",
            "    'models': MODELS,",
            "    'datasets': ['muharaf_lines'],",
            "    'limit': 25,            # keep small — paid APIs. bump when happy.",
            "    'prompt_version': 'line-v1',",
            "    'normalization_profiles': ['raw', 'light', 'diac_insensitive'],",
            "    'concurrency': 4,",
            "    'results_dir': 'results',",
            "}",
            "from arabic_ocr_bench.runner import run",
            "from arabic_ocr_bench.report import score_run",
            "run_dir = run(run_cfg)",
            "summary_md = score_run(run_dir, profiles=tuple(run_cfg['normalization_profiles']))",
            "print(open(summary_md).read())",
        ),
        md("## 4. Inspect predictions"),
        PEEK_CELL,
        md("## 5. Save results to Google Drive"),
        DRIVE_SAVE_CELL,
        md(
            "## Notes",
            "",
            "- `cost_usd_total` in the summary is estimated from `configs/models.yaml` pricing.",
            "- Gemini's free tier is rate-limited; the runner retries 429s with backoff.",
            "- Refusals (model declines an image) are logged as `error` and scored as empty output.",
        ),
    ]
    write("02_colab_hosted_apis.ipynb", notebook(cells, gpu=False))


# --------------------------------------------------------------------------- #
# 03 - analysis
# --------------------------------------------------------------------------- #
def build_analysis():
    cells = [
        md(
            "# Arabic Cursive OCR Benchmark — Analysis",
            "",
            "Compare runs and browse the worst errors. Works in Colab or locally.",
            "Point `RESULTS` at a folder that holds one or more run directories",
            "(e.g. `/content/drive/MyDrive/arabic-ocr-bench/results` on Colab, or `results` locally).",
        ),
        code(
            "# Colab only: mount Drive to read saved runs. Skip if running locally.",
            "try:",
            "    from google.colab import drive; drive.mount('/content/drive')",
            "    RESULTS = '/content/drive/MyDrive/arabic-ocr-bench/results'",
            "except Exception:",
            "    RESULTS = 'results'",
            "print('results dir:', RESULTS)",
        ),
        md("## 1. Leaderboard across all runs"),
        code(
            "import json, pandas as pd, pathlib",
            "rows = []",
            "for sc in pathlib.Path(RESULTS).glob('*/scores.jsonl'):",
            "    run = sc.parent.name",
            "    recs = [json.loads(l) for l in sc.read_text().splitlines() if l.strip()]",
            "    d = pd.DataFrame(recs)",
            "    for (model, ds), g in d.groupby(['model','dataset']):",
            "        rows.append({",
            "            'run': run, 'model': model, 'dataset': ds, 'n': len(g),",
            "            'cer_light': round(g['cer_light'].mean(), 4),",
            "            'cer_raw': round(g['cer_raw'].mean(), 4),",
            "            'wer_light': round(g['wer_light'].mean(), 4),",
            "            'exact_light': round(g['exact_light'].mean(), 4),",
            "            'diac_f1': round(g['diac_f1'].mean(), 4),",
            "            'latency_ms': round(g['latency_ms'].mean(), 1),",
            "        })",
            "board = pd.DataFrame(rows).sort_values(['dataset','cer_light'])",
            "board",
        ),
        md("## 2. CER distribution for one run/model"),
        code(
            "RUN = board.iloc[0]['run']; MODEL = board.iloc[0]['model']",
            "recs = [json.loads(l) for l in (pathlib.Path(RESULTS)/RUN/'scores.jsonl').read_text().splitlines() if l.strip()]",
            "s = pd.DataFrame([r for r in recs if r['model']==MODEL])",
            "ax = s['cer_light'].plot(kind='hist', bins=20, title=f'{MODEL} — CER (light) per line')",
            "ax.set_xlabel('CER'); ax",
        ),
        md(
            "## 3. Worst errors — image + ground truth + prediction",
            "",
            "Re-loads the Muharaf line images by row index (needs internet the first time).",
        ),
        code(
            "from datasets import load_dataset",
            "import matplotlib.pyplot as plt",
            "ds = load_dataset('aamijar/muharaf-public', split='test')",
            "",
            "preds = [json.loads(l) for l in (pathlib.Path(RESULTS)/RUN/'predictions.jsonl').read_text().splitlines() if l.strip()]",
            "pmap = {p['sample_id']: p for p in preds if p['model']==MODEL}",
            "worst = s.sort_values('cer_light', ascending=False).head(8)",
            "",
            "for _, r in worst.iterrows():",
            "    row_idx = int(r['sample_id'].split('/')[-1])",
            "    p = pmap[r['sample_id']]",
            "    plt.figure(figsize=(12, 2)); plt.imshow(ds[row_idx]['image']); plt.axis('off')",
            "    plt.title(f\"CER {r['cer_light']:.2f}\"); plt.show()",
            "    print('GT  :', p['ground_truth'])",
            "    print('PRED:', p['cleaned_text'] or '(empty / error: %s)' % p.get('error'))",
            "    print('-' * 80)",
        ),
    ]
    write("03_analysis.ipynb", notebook(cells, gpu=False))


if __name__ == "__main__":
    build_local()
    build_hosted()
    build_analysis()
