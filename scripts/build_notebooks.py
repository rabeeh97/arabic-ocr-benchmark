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
GPU_CHECK_CELL = code(
    "import torch",
    "HAS_GPU = torch.cuda.is_available()",
    "if HAS_GPU:",
    "    print('GPU:', torch.cuda.get_device_name(0))",
    "    !nvidia-smi --query-gpu=memory.total,memory.used --format=csv",
    "else:",
    "    print('NO GPU — will run on CPU (much slower; big 4-bit models are skipped).')",
    "    print('For a real run: Runtime -> Change runtime type -> T4 GPU.')",
)

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
    "# src-layout package: importable straight from the checkout even if the",
    "# `pip install -e` cell below fails or is skipped.",
    "import sys",
    "for _p in (os.path.join(os.getcwd(), 'src'), os.getcwd()):",
    "    if _p not in sys.path: sys.path.insert(0, _p)",
    "print(subprocess.run(['git','log','--oneline','-1'], capture_output=True, text=True).stdout)",
)

HF_CACHE_CELL = code(
    "# Keep all downloads on Colab's own scratch disk (ephemeral, ~78 GB) --",
    "# never the local machine. Nothing here touches your computer.",
    "import os",
    "os.environ['HF_HOME'] = '/content/hf_cache'",
    "os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'",
    "os.environ['TOKENIZERS_PARALLELISM'] = 'false'",
    "# Optional HF_TOKEN secret: lifts anonymous rate limits and speeds the",
    "# Muharaf download (and is required for gated models like PaliGemma's base).",
    "try:",
    "    from google.colab import userdata",
    "    _hf = userdata.get('HF_TOKEN')",
    "    if _hf:",
    "        os.environ['HF_TOKEN'] = _hf",
    "        print('HF_TOKEN set')",
    "except Exception:",
    "    pass",
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

LEADERBOARD_CELL = code(
    "# Build the shareable HTML leaderboard from every run under results/",
    "!python scripts/make_leaderboard.py results -o results/leaderboard.html",
    "from IPython.display import HTML",
    "HTML(open('results/leaderboard.html').read())",
)

COMMIT_RESULTS_CELL = code(
    "# Push this run's results back to the repo (needs the GH_TOKEN secret used to",
    "# clone). Skips silently if no token. Commits results/<run_id>/ + leaderboard.",
    "import subprocess, os",
    "if token:",
    "    subprocess.run(['git','config','user.email','colab@arabic-ocr-bench'], check=True)",
    "    subprocess.run(['git','config','user.name','colab-runner'], check=True)",
    "    subprocess.run(['git','add','-f', f'results/{run_dir.name}', 'results/leaderboard.html', 'results/leaderboard.json'], check=True)",
    "    msg = f'results: {\", \".join(run_cfg[\"models\"])} x {\", \".join(run_cfg[\"datasets\"])} (n={run_cfg.get(\"limit\") or \"all\"})'",
    "    r = subprocess.run(['git','commit','-m',msg], capture_output=True, text=True)",
    "    print(r.stdout or r.stderr)",
    "    push = subprocess.run(['git','push','origin','HEAD:main'], capture_output=True, text=True)",
    "    print(push.stdout or push.stderr)",
    "else:",
    "    print('no GH_TOKEN -- results are on Drive only; download them from the file browser')",
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
        GPU_CHECK_CELL,
        md("## 2. Clone the benchmark repo"),
        CLONE_CELL,
        md("## 3. Install the harness (with local-model deps)"),
        HF_CACHE_CELL,
        code(
            "!pip install -q -e '.[local]'",
            "# Colab ships a recent torch already; this pulls transformers>=4.49, peft, qwen-vl-utils, ...",
            "!pip uninstall -y -q torchao 2>/dev/null  # old Colab torchao breaks model load; unused",
        ),
        md(
            "## 4. Pick the model(s)",
            "",
            "One name for a single run, or a list for a **sweep** — the runner loads",
            "each model, runs every line, frees it, then loads the next, so a list fits",
            "on one T4. Names are in `configs/models.yaml`:",
            "`qari-ocr-0.3`, `qari-ocr-0.2`, `qwen2.5-vl-3b` (all fp16-safe on a free T4),",
            "`qwen2.5-vl-7b` (needs 4-bit), `paligemma-arabic-hw` (gated + line-only).",
            "The list below mirrors `configs/run.sweep-local.yaml`.",
        ),
        code(
            "MODELS = ['qari-ocr-0.3']            # e.g. ['qari-ocr-0.3','qari-ocr-0.2','qwen2.5-vl-3b']",
            "# kitab_khatt | kitab_muharaf | kitab_historicalbooks | kitab_hindawi | muharaf_lines",
            "DATASET = 'kitab_khatt'             # ungated, 200 lines; muharaf_lines needs HF_TOKEN",
            "LIMIT  = 50                          # lines; None = the whole split",
            "",
            "run_cfg = {",
            "    'models': MODELS,",
            "    'datasets': [DATASET],",
            "    'limit': LIMIT,",
            "    'prompt_version': 'line-v2',",
            "    'normalization_profiles': ['raw', 'light', 'diac_insensitive'],",
            "    'concurrency': 1,",
            "    'results_dir': 'results',",
            "}",
        ),
        md(
            "## 5. Sanity check on one image",
            "",
            "Loads the model and transcribes a single line — surfaces any load / prompt /",
            "dtype problem in ~1 min instead of after the whole run.",
        ),
        code(
            "from arabic_ocr_bench.models.registry import build_model, load_model_configs",
            "from arabic_ocr_bench.datasets.registry import build_dataset, load_dataset_configs",
            "from arabic_ocr_bench.prompting import get_prompt",
            "",
            "_m = load_model_configs(); _d = load_dataset_configs()",
            "sample = next(iter(build_dataset(DATASET, _d[DATASET], limit=1)))",
            "display(sample.image); print('GT   :', sample.ground_truth)",
            "for _name in MODELS:",
            "    try:",
            "        _client = build_model(_name, _m[_name])",
            "        pred = _client.transcribe(sample.image, get_prompt('line-v2'))",
            "        print(f'{_name:22} -> ' + (pred.cleaned_text or f'(empty / error: {pred.error})'))",
            "        if hasattr(_client, 'close'): _client.close()   # free VRAM before the real run",
            "    except Exception as e:",
            "        print(f'{_name:22} -> LOAD/RUN FAILED: {type(e).__name__}: {e}')",
            "# non-fatal: the real run below skips a model that errors here too",
        ),
        md("## 6. Run the benchmark"),
        code(
            "from arabic_ocr_bench.runner import run",
            "from arabic_ocr_bench.report import score_run",
            "",
            "run_dir = run(run_cfg)",
            "summary_md = score_run(run_dir, profiles=tuple(run_cfg['normalization_profiles']))",
            "print(open(summary_md).read())",
        ),
        md("## 7. Inspect predictions"),
        PEEK_CELL,
        md(
            "## 8. Export sample transcriptions",
            "",
            "Picks a few illuminating lines (shared anchors + this model's best/worst)",
            "and writes `results/<run_id>/samples.json` with the image, ground truth and",
            "prediction — this is what the leaderboard's *Sample transcriptions* section shows.",
        ),
        code(
            "from arabic_ocr_bench.samples import export_run_samples",
            "from datasets import load_dataset",
            "_HF = {'muharaf_lines': ('aamijar/muharaf-public', 'test'),",
            "       'kitab_khatt': ('ahmedheakl/arocrbench_khatt', 'train'),",
            "       'kitab_muharaf': ('ahmedheakl/arocrbench_muharaf', 'train'),",
            "       'kitab_historicalbooks': ('ahmedheakl/arocrbench_historicalbooks', 'train'),",
            "       'kitab_hindawi': ('ahmedheakl/arocrbench_hindawi', 'train')}",
            "_hf_id, _split = _HF[DATASET]",
            "_ds = load_dataset(_hf_id, split=_split)",
            "export_run_samples(run_dir, get_image=lambda i: _ds[int(i)]['image'], k=3)",
            "print('wrote', run_dir / 'samples.json')",
        ),
        md("## 9. Leaderboard"),
        LEADERBOARD_CELL,
        md("## 10. Save results to Google Drive"),
        DRIVE_SAVE_CELL,
        md(
            "## 11. Commit results back to the repo",
            "",
            "So they show up here and on GitHub without a manual download.",
        ),
        COMMIT_RESULTS_CELL,
        md(
            "## Next",
            "",
            "- **Sweep**: set `MODELS = ['qari-ocr-0.3','qari-ocr-0.2','qwen2.5-vl-3b']` and re-run cells 4-11. One `results/<run_id>/` holds all of them; the leaderboard shows a row each.",
            "- Set `LIMIT = None` for the whole split (kitab_khatt = 200 lines, muharaf_lines = 1,334).",
            "- Hosted models (Gemini/GPT/Claude) → `02_colab_hosted_apis.ipynb`, same dataset & metrics.",
            "- `03_analysis.ipynb` — worst-error browser (image + GT + each model's output).",
            "- Full walkthrough incl. publishing the artifact: `docs/first-benchmark-runbook.md`.",
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
            "Same sweep engine as `04_colab_sweep.ipynb` (partial then full phase,",
            "per-phase checkpoint + push, resume-safe) — but the models are hosted",
            "APIs, so **no GPU**. Add whichever Colab secrets you have keys for:",
            "`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (free tier at aistudio.google.com/apikey),",
            "`OPENAI_API_KEY`. Only models with a key present are run.",
        ),
        md("## 1. Repo"),
        ROBUST_CLONE_CELL,
        md("## 2. Install + caches"),
        HF_CACHE_CELL,
        code("!pip install -q -e '.'   # core only — no torch/transformers"),
        md("## 3. Mount Drive"),
        code("from google.colab import drive", "drive.mount('/content/drive')"),
        md(
            "## 4. Keys -> models",
            "",
            "Copies whatever provider secrets you've added into the environment, then",
            "`available_hosted_models()` selects **every `client: litellm` model in",
            "`configs/models.yaml` whose key is present** — so adding a model there is",
            "all it takes. Add these Colab secrets as needed: `ANTHROPIC_API_KEY`,",
            "`GEMINI_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`,",
            "`OLLAMA_API_KEY` + `OLLAMA_API_BASE`.",
        ),
        code(
            "import os",
            "from google.colab import userdata",
            "for k in ('ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'OPENAI_API_KEY',",
            "          'MISTRAL_API_KEY', 'OPENROUTER_API_KEY',",
            "          'OLLAMA_API_KEY', 'OLLAMA_API_BASE', 'HF_TOKEN'):",
            "    try:",
            "        v = userdata.get(k)",
            "        if v: os.environ[k] = v",
            "    except Exception:",
            "        pass",
            "",
            "from arabic_ocr_bench.models.registry import available_hosted_models",
            "MODELS = available_hosted_models()",
            "assert MODELS, 'no provider API-key secrets found — add at least one'",
            "print('will run:', MODELS)",
            "",
            "# KITAB-Bench subsets are small + ungated: kitab_khatt (handwriting),",
            "# kitab_muharaf, kitab_historicalbooks, kitab_hindawi. Or 'muharaf_lines'.",
            "DATASETS = ['kitab_khatt']   # add more, e.g. ['kitab_khatt', 'kitab_hindawi']",
            "PARTIAL_LIMIT = 20       # paid APIs — keep the partial small",
            "RUN_FULL = False         # KITAB subsets are only 200-500 rows; True does the whole split",
        ),
        md("## 5. Run the sweep — checkpoints + pushes after every model/phase"),
        _hosted_sweep_cell(),
        md("## 6. Final leaderboard"),
        code(
            "!python scripts/make_leaderboard.py results -o results/leaderboard.html",
            "from IPython.display import HTML",
            "HTML(open('results/leaderboard.html').read())",
        ),
        md(
            "## Notes",
            "",
            "- `cost_usd_total` in the summary is estimated from `configs/models.yaml` pricing.",
            "- Output is capped at 256 tokens/line (a line is <=181 chars) — same idea as the local 160-token cap.",
            "- Gemini's free tier is rate-limited; the runner retries 429s with backoff.",
            "- Refusals are detected (English + Arabic) and scored as empty, not as the apology text.",
            "- Re-running skips models already in `results/sweep_manifest.json`.",
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
            "Re-loads the line images by row index (needs internet the first time).",
            "The dataset is read from the run's sample ids (`<dataset>/<split>/<idx>`).",
        ),
        code(
            "from datasets import load_dataset",
            "import matplotlib.pyplot as plt",
            "_HF = {'muharaf_lines': ('aamijar/muharaf-public', 'test'),",
            "       'kitab_khatt': ('ahmedheakl/arocrbench_khatt', 'train'),",
            "       'kitab_muharaf': ('ahmedheakl/arocrbench_muharaf', 'train'),",
            "       'kitab_historicalbooks': ('ahmedheakl/arocrbench_historicalbooks', 'train'),",
            "       'kitab_hindawi': ('ahmedheakl/arocrbench_hindawi', 'train')}",
            "_dsname = s.iloc[0]['sample_id'].split('/')[0] if 'sample_id' in s else 'muharaf_lines'",
            "_hf_id, _split = _HF.get(_dsname, ('aamijar/muharaf-public', 'test'))",
            "ds = load_dataset(_hf_id, split=_split)",
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


# --------------------------------------------------------------------------- #
# 04 - unattended sweep
# --------------------------------------------------------------------------- #
ROBUST_CLONE_CELL = code(
    "# Bulletproof checkout: works whether the repo is missing, at the wrong",
    "# path, or has local edits. Always ends up at the latest origin/main.",
    "import os",
    "REPO = '/content/arabic-ocr-benchmark'",
    "token = None",
    "try:",
    "    from google.colab import userdata",
    "    token = userdata.get('GH_TOKEN')",
    "except Exception as e:",
    "    print('no GH_TOKEN secret:', e)",
    "url = f\"https://{(token + '@') if token else ''}github.com/rabeeh97/arabic-ocr-benchmark.git\"",
    "if os.path.isdir(REPO + '/.git'):",
    "    os.system(f'git -C {REPO} fetch --depth 1 origin main && git -C {REPO} reset --hard origin/main')",
    "    # drop untracked leftovers (e.g. a stale sweep_manifest.json) so the",
    "    # manifest is always exactly what's in the repo. Keeps gitignored _cache.",
    "    os.system(f'git -C {REPO} clean -fd -- results')",
    "else:",
    "    os.system(f'rm -rf {REPO} && git clone --depth 1 {url} {REPO}')",
    "os.chdir(REPO)",
    "# src-layout package: importable straight from the checkout even if the",
    "# `pip install -e` cell fails or is skipped (this is what caused the old",
    "# 'No module named arabic_ocr_bench' in the sweep cell).",
    "import sys",
    "for _p in (os.path.join(os.getcwd(), 'src'), os.getcwd()):",
    "    if _p not in sys.path: sys.path.insert(0, _p)",
    "print(os.popen('git log --oneline -3').read())",
)

def _sweep_run_cell(concurrency: int = 1):
    return code(
    "import os, sys, shutil, subprocess, pathlib",
    "# make the src-layout package importable even if `pip install -e` didn't take",
    "for _p in (os.path.join(os.getcwd(), 'src'), os.getcwd()):",
    "    if _p not in sys.path: sys.path.insert(0, _p)",
    "from arabic_ocr_bench.sweep import run_sweep",
    "from arabic_ocr_bench.samples import export_run_samples",
    "",
    "DRIVE = pathlib.Path('/content/drive/MyDrive/arabic-ocr-bench')",
    "DRIVE_OK = DRIVE.parent.exists()   # is Drive mounted?",
    "",
    "_DS = {}",
    "_HF_SPLIT = {",
    "    'muharaf_lines': ('aamijar/muharaf-public', 'test'),",
    "    'kitab_khatt': ('ahmedheakl/arocrbench_khatt', 'train'),",
    "    'kitab_muharaf': ('ahmedheakl/arocrbench_muharaf', 'train'),",
    "    'kitab_historicalbooks': ('ahmedheakl/arocrbench_historicalbooks', 'train'),",
    "    'kitab_hindawi': ('ahmedheakl/arocrbench_hindawi', 'train'),",
    "}  # add datasets here",
    "def _get_image_fn(dataset):",
    "    # row-index -> PIL image, for the sample export",
    "    if dataset not in _HF_SPLIT:",
    "        return lambda i: None   # unknown dataset -> samples show 'image unavailable'",
    "    if dataset not in _DS:",
    "        from datasets import load_dataset",
    "        hf_id, split = _HF_SPLIT[dataset]",
    "        _DS[dataset] = load_dataset(hf_id, split=split)",
    "    ds = _DS[dataset]",
    "    return lambda i: ds[int(i)]['image']",
    "",
    "def _git(*a):",
    "    return subprocess.run(['git', *a], capture_output=True, text=True)",
    "",
    "def _mirror_to_drive(run_dir):",
    "    if not DRIVE_OK:",
    "        return",
    "    (DRIVE / 'results').mkdir(parents=True, exist_ok=True)",
    "    if pathlib.Path('results/_cache').is_dir():",
    "        shutil.copytree('results/_cache', DRIVE / '_cache', dirs_exist_ok=True)",
    "    shutil.copytree(run_dir, DRIVE / 'results' / pathlib.Path(run_dir).name, dirs_exist_ok=True)",
    "    for f in ('sweep_manifest.json', 'leaderboard.html', 'leaderboard.json'):",
    "        if pathlib.Path('results', f).is_file():",
    "            shutil.copy2(pathlib.Path('results', f), DRIVE / 'results' / f)",
    "",
    "def checkpoint(entry, run_dir):",
    "    # Called by run_sweep AFTER a phase is scored. Persist the results here.",
    "    # RAISE if they don't make it to the repo -> the phase is rolled back and",
    "    # retried next run (so a 'done' entry always means 'in the repo').",
    "    run_dir = str(run_dir)",
    "    try:",
    "        export_run_samples(run_dir, get_image=_get_image_fn(entry['dataset']), k=3)",
    "    except Exception as e:",
    "        print('   sample export failed:', e)",
    "    subprocess.run(['python', 'scripts/make_leaderboard.py', 'results',",
    "                    '-o', 'results/leaderboard.html'])",
    "    _mirror_to_drive(run_dir)",
    "    if not token:",
    "        print('   no GH_TOKEN -> Drive only (this counts as persisted)')",
    "        return",
    "    _git('config', 'user.email', 'colab@arabic-ocr-bench')",
    "    _git('config', 'user.name', 'colab-sweep')",
    "    _git('add', '-f', run_dir, 'results/_cache', 'results/sweep_manifest.json',",
    "         'results/leaderboard.html', 'results/leaderboard.json')",
    "    _git('commit', '-m', f\"sweep: {entry['model']} {entry['phase']} \"",
    "         f\"CER-light {entry['cer_micro_light']*100:.1f}% n={entry['n']}\")",
    "    _git('pull', '--rebase', '--autostash', 'origin', 'main')",
    "    p = _git('push', 'origin', 'HEAD:main')",
    "    if p.returncode != 0:",
    "        raise RuntimeError('git push failed: ' + (p.stderr.strip()[:300] or 'unknown'))",
    "    print('   push: ok')",
    "",
    "# The manifest is git-tracked (comes from the fresh checkout) — it is the",
    "# single source of truth for what's done. Only restore the per-line cache",
    "# from Drive, to make a re-running phase fast.",
    "if DRIVE_OK and (DRIVE / '_cache').is_dir():",
    "    shutil.copytree(DRIVE / '_cache', 'results/_cache', dirs_exist_ok=True)",
    "    print('restored line cache from Drive')",
    "",
    "# One run_sweep per dataset. Each is independently resumable and wrapped so",
    "# a dataset that fails outright (e.g. a gated download) doesn't stop the rest.",
    "# Per-model / per-phase / per-line failures are already caught inside run_sweep.",
    "for _ds in DATASETS:",
    "    print(f'\\n########## DATASET: {_ds} ##########')",
    "    try:",
    "        run_sweep(",
    "            models=MODELS, dataset=_ds,",
    "            partial_limit=PARTIAL_LIMIT, run_full=RUN_FULL,",
    f"            concurrency={concurrency},",
    "            manifest_path='results/sweep_manifest.json', results_dir='results',",
    "            on_progress=print, on_done=checkpoint,",
    "        )",
    "    except Exception as e:",
    "        import traceback; traceback.print_exc()",
    "        print(f'!!! DATASET {_ds} FAILED ({type(e).__name__}: {e}) — continuing !!!')",
    "print('\\n=== SWEEP COMPLETE ===')",
    )


SWEEP_RUN_CELL = _sweep_run_cell(1)


def _hosted_sweep_cell():
    return _sweep_run_cell(4)


def build_sweep():
    cells = [
        md(
            "# Arabic Cursive OCR Benchmark — Unattended Sweep",
            "",
            "Benchmarks several local models across one or more datasets. **Run all and",
            "walk away**. After every (model, dataset, phase) it rebuilds the",
            "leaderboard, mirrors results to Drive, and pushes them to the repo — so a",
            "disconnect never loses more than the model in flight.",
            "",
            "**Resumable + fault-tolerant:** re-running skips anything already in",
            "`results/sweep_manifest.json` (and, within a half-done phase, any line",
            "already in `results/_cache/`). A model that OOMs, a dataset that won't",
            "download, a line the model refuses — each is caught and logged; the sweep",
            "keeps going. Safe to Run all again after a timeout.",
            "",
            "**Before running:** `Runtime -> T4 GPU`; add `GH_TOKEN` (write-scoped) and",
            "`HF_TOKEN` secrets (see `notebooks/README.md`).",
        ),
        md("## 1. GPU (falls back to CPU)"),
        GPU_CHECK_CELL,
        md("## 2. Repo"),
        ROBUST_CLONE_CELL,
        md("## 3. Install + caches"),
        HF_CACHE_CELL,
        code(
            "!pip install -q -e '.[local]'",
            "# Colab ships an old torchao; recent transformers hard-errors importing it",
            "# at model-load ('incompatible version of torchao'). We don't use it.",
            "!pip uninstall -y -q torchao 2>/dev/null; echo 'torchao removed (unused)'",
        ),
        md("## 4. Mount Drive"),
        code(
            "from google.colab import drive",
            "drive.mount('/content/drive')",
        ),
        md(
            "## 5. Config",
            "",
            "`PARTIAL_LIMIT` runs a quick pass on every model first (fast feedback);",
            "with `RUN_FULL = True` it then does the whole split on each. Both phases",
            "are checkpointed independently (manifest + per-line cache, both pushed),",
            "so interrupting only costs a little repeated cheap work.",
            "",
            "**Dataset** — `DATASET` picks what to run on. `kitab_khatt` (KITAB-Bench /",
            "KHATT handwriting, 200 lines, ungated) is the default; other options:",
            "`kitab_muharaf`, `kitab_historicalbooks`, `kitab_hindawi`, or",
            "`muharaf_lines` (1,334 lines, needs HF_TOKEN). Change it and re-run —",
            "results for a dataset already in the manifest are skipped, not repeated.",
            "",
            "The 7B models (`qwen2.5-vl-7b`, `ain-7b`) load in **4-bit** — fine on a",
            "T4. **On CPU** (no GPU) 4-bit isn't supported and full fp32 7B OOMs the",
            "12 GB box, so they're dropped and the run shrinks to a tiny partial-only",
            "pass — enough to prove the pipeline, not a real benchmark.",
        ),
        code(
            "MODELS = [",
            "    'qari-ocr-0.2',    # Arabic OCR, 2B, raw-text focused (lead)",
            "    'qari-ocr-0.3',    # Arabic OCR, 2B, structure-first (emits markup)",
            "    'qwen2-vl-2b',     # untuned base — anchor",
            "    'qwen2.5-vl-3b',   # general VLM, 3B",
            "    'qwen2.5-vl-7b',   # general VLM, 7B (4-bit on GPU)",
            "    'ain-7b',          # Arabic-centric VLM, 7B (4-bit on GPU)",
            "    'minicpm-v',       # multilingual doc-OCR VLM, 8B (4-bit on GPU)",
            "    'trocr-large-arabic',  # dedicated HTR (TrOCR-Large, KHATT+IAM); CPU-ok",
            "]",
            "# any of: kitab_khatt, kitab_muharaf, kitab_historicalbooks, kitab_hindawi, muharaf_lines",
            "DATASETS = ['kitab_khatt']    # add more, e.g. ['kitab_khatt', 'kitab_historicalbooks']",
            "PARTIAL_LIMIT = 25       # quick pass per model",
            "RUN_FULL = True          # then the whole split for each dataset",
            "",
            "import yaml",
            "_mc = yaml.safe_load(open('configs/models.yaml'))",
            "if not HAS_GPU:",
            "    MODELS = [m for m in MODELS if not _mc.get(m, {}).get('load_in_4bit')]",
            "    PARTIAL_LIMIT = min(PARTIAL_LIMIT, 10)",
            "    RUN_FULL = False",
            "    print(f'CPU mode -> MODELS={MODELS}, PARTIAL_LIMIT={PARTIAL_LIMIT}, RUN_FULL=False')",
        ),
        code(
            "# bitsandbytes only if a 4-bit model is in the list (GPU mode)",
            "if HAS_GPU and any(_mc.get(m, {}).get('load_in_4bit') for m in MODELS):",
            "    !pip install -q bitsandbytes",
            "    print('bitsandbytes installed for 4-bit models')",
        ),
        md("## 6. Run the sweep — this is the long one; leave it running"),
        SWEEP_RUN_CELL,
        md("## 7. Final leaderboard"),
        code(
            "!python scripts/make_leaderboard.py results -o results/leaderboard.html",
            "from IPython.display import HTML",
            "HTML(open('results/leaderboard.html').read())",
        ),
        md(
            "## Notes",
            "",
            "- **Resume after a timeout:** just Run all again. Completed (model, phase)",
            "  pairs are skipped; a partially-done model restarts but its already-scored",
            "  lines come from the cache (restored from Drive), so it's quick.",
            "- **Force a re-run** of one model: delete its entries from",
            "  `results/sweep_manifest.json` (and optionally its `results/<run_id>/`).",
            "- **Add models:** edit `MODELS` and Run all — only the new ones execute.",
            "- Hosted models (Gemini/GPT/Claude): `02_colab_hosted_apis.ipynb`.",
        ),
    ]
    write("04_colab_sweep.ipynb", notebook(cells, gpu=True))


if __name__ == "__main__":
    build_local()
    build_hosted()
    build_analysis()
    build_sweep()
