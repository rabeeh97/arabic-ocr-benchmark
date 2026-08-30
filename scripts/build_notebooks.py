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
            "LIMIT  = 50                          # Muharaf test lines; None = all 1,334",
            "",
            "run_cfg = {",
            "    'models': MODELS,",
            "    'datasets': ['muharaf_lines'],",
            "    'limit': LIMIT,",
            "    'prompt_version': 'line-v1',",
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
            "_client = build_model(MODELS[0], _m[MODELS[0]])",
            "sample = next(iter(build_dataset('muharaf_lines', _d['muharaf_lines'], limit=1)))",
            "pred = _client.transcribe(sample.image, get_prompt('line-v1'))",
            "display(sample.image)",
            "print('MODEL:', MODELS[0])",
            "print('GT   :', sample.ground_truth)",
            "print('PRED :', pred.cleaned_text or f'(empty / error: {pred.error})')",
            "if hasattr(_client, 'close'): _client.close()   # free VRAM before the real run",
            "assert pred.error is None, pred.error",
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
        md("## 8. Leaderboard"),
        LEADERBOARD_CELL,
        md("## 9. Save results to Google Drive"),
        DRIVE_SAVE_CELL,
        md(
            "## 10. Commit results back to the repo",
            "",
            "So they show up here and on GitHub without a manual download.",
        ),
        COMMIT_RESULTS_CELL,
        md(
            "## Next",
            "",
            "- **Sweep**: set `MODELS = ['qari-ocr-0.3','qari-ocr-0.2','qwen2.5-vl-3b']` and re-run cells 4-11. One `results/<run_id>/` holds all of them; the leaderboard shows a row each.",
            "- Set `LIMIT = None` for the full 1,334-line test split.",
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
            "Runs Gemini / OpenAI / Claude against the **Muharaf** test set via their APIs.",
            "No GPU needed. Add whichever of these Colab secrets you have keys for:",
            "`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`",
            "(a free Gemini key: https://aistudio.google.com/apikey).",
        ),
        md("## 1. Clone + install"),
        CLONE_CELL,
        code("!pip install -q -e '.'  # core only — no torch/transformers"),
        HF_CACHE_CELL,
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
        md("## 5. Leaderboard"),
        LEADERBOARD_CELL,
        md("## 6. Save results to Google Drive"),
        DRIVE_SAVE_CELL,
        md("## 7. Commit results back to the repo"),
        COMMIT_RESULTS_CELL,
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
    "else:",
    "    os.system(f'rm -rf {REPO} && git clone --depth 1 {url} {REPO}')",
    "os.chdir(REPO)",
    "print(os.popen('git log --oneline -3').read())",
)

SWEEP_RUN_CELL = code(
    "import shutil, subprocess, pathlib",
    "from arabic_ocr_bench.sweep import run_sweep",
    "",
    "DRIVE = pathlib.Path('/content/drive/MyDrive/arabic-ocr-bench')",
    "DRIVE_OK = DRIVE.parent.exists()   # is Drive mounted?",
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
    "    run_dir = str(run_dir)",
    "    subprocess.run(['python', 'scripts/make_leaderboard.py', 'results',",
    "                    '-o', 'results/leaderboard.html'])",
    "    _mirror_to_drive(run_dir)",
    "    if not token:",
    "        print('   (no GH_TOKEN -> results saved to Drive only)'); return",
    "    _git('config', 'user.email', 'colab@arabic-ocr-bench')",
    "    _git('config', 'user.name', 'colab-sweep')",
    "    _git('add', '-f', run_dir, 'results/sweep_manifest.json',",
    "         'results/leaderboard.html', 'results/leaderboard.json')",
    "    _git('commit', '-m', f\"sweep: {entry['model']} {entry['phase']} \"",
    "         f\"CER-light {entry['cer_micro_light']*100:.1f}% n={entry['n']}\")",
    "    _git('pull', '--rebase', '--autostash', 'origin', 'main')",
    "    p = _git('push', 'origin', 'HEAD:main')",
    "    print('   push:', 'ok' if p.returncode == 0 else p.stderr.strip()[:160])",
    "",
    "# resume: restore the per-line cache from Drive if we have one",
    "if DRIVE_OK and (DRIVE / '_cache').is_dir():",
    "    shutil.copytree(DRIVE / '_cache', 'results/_cache', dirs_exist_ok=True)",
    "    print('restored line cache from Drive')",
    "",
    "run_sweep(",
    "    models=MODELS, dataset=DATASET,",
    "    partial_limit=PARTIAL_LIMIT, run_full=RUN_FULL,",
    "    manifest_path='results/sweep_manifest.json', results_dir='results',",
    "    on_progress=print, on_done=checkpoint,",
    ")",
    "print('\\n=== SWEEP COMPLETE ===')",
)


def build_sweep():
    cells = [
        md(
            "# Arabic Cursive OCR Benchmark — Unattended Sweep",
            "",
            "Benchmarks several local models on **Muharaf** lines, **Run all and walk",
            "away**. After every (model, phase) it rebuilds the leaderboard, mirrors",
            "results to Drive, and pushes them to the repo — so a disconnect never",
            "loses more than the model in flight.",
            "",
            "**Resumable:** re-running skips anything already in",
            "`results/sweep_manifest.json`. Safe to Run all again after a timeout.",
            "",
            "**Before running:** `Runtime -> T4 GPU`; add `GH_TOKEN` (write-scoped) and",
            "`HF_TOKEN` secrets (see `notebooks/README.md`).",
        ),
        md("## 1. GPU"),
        code(
            "import torch",
            "assert torch.cuda.is_available(), 'Runtime -> Change runtime type -> T4 GPU'",
            "print(torch.cuda.get_device_name(0))",
        ),
        md("## 2. Repo"),
        ROBUST_CLONE_CELL,
        md("## 3. Install + caches"),
        HF_CACHE_CELL,
        code("!pip install -q -e '.[local]'"),
        md("## 4. Mount Drive"),
        code(
            "from google.colab import drive",
            "drive.mount('/content/drive')",
        ),
        md(
            "## 5. Config",
            "",
            "`PARTIAL_LIMIT` runs a quick pass on every model first (fast feedback);",
            "with `RUN_FULL = True` it then does the full 1,334-line pass on each.",
            "Both phases are checkpointed independently.",
        ),
        code(
            "MODELS = ['qari-ocr-0.3', 'qari-ocr-0.2', 'qwen2.5-vl-3b']",
            "DATASET = 'muharaf_lines'",
            "PARTIAL_LIMIT = 100      # quick pass per model",
            "RUN_FULL = True          # then the whole test split",
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
