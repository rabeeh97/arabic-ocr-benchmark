# First Benchmark — End-to-End Runbook (v1)

**Goal of v1:** one local model, run entirely on Colab's own GPU — **no paid
API, no Gemini** — producing the first published leaderboard for Arabic cursive
script recognition.

| Setting | v1 value |
| --- | --- |
| Model | `qari-ocr-0.3` (`NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct`) — 2B, Arabic-specific, full merged model, runs on a free T4 |
| Dataset | `muharaf_lines` — Muharaf-public `test` split, line-level (`aamijar/muharaf-public`) |
| Sample count | 50 to start, then the full 1,334 |
| Metrics | CER/WER (raw · light · diacritics-insensitive), diacritic F1, latency |
| Compute | Google Colab, T4 GPU (free tier) |
| Cost | $0 |

Everything below refers to files in this repo: notebook
`notebooks/01_colab_local_models.ipynb`, generator `scripts/make_leaderboard.py`.

---

## 0. Prerequisites (once)

1. A Google account (for Colab + Drive).
2. A **GitHub fine-grained token** so Colab can clone this private repo:
   - https://github.com/settings/personal-access-tokens/new → Fine-grained
   - Resource owner `rabeeh97` · Repository access → Only select → `arabic-ocr-benchmark`
   - Permissions → Repository → **Contents: Read and write** (write lets the
     notebook push results back; use Read-only if you'll download manually)
   - Generate, copy the token string.
3. *(optional, recommended)* A **Hugging Face read token**
   (https://huggingface.co/settings/tokens) — speeds the Muharaf download and
   avoids anonymous rate limits.

---

## 1. Open the notebook in Colab

**Option A — straight from GitHub (keeps it in sync):**
1. https://colab.research.google.com → **File → Open notebook → GitHub tab**
2. Tick **Include private repos**; authorize Colab for GitHub if prompted
3. Repository `rabeeh97/arabic-ocr-benchmark`, branch `main`
4. Open `notebooks/01_colab_local_models.ipynb`

**Option B — upload:** on GitHub open the `.ipynb`, **Download raw file**, then in
Colab **File → Upload notebook**.

---

## 2. Configure the runtime

1. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
2. Left sidebar **🔑 (Secrets) → Add new secret**, for each:
   | Name | Value | Notebook access |
   | --- | --- | --- |
   | `GH_TOKEN` | the GitHub token from step 0.2 | ON |
   | `HF_TOKEN` | the HF token from step 0.3 (optional) | ON |

---

## 3. Run the experiment

**Runtime → Run all.** Cell by cell:

| Cell | What it does | Expect |
| --- | --- | --- |
| 1 — GPU check | asserts a CUDA GPU | prints `Tesla T4`, memory line |
| 2 — clone | clones the repo with `GH_TOKEN` | prints the latest commit line |
| 3 — HF cache | points HF at `/content` (Colab scratch, not your machine) | `HF_TOKEN set` if you added it |
| 4 — install | `pip install -e '.[local]'` | ~2–4 min, some resolver noise is fine |
| 5 — pick model | sets `MODEL='qari-ocr-0.3'`, `LIMIT=50` | no output |
| 6 — 1-image sanity | loads the model, transcribes one line | shows the line image, `GT:` and `PRED:`; first model download ~3–5 min |
| 7 — benchmark | runs all 50 lines, scores them | progress, then the `summary.md` table |
| 8 — inspect | first 20 predictions as a table | GT vs prediction |
| 9 — leaderboard | builds `results/leaderboard.html` | renders inline |
| 10 — save to Drive | copies the run to `MyDrive/arabic-ocr-bench/results/` | `saved -> ...` |
| 11 — commit to repo | `git add/commit/push` the run + leaderboard | push output (needs write-scoped `GH_TOKEN`) |

If cell 6 fails, fix that before the full run — see Troubleshooting.

**Scale up:** once the 50-line run looks sane, set `LIMIT = None` in cell 5 and
re-run cells 5–11. The full test split is ~1,334 lines; on a T4, Qari-OCR is
roughly 1–3 s/line, so ~20–60 min. The runner caches per line, so a disconnect
mid-run resumes on re-run.

---

## 4. Where the results are

After a run, three copies (depending on which cells you ran):

| Location | How to get it |
| --- | --- |
| Colab VM: `results/<run_id>/` | file browser (left) → download folder — **lost when Colab recycles** |
| Google Drive: `MyDrive/arabic-ocr-bench/results/<run_id>/` | persistent; cell 10 |
| This repo: `results/<run_id>/` on `main` | persistent + versioned; cell 11 |

Each `results/<run_id>/` holds `summary.md`, `summary.csv`, `scores.jsonl`,
`predictions.jsonl`, `run_config.snapshot.yaml`. See `results/README.md`.

---

## 5. Bring the results back here

**If cell 11 pushed them** (write-scoped `GH_TOKEN`):

```bash
cd /Volumes/PortableSSD/repos/arabic_ocr
source env.sh
git pull
ls results/
```

**If not** (Read-only token / you skipped cell 11): in Colab's file browser,
right-click `results/<run_id>` → Download, unzip into `results/` here, then:

```bash
cd /Volumes/PortableSSD/repos/arabic_ocr && source env.sh
git add results/<run_id> && git commit -m "results: qari-ocr-0.3 x muharaf_lines" && git push
```

---

## 6. Regenerate the leaderboard locally

```bash
source env.sh
uv run python scripts/make_leaderboard.py results -o results/leaderboard.html
open results/leaderboard.html          # eyeball it
git add results/leaderboard.html results/leaderboard.json && git commit -m "leaderboard: add qari-ocr-0.3 run" && git push
```

`results/leaderboard.html` is a self-contained, theme-aware page (all runs under
`results/` merged, sorted by CER-light).

---

## 7. Publish the artifact (first version)

The artifact is that leaderboard page, published as a shareable Claude Artifact.

1. Make sure the real run is committed and `results/leaderboard.html` is regenerated (steps 5–6).
2. Tell Claude Code in this repo: **"publish results/leaderboard.html as the benchmark artifact"** (paste the `summary.md` table too if the results aren't pushed yet).
3. Claude publishes it and returns a private URL. Updating later (more models,
   full split) = regenerate `leaderboard.html`, ask Claude to republish to the
   **same URL** — the link stays stable.

The artifact starts private to your Claude account; share it when you're ready.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Cell 1 says `NO GPU` | It still runs on CPU (fp32, slow, 4-bit models skipped). For a real run: Runtime → Change runtime type → T4 GPU, then Run all. |
| Cell 2: clone asks for a password / 403 | `GH_TOKEN` missing, wrong name, Notebook-access off, or not granted `Contents` on this repo |
| Cell 4: install errors about `torch` | usually harmless (Colab's torch already satisfies the pin); continue |
| Cell 6: `KeyError` / chat-template error | model/processor mismatch — try `MODEL='qwen2.5-vl-3b'`; report the traceback |
| Cell 6/7: CUDA out of memory | lower `LIMIT`; use `qari-ocr-0.3` or `qwen2.5-vl-3b` (not 7B) on a T4 |
| `'Parameter' object has no attribute 'compress_statistics'` | `bitsandbytes` is imported and has patched torch. `!pip uninstall -y bitsandbytes`, remove any `import bitsandbytes` cell, restart the runtime. The harness never needs it for these models. |
| `Unrecognized model ... model_type` | pointing at a LoRA-adapter repo as if it were a full model. Use `qari-ocr-0.3` (full model); `qari-ocr-0.2` is already wired as base + adapter. |
| Muharaf download stalls (~25 MB) | add the `HF_TOKEN` secret and re-run cell 7 |
| Cell 11: `push` rejected | `GH_TOKEN` is Read-only — regenerate it with `Contents: Read and write`, or download manually (step 5) |
| Gemini/paid stuff appears | wrong notebook — v1 is `01_colab_local_models.ipynb`, no API keys needed |

---

## Moving forward after v1

1. **Full split** — `LIMIT = None`, re-run. This is the real v1 number.
2. **More local models** — cell 5: `MODEL='qari-ocr-0.3'`, then `'qwen2.5-vl-3b'`,
   `'qwen2.5-vl-7b'`. Each adds a leaderboard row.
3. **Hosted models** — `notebooks/02_colab_hosted_apis.ipynb` with a free Gemini
   key (or paid OpenAI/Claude). Same dataset, same metrics → directly comparable.
4. **Error analysis** — `notebooks/03_analysis.ipynb`: worst-N lines with image +
   GT + each model's output.
5. **Page track** — Phase 3 in `docs/plan.md` (raw Muharaf pages).
6. **Fine-tuning** — Phase 5, once the zero-shot leaderboard is trusted.
