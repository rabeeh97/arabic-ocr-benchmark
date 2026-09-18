# PaddleOCR — eval + fine-tuning integration

PaddleOCR is a fundamentally different stack from the rest of this repo:
**PaddlePaddle, not PyTorch**, and a dedicated detection+recognition
pipeline (a 5M-parameter recognition model here), not a VLM. Nothing here
shares code with `sft_vlm.py`/TRL, and — deliberately — nothing here shares
a **venv** with the training suite either (see §1).

The 3090 launcher is `scripts/run_paddleocr_experiment.py`. It scores the
**untuned** `arabic_PP-OCRv5_mobile_rec` on the same held-out mix the VLM
bench uses, then writes recognition labels for a later GPU fine-tune.
Inference is pinned to **CPU** so it can run while suite_v1 still owns the
GPU.

```powershell
python scripts\run_paddleocr_experiment.py --dir C:\Users\User\a-experiment\paddleocr --detach
python scripts\run_paddleocr_experiment.py --dir C:\Users\User\a-experiment\paddleocr status
```

## 0. What's already verified, on this exact box (2026-09-17)

- `paddlepaddle-gpu==3.2.2` (cu126 wheel) and `paddleocr==3.7.0` **install
  and import cleanly** — no repeat of the WDAC/DLL-loading failure we
  half-expected by analogy with torch's WinError 4551. This was a real
  concern going in (a public GitHub issue reports exactly that failure mode
  on a locked-down Windows Server), tested directly rather than assumed.
- `paddle.device.cuda.get_device_properties(0)` correctly reports the RTX
  3090 — real GPU access confirmed, not just "compiled with CUDA."
- **A real, non-fatal warning to watch**: `paddlepaddle-gpu==3.2.2` is
  compiled against cuDNN 9.9; this box has cuDNN 9.5. Paddle prints "may
  cause serious incompatible bug" but ran successfully in this test. Revisit
  if inference results look wrong in ways this doc's own test didn't catch.
- **Confirmed bug, not a guess**: `arabic_PP-OCRv5_mobile_rec` returns
  Arabic text in **raw left-to-right pixel order**, uncorrected for RTL
  script. `src/arabic_ocr_bench/models/paddleocr_client.py`'s
  `reverse_output` (set `true` on the `arabic-paddleocr-v5` row in
  `configs/models.yaml`) undoes this — do not skip it, the raw output is
  unusable garbage without it.
- **Real quality test** (our own synthetic renderer, ground truth known
  exactly): after fixing the ordering bug, diacritic-insensitive similarity
  ranged **81-97%** on clean, non-diacritized synthetic Arabic — genuinely
  usable as a fast baseline. On a fully-diacritized (harakat) line, the
  model **dropped every diacritic** (none appeared in its output at all) —
  confirms the pre-integration research's caveat that this shared
  8-language, ~2,676-image model is not a real Arabic-diacritics
  specialist. Full transcript:

  | ground truth | predicted (after RTL fix) | note |
  |---|---|---|
  | بسم الله الرحمن الرحيم | بسمالم الدحمنا لر يم | 83% similar, no diacritics to lose |
  | الحمد لله رب العالمين | الحمد له ربالعالمين | 97% similar |
  | السلام عليكم ورحمة الله | الس لام عل يكم و حمةالل | 95% similar |
  | بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ (harakat) | بسمالة الرحمنا | 81% similar *ignoring* diacritics — but zero diacritics survived |

  Expect real (handwritten, degraded) benchmark lines to score worse than
  this clean synthetic best case — this is a ceiling, not a typical result.

## 1. Why a separate venv from the training suite

`paddlepaddle-gpu` bundles its own CUDA/cuDNN wheels (`nvidia-cudnn-cu12`,
`nvidia-cusolver-cu12`, etc. — ~600 MB of downloads), the same
self-contained-wheel model `torch` uses. Installing both into the suite's
one shared venv (`_shared/venv`, see the multi-model runbook §10) is
**untested for conflicts** and risks destabilizing the training suite for a
side evaluation — not worth the risk. Keep PaddleOCR in its own venv:

```powershell
python -m venv C:\Users\User\a-experiment\paddleocr_venv
C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe -m pip install paddleocr
```

(This repo needs Python >=3.11. The original `paddleocr_venv` on the box
is 3.10; `run_paddleocr_experiment.py`'s install phase builds
`C:\Users\User\a-experiment\paddleocr_venv311` next to it and installs
the same `paddlepaddle-gpu==3.2.2` + `paddleocr` wheels there. Leave the
3.10 tree in place.)

Console output containing Arabic text needs `$env:PYTHONIOENCODING='utf-8'`
set first, or `print()` crashes with the same `UnicodeEncodeError` this
repo's other launcher scripts already guard against for file writes — this
is the one place that guard doesn't reach, since it's the *console's*
default codepage, not a file open call. The launcher sets `PYTHONIOENCODING`
and `PYTHONUTF8` itself.

## 2. Evaluating the raw model through this repo's own benchmark

`arabic-paddleocr-v5` is registered in `configs/models.yaml` (`client:
paddleocr`). The dataset list lives in
`configs/training/paddleocr/baseline.run.yaml` (Muharaf test + the 3090
held-out mix). The launcher generates one `bench run` per dataset, smoke
then full, under `C:\Users\User\a-experiment\paddleocr\bench\`.

It **must** run from `paddleocr_venv`, not the suite's shared venv, and it
**must not** share a `bench run` invocation with the torch-based models.
The launcher also sets `CUDA_VISIBLE_DEVICES=` and `PADDLEOCR_DEVICE=cpu`
so a concurrent VLM train keeps the 3090. The rec model is 5M params; CPU
is enough for a line-level sweep.

Hand-rolled equivalent (one dataset):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:PADDLEOCR_DEVICE='cpu'
$env:PYTHONIOENCODING='utf-8'
C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe -m pip install -e C:\Users\User\a-experiment\aocr-repo
C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe -m arabic_ocr_bench.cli run --config configs\training\paddleocr\baseline.run.yaml --report
```

(`baseline.run.yaml` lists every dataset at once; the launcher splits them
so a failure on one set doesn't discard the rest.)

## 3. Fine-tuning — labels from the same mix, training waits for the GPU

**Label prep (CPU, included in the launcher `labels` phase):**
`scripts/prep_paddleocr_labels.py` converts suite_v1's `data.sources` (the
same Muharaf/AraMS/RASAM mix a Qwen fine-tune trains on) into PaddleOCR's
`image_path<TAB>label` files plus PaddleX's `train.txt` / `val.txt`. Output
lands in `<dir>/labels/suite_v1/`, not the git tree.

```powershell
# already the last phase of run_paddleocr_experiment.py; or by hand:
C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe scripts\prep_paddleocr_labels.py --config configs\training\suite_v1\qwen3vl4b.yaml --out C:\Users\User\a-experiment\paddleocr\labels\suite_v1
```

HF datasets are reused from `suite-v1\_shared\hf_cache`; AraMS is the
existing `aocr-repo\data\arams_28k_htr`.

**WDAC / pyarrow:** `pip install -e` of this repo pulls `datasets`, which
unbounded-resolves to `pyarrow==25.x`. On this box, 25.x's `_dataset.pyd`
is blocked (`An Application Control policy has blocked this file`) and
`bench run` dies on the first `load_dataset`. The suite venv already runs
`pyarrow==21.0.0` successfully; the launcher re-pins that wheel after
install, and `pyproject.toml` caps `pyarrow<22`. The `check` phase now
imports `pyarrow.dataset` / `datasets.load_dataset` so this fails before
smoke, not mid-sweep.

**Training invocation (GPU, not yet run):** `scripts/finetune_paddleocr.py`
fills `configs/training/paddleocr/arabic_PP-OCRv5_mobile_rec.finetune.yaml`
and prints the `tools/train.py` command. It **refuses `--run`** if
`nvidia-smi` already shows a compute app (i.e. the VLM suite). Before the
first real train:

1. Clone `github.com/PaddlePaddle/PaddleOCR` (`--clone-repo`) so
   `tools/train.py` and `ppocrv5_arabic_dict.txt` exist. The pip package
   alone does not ship them.
2. Confirm the pretrained checkpoint PaddleX cached during the eval test:
   `C:\Users\User\.paddlex\official_models\arabic_PP-OCRv5_mobile_rec\`.
3. Run a tiny smoke first (`--labels-limit 20` on the experiment launcher,
   then `--run` on a handful of epochs) before a full mix.

```powershell
python scripts\finetune_paddleocr.py --labels C:\Users\User\a-experiment\paddleocr\labels\suite_v1 --out C:\Users\User\a-experiment\paddleocr\finetune --clone-repo
# later, GPU free:
python scripts\finetune_paddleocr.py --labels ... --out ... --venv C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe --run
```

The overlay raises `max_text_length` from the upstream 25 to 80 — Arabic
manuscript lines routinely exceed 25 characters. Expect the NRTR head to
reinit extra positions; CTC is the metric we actually score.

This is the one piece of the PaddleOCR integration that's genuinely
unverified on this box — everything else in this document was tested
directly, not assumed.
