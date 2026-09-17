# PaddleOCR — eval + fine-tuning integration

PaddleOCR is a fundamentally different stack from the rest of this repo:
**PaddlePaddle, not PyTorch**, and a dedicated detection+recognition
pipeline (a 5M-parameter recognition model here), not a VLM. Nothing here
shares code with `sft_vlm.py`/TRL, and — deliberately — nothing here shares
a **venv** with the training suite either (see §1).

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

(This exact venv already exists on the box as of 2026-09-17 at
`C:\Users\User\a-experiment\paddleocr_venv` — built and verified during this
integration, moved there rather than rebuilt to avoid re-downloading the
~600 MB of CUDA wheels. Re-run the two `pip install` lines only if it's ever
deleted.)

Console output containing Arabic text needs `$env:PYTHONIOENCODING='utf-8'`
set first, or `print()` crashes with the same `UnicodeEncodeError` this
repo's other launcher scripts already guard against for file writes — this
is the one place that guard doesn't reach, since it's the *console's*
default codepage, not a file open call.

## 2. Evaluating it through this repo's own benchmark

`arabic-paddleocr-v5` is registered in `configs/models.yaml` (`client:
paddleocr`). Score it exactly like any other model:

```powershell
C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe -m pip install -e C:\Users\User\a-experiment\aocr-repo
C:\Users\User\a-experiment\paddleocr_venv\Scripts\python.exe -m arabic_ocr_bench.cli run --config <a run.yaml listing arabic-paddleocr-v5> --report
```

Run it from `paddleocr_venv`, not the suite's shared venv — `bench run`
only needs whichever clients the run config actually lists to be
importable, so a paddleocr-only venv with this repo installed editable
works fine for scoring just this one model. **Don't mix it into the same
`bench run` invocation as the torch-based models** (`qwen3-vl-*`, `ain-7b`,
etc.) — that would need both frameworks importable in one process, which is
exactly the untested-conflict scenario §1 avoids.

## 3. Fine-tuning — data ready, training invocation needs on-box verification

**Ready and tested**: `scripts/prep_paddleocr_labels.py` converts any of our
training configs' `data.sources` (the same Muharaf/AraMS/RASAM mix a Qwen
fine-tune trains on) into PaddleOCR's plain-text recognition label format
(`image_path<TAB>label` per line):

```powershell
C:\Users\User\a-experiment\aocr-repo\venv\Scripts\python.exe scripts\prep_paddleocr_labels.py --config configs\training\suite_v1\qwen3vl4b.yaml --out data\paddleocr_labels\suite_v1
```

(Run this with the **training suite's** venv, not `paddleocr_venv` — it only
needs this repo's dataset adapters, no paddle import at all. Verified
end-to-end against the offline `synthetic` dataset during this integration;
against the suite's real sources it will download the same Muharaf/AraMS/
RASAM data the Qwen fine-tunes already use, no separate corpus needed.)

**Needs verification, not yet run**: PaddleOCR's own recognition
fine-tuning entrypoint. Public documentation for the training workflow
(`tools/train.py -c <config>.yml`, a `pretrained_model:` field pointing at
`arabic_PP-OCRv5_mobile_rec`'s checkpoint, `Train.dataset.label_file_list`
pointing at the `train_list.txt` above) describes what has historically been
PaddleOCR's training CLI — but this integration only verified the
**inference** API (`paddleocr==3.7.0`'s `TextRecognition` class), and the
3.x line restructured that API surface significantly from older PaddleOCR
versions. Before trusting the training invocation:

1. Clone `github.com/PaddlePaddle/PaddleOCR` into
   `paddleocr_venv`'s working tree (the pip package alone may not ship
   `tools/train.py` + the recognition YAML configs — this needs checking).
2. Locate the actual `arabic_PP-OCRv5_mobile_rec` pretrained checkpoint
   PaddleX cached during the eval test:
   `C:\Users\User\.paddlex\official_models\arabic_PP-OCRv5_mobile_rec\` —
   confirm its files match what a `pretrained_model:` YAML field expects.
3. Run a tiny smoke fine-tune (a handful of steps, `--out data/paddleocr_labels/smoke`
   from step 1 above with `--split train` on a `limit: 20`-capped config)
   before committing to a full run, the same way this repo's own Qwen
   configs get a `--limit` smoke pass first.

This is the one piece of the PaddleOCR integration that's genuinely
unverified — everything else in this document was tested directly against
this box, not assumed.
