# Full KHATT v1.0 on the 3090 (checklist 1.6, for 4.1 and 4.6)

Built on the laptop by `.geomvenv/bin/python scripts/khatt_full_panel.py` from the Teklia/KHATT_v1.0 snapshot in
`/Volumes/PortableSSD/huggingface/hub` (plus the 200 KITAB-KHATT lines for the overlap). Counts, source paths and sha256 are in
`results/panel/khatt_full_manifest.json`. Per-line flags for CPU analysis (all 6,673 lines, no images) are in
`results/panel/khatt_full_flags.parquet`.

## What the box needs

| File (laptop `results/panel/`) | Size (gzip) | Fingerprint (`pickle_fingerprint.py`) | Used by |
|---|---|---|---|
| `khatt_full_panel.pkl`: 1,038 test lines × {as stored, ink-cropped} = 2,076 rows, tags `kitab_khatt_full` and `kitab_khatt_full__inkcrop` | 112 MB (19 MB) | `44bcfb4aec718ef0` | 4.1, 4.6 scoring |
| `khatt_full_train.pkl`: 4,672 train (`role` train) and 963 validation (`role` heldout) lines, as stored | 302 MB (52 MB) | `1f3f04d1b4df08b8` | 4.6 in-domain CRNN |

No new code: `crnn_eval_panel.py` and `xai_train_aligner.py` already read these formats (checked on the laptop with a
24-row subset and `aligner_run`). Upload gzipped (about 2 min at 0.6 MB/s), unpack, compare fingerprints.

```
# laptop
gzip -k results/panel/khatt_full_panel.pkl results/panel/khatt_full_train.pkl
scp results/panel/khatt_full_*.pkl.gz spring-win-server-3090:C:/Users/User/a-experiment/xai/aligner/
# box (in xai/aligner): gunzip, then python pickle_fingerprint.py khatt_full_panel.pkl  -> 44bcfb4aec718ef0
#                                         python pickle_fingerprint.py khatt_full_train.pkl  -> 1f3f04d1b4df08b8
# 4.6a  every existing CRNN on the full test (separate out dir, so the "panel" field of the panel.pkl results is untouched)
python crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run aligner_run_combo ... --out-dir khatt_full_out   # list every run folder: cmd.exe does not expand globs
# 4.6b  in-domain KHATT CRNN (2 seeds, per evaluation-standards §3), then score it on both panels
python xai_train_aligner.py --data khatt_full_train.pkl --out khatt_run_s0 --seed 0
python crnn_eval_panel.py --data khatt_full_panel.pkl --runs khatt_run_s0 --out-dir khatt_full_out
python crnn_eval_panel.py --data panel.pkl --runs khatt_run_s0 --out-dir panel_out
# 4.1  the CTC posterior dump takes --data khatt_full_panel.pkl next to panel.pkl
```

Bring back `khatt_full_out/*.json` to `results/panel/khatt_full_out/` and back up `khatt_run_*` to `aocr-box-backup/`.

## Read the numbers with these flags (in each row's `meta` and in the flags parquet)

| Test lines (1,038) | Count | Meaning |
|---|---|---|
| `fixed_text` | 1,035 | ≥ half the line's word 3-grams occur in another KHATT v1.0 line |
| `text_repeat` | 752 | exact (letters-only) text occurs more than once in KHATT v1.0 |
| `text_in_train` | 713 | the exact text occurs in the v1.0 train split |
| `kitab_text` | 194 | the exact text equals a KITAB-KHATT text |
| `kitab_image_id` set | 12 | the line *is* a KITAB-KHATT image. *Not* in-train for the aligner-recipe CRNNs (they train on Muharaf + RASAM only; `xai_prepare_aligner_data.SOURCES`); `store_build` marks a full-KHATT line in-train only when its KITAB twin is (0 today) |

The Teklia copy is almost entirely KHATT's fixed-text portion (vocabulary of 1,061 words over 6,673 lines; all lines with a
shared 3-gram fall in one connected component). The design's "score fixed-text and unique-text lines separately" therefore has
only 3 unique test lines; split instead by `text_in_train` (memorisation for the in-domain model); the 12
`kitab_image_id` test lines are simply the KITAB images (not in-train for the aligner CRNNs; corrected 2026-10-17). Only 89 of the 200 KITAB-KHATT texts occur in v1.0 (the design said 127;
the other 111 are unique-text lines absent from this copy), and all 89 are the identical images (64 train, 13 validation,
12 test).
