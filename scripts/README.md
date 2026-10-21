# scripts/ index

Command-line tools, grouped by component. Reusable logic lives in
`src/arabic_ocr_bench/`; scripts are thin drivers around it. Run from the repo root:
`PYTHONPATH=src .venv/bin/python scripts/<name>.py`.

Generated from each script's module docstring — regenerate after adding scripts
(`python scripts/_gen_index.py`). Sub-folders: `box/` (3090 machine queues/backups), `lexicon/` (word-list build).

## Run & orchestrate (benchmarks, 3090 box, suites, queues)

| script | what it does |
|---|---|
| `_detach_spawn.py` | Spawn a worker that keeps running after the SSH session that launched it ends. |
| `_envfile.py` | Load KEY=VALUE experiment.env files (stdlib, no python-dotenv). |
| `_experiment_bases.py` | Single source of truth for '--base <name>' specs. |
| `backfill_runner_id.py` | One-time migration: stamp a runner id onto results produced before the |
| `cost_report.py` | Cost report for a hosted-API benchmark run. |
| `eval_ain_nakba_ckpts.py` | Snapshot AIN NAKBA LoRA checkpoints and score them after the 3090 is free. |
| `finetune_paddleocr.py` | Fill in PaddleOCR recognition-finetune YAML paths. Does not train. |
| `fix_runbook_launch.py` | One-off: switch the runbook's detached launches from Start-Process to Win32_Process Create (survives SSH logout). |
| `handoff_ain_nakba_visft.py` | After contest-prompt Baseer finishes, start ViT-unfreeze training. |
| `list_paid_models.py` | Zero-cost discovery of models visible to the *paid* API keys. |
| `queue_n1a.bat` |  |
| `queue_n1b.bat` |  |
| `queue_n2.bat` |  |
| `queue_n2c.bat` |  |
| `queue_n3.bat` |  |
| `queue_night.bat` |  |
| `queue_shape_runs.bat` |  |
| `run_3090_benchmark.py` | Held-out benchmark for the RTX-3090 fine-tune (Windows-first, also POSIX). |
| `run_3090_experiment.py` | Self-contained RTX-3090 fine-tuning runner (Windows-first, also POSIX). |
| `run_3090_pipeline.py` | One detached command: train a base + benchmark it vs its untuned self. |
| `run_charctc_experiment.py` | Character-CTC CPU experiment (Windows-first, also POSIX). |
| `run_detectors.py` | Pattern detectors over the per-line store (plan 2026-10-13, §3.5). |
| `run_experiment_suite.py` | Sequential multi-model training + full-benchmark queue (suite_v1, Windows-first). |
| `run_gemini_full.py` | Drive configs/run.gemini-full.yaml to completion. |
| `run_hosted.py` | Run hosted-API models (Gemini / Mistral / OpenRouter / Claude / GPT) on a |
| `run_paddleocr_experiment.py` | Untuned PaddleOCR baseline + label prep (Windows-first, also POSIX). |
| `run_prompt_study.py` | Prompt-sensitivity study driver (see docs/experiments/prompt-study.md). |
| `slim_ledger.py` | Shrink the ledger HTML so it loads on phones and slow links. |
| `stage_ain_nakba_visft.py` | Staged ViT-unfreeze: train 400 → official score → continue only if CER improves. |
| `sweep_audit.py` | Audit a distributed sweep: coverage, per-runner tallies, duplicate work. |
| `token_cap_report.py` | X6: which models' stored readings stop at their output-token cap. |

## Experiments (single-question studies)

| script | what it does |
|---|---|
| `calib_sentence_cover.py` | Checklist 2.7: calibration-sentence candidates (design: docs/research/designs/writer-adaptation.md, part C). |
| `experiment_agreement_ablation.py` | Robustness of the cross-model-agreement confidence signal. |
| `experiment_confidence_baselines.py` | Compare reference-free confidence signals for OCR lines, within each model. |
| `experiment_consensus_transcription.py` | Consensus transcription by minimum-Bayes-risk (MBR) selection, from existing predictions. |
| `experiment_gliner2_verifier.py` | Text-only confidence gate on OCR output, using an Arabic GLiNER2 checkpoint. |
| `experiment_label_noise_detection.py` | Flag suspect ground-truth labels using the model consensus, from existing predictions. |
| `pair_diff.py` | Per-sample paired diff between two (model, dataset) score sets. |
| `rescore_muharaf_clean.py` | Re-score `muharaf_lines` (Muharaf test) without the lines CER-HV flags as |
| `seed_instability.py` | Seed and numeric instability: the lines that flip between identical trainings (checklist 1.4, evaluation-standards |
| `vlm_ink_extent.py` | Do loose KHATT crops hurt the VLMs too? Per-line CER of every benchmarked model against the ink extent of the native image. |
| `vlm_n2.py` | GPU night 2 VLM jobs (checklist 7.2-7.6) on the 3090, through the patched benchmark client (0.2b: token cap + logprobs). |
| `vlm_n2_export.py` | GPU night 2 (checklist 7.2-7.6): the line images the VLM jobs read, built on the laptop from the cached Arrow files. |

## Analysis (letters, dots, PAWs, text, errors)

| script | what it does |
|---|---|
| `_feasibility_paws.py` | Feasibility of a PAW library: do the large connected components of a line image match the label's PAWs one to one? |
| `_feasibility_spans.py` | Contact sheet of CTC-span letter crops (base CRNN alignment) to judge whether spans cut letters cleanly enough |
| `analyze_conventions.py` | Transcription conventions, writers' dialect spellings, and NAKBA's train/test text leak (themes T0, T4; 2026-10-15). |
| `analyze_dots.py` | Dots and skeletons across every system (plan addition 2026-10-15, theme T2). |
| `analyze_hamza_emission.py` | Is the alif-hamza a reading or a learned convention? (plan 2026-10-13, P1.4, T11 across families) |
| `analyze_ijam.py` | The iʿjām question (theme T2/T6, 2026-10-15) on our training texts: after folding orthographic variants (hamza/alef, ya/maqsura, ta marbuta), |
| `analyze_letters.py` | Analysis B, the letter, PAW and word level (plan 2026-10-13, §3.4), on the per-line store. |
| `analyze_lines.py` | Analysis A, the sample level (plan 2026-10-13, §3.3), on the per-line store. |
| `analyze_novelty.py` | The X-G10 confound battery: does letter-shape novelty predict letter errors beyond the line it sits in? (plan P1.2) |
| `analyze_paw_gaps.py` | Spaces inserted inside words: do they fall at PAW gaps more often than chance? (plan 2026-10-13, P1.5 / V5) |
| `analyze_phonetic_confusions.py` | Do systems confuse letters that sound alike, or letters that look alike? (makhārij question, 2026-10-13) |
| `analyze_shape_feasibility.py` | Feasibility of an allograph library (shape programme, 2026-10-15): are CTC letter spans usable as cut lines, and do |
| `analyze_text_tables.py` | Text-side tables (plan 2026-10-13, P1.6: R1.6 lexical diversity, R1.7 shape tail), from GT only. |
| `analyze_tta_gate.py` | Gate slant TTA on measured slant (plan 2026-10-13, P1.3), offline from stored readings. |
| `analyze_validity.py` | Which measured characteristics predict a line's CER, jointly and across systems? (plan 2026-10-13, P1.7 / R7, V2 gate) |
| `dec_common.py` | Shared helpers for the decoding side studies (checklist 2.14 tokenizer view, 2.15 root-sharing). |
| `dec_fetch_tokenizers.py` | Fetch only the tokenizer files of the benchmark VLMs into HF_HOME (checklist 2.14, E2). No weights. |
| `dec_root_sharing.py` | Checklist 2.15 / map perea-root: do real-word substitutions share the root more often than chance? CPU, decoding venv. |
| `dec_tokenizer_view.py` | Checklist 2.14 / map E2: the tokenizer view (decoding-and-restoration.md §6). CPU, decoding venv. |
| `decompose_errors.py` | Error decomposition over every canonical prediction (characterisation P2). |
| `dots_analyze.py` | Image-side studies on the span features (checklist 2.1 PAW gaps, 2.4 dot styles, 2.5 vertical stacking, 2.18 dot |
| `dots_common.py` | Shared loaders for the dots / skeleton / spacing studies (checklist 2.1, 2.3, 2.4, 2.5, 2.13, 2.18). |
| `dots_extract.py` | Image features per GT letter and per letter boundary, from CTC spans on the 64 px panel images (checklist 2.1, 2.4, 2.5, |
| `dots_gallery.py` | One HTML gallery for the dots / skeleton / spacing studies (2.1, 2.3, 2.4, 2.5): worst, best and random cases with the |
| `dots_text.py` | Text-side studies on every reading (checklist 2.3 skeleton classes and the tooth-count detector, 2.13 similarity-choice |
| `eval_stats.py` | Shared statistics for the evaluation contract (docs/research/designs/evaluation-standards.md, items 5 and 6). |
| `line_end_check.py` | Where does the end-of-line error excess come from? (task R2.8) |
| `profile_datasets.py` | Text-layer profile of every dataset we have ground truth for (characterisation P1). |
| `sample_characteristics.py` | Per-sample characteristics over complete model × sample blocks |
| `shape_analyze.py` | Shape-targeted oversampling: baseline, targeted and random-control CRNNs on KHATT half B (never used for the selection). |
| `shape_letter_feats.py` | Letter features for the novel-shape clusters (checklist 2.2, shape-programme step 1), CPU. |
| `shape_novel_clusters.py` | Novel-shape clusters, atlas v0 (checklist 2.2; shape-programme step 1; map novel-clusters). CPU. |
| `shape_paw_coverage.py` | PAW-inventory coverage curve (checklist 2.6, map P-L3; shape-programme "related checks"). Text only, CPU. |
| `shape_paw_report.py` | One-page report for the PAW-inventory coverage curve (checklist 2.6): curves, table, decision, and a per-line gallery. |
| `writer_khatt_fixed_text.py` | KHATT fixed-text natural experiment: same text, many writers (checklist 2.16, map `P1.8`; design writer-adaptation D). |
| `writer_khatt_pages.py` | Writer and page structure of the full KHATT v1.0 copy (checklist 2.17, map `R0.4`; feeds 2.16 / `P1.8`). |
| `writer_khatt_selftest.py` | Unit tests for the pure helpers of writer_khatt_pages.py and writer_khatt_fixed_text.py (checklist 2.16 / 2.17). |

## Geometry (ink extent, transfer, augmentation)

| script | what it does |
|---|---|
| `check_geom_aug.py` | Montage of real lines and four random draws of geom_aug each, to check the augmentation by eye. |
| `geom_analyze.py` | Summaries and tests on the per-line geometry (X-G1, X-G2, X-G4, X-G5, X-G6). |
| `geom_bench_step.py` | Time and peak memory of CRNN training steps on a device: python geom_bench_step.py DEVICE BATCH WIDTH_MULT LINE_WIDTH |
| `geom_data.py` | Shared loader for the geometry measurements (X-G1 to X-G6): reads cached Arrow files, no network. |
| `geom_dots.py` | X-G7 CPU proxy: how much dot contrast survives resizing a line to height 32 / 48 / 64, plain and dot-preserving. |
| `geom_extra.py` | Extra descriptive numbers from the geometry run: slant coverage against the training pool, rule contamination, pen contrast. |
| `geom_hamza.py` | Is NAKBA's alif-hamza error a spelling convention? Hamza use in training against evaluation labels, and CER with hamza folded. |
| `geom_ink_extent.py` | How much of each 64 px line image does the writing fill, and how does CER depend on it? |
| `geom_inkcrop_export.py` | Original and ink-cropped versions of the evaluation lines, for the size-normalization test. |
| `geom_measure.py` | Per-line geometry of Arabic text lines: stroke width, dots, baseline, slant, hinge, characteristic loci. |
| `geom_overlay.py` | Overlay check for geom_measure: dots (red), body skeleton (blue), fitted baseline (green), slant ticks (orange). |
| `geom_paw_coverage.py` | PAW inventory and coverage (queue item "PAW-inventory coverage curve"), plus unseen-PAW rates against the CRNN's training text. |
| `geom_prepare_replica_data.py` | Training data for the laptop replica of the CRNN-CTC aligner (same format as xai_prepare_aligner_data.py). |
| `geom_rules_check.py` | Montage of NAKBA lines before and after ruled-line removal (from the export pickle), for a visual check. |
| `geom_text_shift.py` | Text-side distance between the CRNN's training text and each evaluation set (R1.6 / R1.7 in the CRNN setting). |
| `geom_tiny.py` | A tiny CRNN-CTC that trains on the laptop CPU, for the geometry-transfer and augmentation tests. |
| `geom_tiny_analyze.py` | Analyse the tiny-CRNN runs: transfer test (X-T2), augmentation effect, per-line geometry against CER. |
| `geom_transfer.py` | Turn lines of one dataset into the look of another, using the geometry we measured (CPU). |
| `geom_transfer_eval.py` | CER of a CRNN-CTC aligner on original and transformed lines, with paired differences (X-T2). |
| `geom_transfer_export.py` | Pack original and transformed lines for the 3090, so CER can tell whether the transformed geometry carries a transfer gap. |
| `geom_transfer_show.py` | Print the per-feature result of a geom_transfer run. |
| `image_layer.py` | Image layer, first pass (task R6.0): does the image explain the line |
| `khatt_full_panel.py` | Full KHATT v1.0 (Teklia copy) for the evaluation panel and for an in-domain KHATT CRNN (checklist 1.6, map `khatt-full`). |

## Explainability (xai_*)

| script | what it does |
|---|---|
| `build_xai_explorer.py` | Build the attribution explorer page from xai_attribute_qwen3vl.py output. |
| `xai_adapt_crnn.py` | CRNN few-shot writer adaptation (checklist 4.3, `writer-adapt-crnn`) and the few-shot Maghrebi curve (checklist 4.4, `E6`). |
| `xai_align.py` | Forced alignment of known transcripts to the trained CRNN-CTC: character spans for the panel lines (X-B1). |
| `xai_attribute_qwen3vl.py` | Per-token pixel attribution for Qwen3-VL-4B (+ our LoRA adapter) on OCR lines. |
| `xai_derive_standard_split.py` | Derive the standard-split training pickle from an existing aligner_data.pkl, without the HF cache (for the 3090). |
| `xai_e8_spike.py` | E8 spike: the frozen CRNN-CTC encoder read by a small LLM through a projector. Runs on the 3090. |
| `xai_eval_aligner.py` | CER of the trained CRNN-CTC aligner on datasets it never saw (generalisation check). |
| `xai_export_error_lines.py` | Export an error-focused panel (worst lines + easy controls) for the attribution spike. |
| `xai_export_lines.py` | Export a small, letter-targeted panel of line images for the attribution spike. |
| `xai_geom_tta.py` | X-G3 / X-G8 on the 3090: shear sweep and confidence-selected test-time augmentation for the CRNN-CTC aligner. |
| `xai_hard_select.py` | Hard-example control for shape targeting (checklist 4.7, shape-programme "Related checks": `hard-example-control`). |
| `xai_inkcrop_traindata.py` | Ink-cropped copy of the aligner training pickle (X-S1 follow-up), for training on cropped lines. Runs on the 3090. |
| `xai_letter_shapes.py` | Letter-shape library and shape novelty from CTC letter spans (X-G10). Runs on the 3090. |
| `xai_pointing_game.py` | Pointing game for the attribution maps against CTC character spans (X-B2). |
| `xai_prepare_aligner_data.py` | Pack line images + labels into one pickle for the CTC aligner (X-B1). |
| `xai_prepare_eval_data.py` | Pack held-out lines from datasets the aligner never saw (NAKBA, KHATT) for a generalisation check. |
| `xai_prepare_panel.py` | The standard evaluation panel for every CRNN run (plan 2026-10-13, §3.9): every line of every test set, as stored and |
| `xai_probe_locateanything.py` | Zero-shot probe: can LocateAnything-3B find Arabic handwriting at word / PAW / letter / dot level? |
| `xai_shape_select.py` | Shape-targeted oversampling (X-G11 lite): find training lines whose letters look like KHATT's unusual letters. Runs on the 3090. |
| `xai_skeleton_labels.py` | Dotless (skeleton) labels for the skeleton-only CRNN (checklist 4.2, decoding-and-restoration §4). |
| `xai_train_aligner.py` | Train a CRNN-CTC line recogniser to serve as a forced aligner (X-B1). Runs on the 3090. |

## Audits & datasets

| script | what it does |
|---|---|
| `audit_common.py` | Shared loaders for the data-and-label audit scripts (checklist 2.8-2.12; docs/research/designs/data-and-label-audit.md). |
| `audit_datasheets.py` | Checklist 2.11b / R1.4: one datasheet card per panel dataset, built from the audit outputs. |
| `audit_diacritics.py` | Checklist 2.8 / R2.7 (the diacritic scoring rule) and 2.11a / R1.3 (diacritic-policy share per set). |
| `audit_difficulty.py` | Checklist 2.12: IRT 2PL line difficulty and discrimination (R3.2), challenge subsets as dataset filters (R3.3), and |
| `audit_leakage.py` | Checklist 2.9 / R4.2: text and image leakage and duplicates, within and across the panel sets and their training pools. |
| `audit_spelling.py` | Checklist 2.10: spelling inconsistency inside a label set (R4.3) and the per-dataset conventions table (R4.4). |
| `crnn_eval_panel.py` | Score CRNN runs on the standard panel (xai_prepare_panel.py) in the store format (plan 2026-10-13, §3.2 / §3.9). |
| `detectors_extra.py` | Rules behind detectors D4, D21 and D23 of run_detectors.py (data-and-label-audit §9, checklist 1.5). |
| `export_run_images.py` | Dump every line image a run touched as `images.json` in the run dir: |
| `extract_ctc_example.py` | Run the real CTC aligner (std-split CRNN) on one real line and save what the timeline needs to draw how letter spans are made: |
| `extract_detector_examples.py` | Run the real geometry detectors (scripts/geom_measure.py) on a few real lines and draw each detector's own output |
| `extract_popup_examples.py` | Compute the real examples behind the timeline's term pop-ups -> docs/research/timeline_popups.json. |
| `extract_timeline_assets.py` | Collect the evidence images for the project timeline into docs/research/timeline_assets.json. |
| `fetch_arams.py` | Download + extract AraMS-28k-HTR (Zenodo) for `src/arabic_ocr_bench/datasets/arams.py`. |
| `fetch_fonts.py` | Download extra Arabic display fonts for the synthetic printed-line renderer. |
| `fetch_training_corpus.py` | Pre-fetch every real training-data source once, shared across a whole suite run. |
| `prep_paddleocr_labels.py` | Convert one of our training configs' `data.sources` into PaddleOCR / |
| `regen_mini_findings.py` | Regenerate the golden findings fixture. Run twice; confirm no diff. |
| `store_build.py` | The per-line store: every reading of every line by every system, scored at every rung (plan 2026-10-13, §3.2). |
| `store_features.py` | Image measurements for every panel line, plus a Muharaf-train reference (plan 2026-10-13, §3.6 item 2). |

## Reports, galleries & site builders

| script | what it does |
|---|---|
| `_leaderboard_findings.py` | Hand-rolled SVG / CSS chart primitives for the leaderboard's Findings view. |
| `_leaderboard_prompt_study.py` | The leaderboard's "Research" tab: the prompt-sensitivity study's findings, |
| `build_atlas.py` | Build the Arabic Script Atlas page (docs/research/atlas/index.html). |
| `build_brief.py` | Build the one-page visual brief of the research (docs/research/brief.html). |
| `build_checklist_section.py` | Render docs/research/MASTER-CHECKLIST.md into the ledger as the "Master checklist" section (after the research map). |
| `build_day2_section.py` | Insert the 2026-10-12 results (ink crop, letter novelty, augmented CRNN, E8, galleries) into the ledger and update the |
| `build_day3_section.py` | Insert the second 2026-10-12 results block (ink crop trained, VLMs and loose crops, hamza convention, shape-targeted data). |
| `build_day4_section.py` | Insert the 2026-10-13 results block (overnight 3090 queue: replicate noise, shape-targeted data on held-out lines, combined recipe, |
| `build_detector_gallery.py` | Per-sample gallery for detectors D4, D21, D23 and D24 (checklist 1.4 / 1.5): for each, the flagged lines the reference |
| `build_evidence_section.py` | Ledger block "Per-line and per-letter evidence" from the analysis bundle (plan 2026-10-13, §3.7). |
| `build_geometry_section.py` | Insert the "Geometry measurements" article into docs/research/paper-ledger.html and update the rows it touches. |
| `build_hamza_gallery.py` | Gallery: NAKBA lines scored as written against scored with hamza folded (is the hamza in the image or only in the label?). |
| `build_letter_gallery.py` | Gallery for the letter-shape library (X-G10): novelty bands against errors, allograph clusters, novel letters. |
| `build_model_gallery.py` | Gallery comparing two models' readings of the same lines (per-line JSON from geom_transfer_eval.py or xai_e8_spike.py). |
| `build_notebooks.py` | Generate the Colab notebooks under notebooks/ from plain cell lists. |
| `build_pair_gallery.py` | Gallery for one before/after image comparison read by one model (per-line JSON from geom_transfer_eval.py). |
| `build_research_map.py` | Render docs/research/research_map.json into the ledger's top "Research map" section and docs/research/RESEARCH-MAP.md. |
| `build_synthesis_section.py` | Build the ledger's opening synthesis section and inject it before the research map. |
| `build_timeline.py` | Build the project timeline (docs/research/timeline.html): step by step, why we did each thing, what came out, |
| `build_tiny_section.py` | Insert the "Laptop experiments" article (tiny CRNN, transfer test, augmentation, text distance, PAW coverage) into the ledger. |
| `build_transfer_html.py` | Build a self-contained HTML page of the dataset-to-dataset transformations. |
| `build_transfer_section.py` | Insert the "Geometry transfer" article into docs/research/paper-ledger.html (after the geometry measurements). |
| `build_xt2_gallery.py` | Gallery for the transfer test (X-T2) on the 3090 CRNN: which lines slant and stroke changes helped or hurt. |
| `change_gallery.py` | Galleries of the lines a change helped most, hurt most, and a few at random (images, ground truth, both readings). |
| `check_master_checklist.py` | Prove that MASTER-CHECKLIST.md lists every unfinished item of the research map (and every ID such an item groups). |
| `make_leaderboard.py` | Turn one or more results/<run_id>/ directories into a single self-contained |
| `pickle_fingerprint.py` | Fingerprint of a line pickle (roles, labels, image bytes) to check that two machines hold the same data. |
| `research_map_coverage.py` | Check that the research map places every idea, task and experiment from every source (nothing silently dropped). |

## Other

| script | what it does |
|---|---|
| `_gen_index.py` | Generate scripts/README.md: scripts grouped by component, one docstring line each. |
