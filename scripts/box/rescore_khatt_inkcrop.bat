@echo off
rem Re-score the ink-cropped full KHATT v1.0 test (kitab_khatt_full__inkcrop) after the 2026-10-18 inkcrop() fix.
rem Before running: copy the rebuilt results\panel\khatt_full_panel.pkl (pack panel-khatt-full-2026-10-18) and
rem scripts\box\drop_tag.py into xai\aligner (the old pickle and JSONs are backed up in aocr-box-backup/xai-2026-10-18-night1).
rem Only the inkcrop tag is scored again (about 1 min per run, 22 runs). Progress: rescore_inkcrop.log.
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe
echo [%date% %time%] inkcrop re-score start >> rescore_inkcrop.log
%PY% -u drop_tag.py khatt_full_out kitab_khatt_full__inkcrop --pack panel-khatt-full-2026-10-18 >> rescore_inkcrop.log 2>&1
%PY% -u drop_tag.py khatt_full_out_skel kitab_khatt_full__inkcrop --pack panel-khatt-full-2026-10-18 >> rescore_inkcrop.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run aligner_run_base_s3 aligner_run_combo aligner_run_combo_s2 aligner_run_combo_s3 aligner_run_geomaug aligner_run_inkcrop aligner_run_randctl aligner_run_randctl_s2 aligner_run_randctl_s3 aligner_run_rep2 aligner_run_shape aligner_run_shape_s2 aligner_run_shape_s3 aligner_run_std_s0 aligner_run_std_s1 khatt_run_s0 aligner_run_nakba_s0 aligner_run_hard_s2 aligner_run_hard_s3 --out-dir khatt_full_out >> rescore_inkcrop.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run_skel_s0 --out-dir khatt_full_out_skel --label-map skeleton_map.json >> rescore_inkcrop.log 2>&1
echo [%date% %time%] inkcrop re-score ALL DONE >> rescore_inkcrop.log
