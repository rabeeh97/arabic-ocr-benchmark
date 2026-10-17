@echo off
rem GPU night 1, part A (planned 2026-10-17): checklist 4.6, full KHATT v1.0.
rem Needs khatt_full_panel.pkl / khatt_full_train.pkl in xai\aligner (docs/research/KHATT-FULL-3090.md).
rem Progress: C:\Users\User\a-experiment\xai\aligner\queue_n1a.log. Launch with Win32_Process Create (see xai-tasks-and-runbook.md).
rem Part B (queue_n1b.bat) waits for "N1A ALL DONE" in that log.
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe
echo [%date% %time%] N1A-1 existing CRNNs on full KHATT test (stored + ink-cropped) >> queue_n1a.log
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run aligner_run_base_s3 aligner_run_combo aligner_run_combo_s2 aligner_run_combo_s3 aligner_run_geomaug aligner_run_inkcrop aligner_run_randctl aligner_run_randctl_s2 aligner_run_randctl_s3 aligner_run_rep2 aligner_run_shape aligner_run_shape_s2 aligner_run_shape_s3 aligner_run_std_s0 aligner_run_std_s1 --out-dir khatt_full_out >> queue_n1a.log 2>&1
echo [%date% %time%] N1A-2 in-domain KHATT CRNN, seed 0 (resumable; ckpt every 10 steps) >> queue_n1a.log
%PY% -u xai_train_aligner.py --data khatt_full_train.pkl --out khatt_run_s0 --seed 0 --ckpt-every 10 > khatt_s0_train.log 2>&1
echo [%date% %time%] N1A-3 in-domain KHATT CRNN on full KHATT and on the panel >> queue_n1a.log
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs khatt_run_s0 --out-dir khatt_full_out >> queue_n1a.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs khatt_run_s0 --out-dir panel_out >> queue_n1a.log 2>&1
echo [%date% %time%] N1A ALL DONE >> queue_n1a.log
