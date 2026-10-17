@echo off
rem GPU night 1, part B (planned 2026-10-17): checklist 4.1, 4.2, 4.5, 4.3, 4.4, 4.7, in priority order, one job at a time.
rem Waits for "N1A ALL DONE" in queue_n1a.log (part A, checklist 4.6). Progress: C:\Users\User\a-experiment\xai\aligner\queue_n1b.log.
rem Launch with Win32_Process Create (see docs/research/xai-tasks-and-runbook.md). Every step is rerun-safe: training resumes
rem from ckpt_last.pt (checkpoint every 10 steps), panel scoring skips finished tags, posterior dumps skip finished runs,
rem adaptation skips finished configurations. Estimated GPU time about 8 h.
rem Not queued: 4.8 (shape targeting for NAKBA, 4 runs of about 70 min plus selection; over the night's budget) -> night 2.
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe
:wait
findstr /c:"N1A ALL DONE" queue_n1a.log >nul 2>&1 && goto go
timeout /t 120 /nobreak >nul
goto wait
:go
echo [%date% %time%] N1B start >> queue_n1b.log

echo [%date% %time%] N1B-1 (4.1) CTC posterior dump: base, combo, standard-split s0 on the panel and full KHATT (about 45 min) >> queue_n1b.log
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run aligner_run_combo aligner_run_std_s0 --out-dir panel_out --dump-posteriors posteriors >> queue_n1b.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run aligner_run_combo aligner_run_std_s0 --out-dir khatt_full_out --dump-posteriors posteriors >> queue_n1b.log 2>&1

echo [%date% %time%] N1B-2 (4.2) skeleton-only CRNN: standard recipe, dotless labels, seed 0 (about 70 min) >> queue_n1b.log
if not exist aligner_data_standard_skel.pkl %PY% -u xai_skeleton_labels.py apply skeleton_map.json aligner_data_standard.pkl aligner_data_standard_skel.pkl >> queue_n1b.log 2>&1
%PY% -u xai_train_aligner.py --data aligner_data_standard_skel.pkl --out aligner_run_skel_s0 --seed 0 --ckpt-every 10 > skel_s0_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_skel_s0 --out-dir panel_out_skel --label-map skeleton_map.json --dump-posteriors posteriors >> queue_n1b.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run_skel_s0 --out-dir khatt_full_out_skel --label-map skeleton_map.json --dump-posteriors posteriors >> queue_n1b.log 2>&1

echo [%date% %time%] N1B-3 (4.5) in-domain NAKBA CRNN, seed 0 (about 50 min) >> queue_n1b.log
%PY% -u xai_train_aligner.py --data aligner_data_nakba.pkl --out aligner_run_nakba_s0 --seed 0 --ckpt-every 10 > nakba_s0_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_nakba_s0 --out-dir panel_out >> queue_n1b.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run_nakba_s0 --out-dir khatt_full_out >> queue_n1b.log 2>&1

echo [%date% %time%] N1B-4 (4.3) CRNN writer adaptation, RASAM-2 manuscripts and NAKBA (about 95 min) >> queue_n1b.log
%PY% -u xai_adapt_crnn.py --testbed rasam2 --run aligner_run_std_s0 --panel panel.pkl --bank-data aligner_data_standard.pkl --out adapt_out\rasam2.json --seed 0 --ckpt-every 10 > adapt_rasam2.log 2>&1
%PY% -u xai_adapt_crnn.py --testbed nakba --run aligner_run_std_s0 --panel panel.pkl --bank-data aligner_data_standard.pkl --pool-data aligner_data_nakba.pkl --out adapt_out\nakba.json --seed 0 --ckpt-every 10 > adapt_nakba.log 2>&1

echo [%date% %time%] N1B-5 (4.4) few-shot Maghrebi curve, RASAM-1, K = 16 / 100 / 300 (about 15 min) >> queue_n1b.log
%PY% -u xai_adapt_crnn.py --testbed rasam1 --run aligner_run_std_s0 --panel panel.pkl --bank-data aligner_data_standard.pkl --draws 2 --out adapt_out\rasam1.json --seed 0 --ckpt-every 10 > adapt_rasam1.log 2>&1

echo [%date% %time%] N1B-6 (4.7) hard-example control: selection, then seeds 2 and 3 (about 3.7 h) >> queue_n1b.log
if not exist shape_select\hard.json %PY% -u xai_hard_select.py --data aligner_data.pkl --run aligner_run --match shape_select\targeted.json --out shape_select\hard.json >> queue_n1b.log 2>&1
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_hard_s2 --repeat-file shape_select\hard.json --seed 2 --ckpt-every 10 > hard_s2_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_hard_s2 --out-dir panel_out >> queue_n1b.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run_hard_s2 --out-dir khatt_full_out >> queue_n1b.log 2>&1
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_hard_s3 --repeat-file shape_select\hard.json --seed 3 --ckpt-every 10 > hard_s3_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_hard_s3 --out-dir panel_out >> queue_n1b.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run_hard_s3 --out-dir khatt_full_out >> queue_n1b.log 2>&1

echo [%date% %time%] N1B ALL DONE >> queue_n1b.log
