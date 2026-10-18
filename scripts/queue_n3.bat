@echo off
rem GPU follow-up after night 2 (planned 2026-10-18): waits for "N2 ALL DONE" in vlm_n2\queue_n2.log, then runs on the full card, one job at a time:
rem   4.8  shape targeting for NAKBA, seed 3 (targeted and random; seed 2 ran in night 2) so the comparison has two seeds
rem   4.5  in-domain NAKBA CRNN, second seed
rem   4.1  posterior dumps for the standard-split CRNN seed 1 (left out of night 1 to stay in budget)
rem About 4.5 h. Every training call is resumable (checkpoint every 10 steps), so a relaunch continues where it stopped.
rem Live view: ssh spring-win-server-3090 "powershell Get-Content C:\Users\User\a-experiment\xai\aligner\queue_n3.log -Wait -Tail 30"
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe
echo [%date% %time%] N3 waiting for N2 ALL DONE >> queue_n3.log
:wait
findstr /c:"N2 ALL DONE" C:\Users\User\a-experiment\xai\vlm_n2\queue_n2.log >nul 2>&1 && goto go
timeout /t 120 /nobreak >nul
goto wait
:go
echo [%date% %time%] N3-1 (4.8) targeted, seed 3 >> queue_n3.log
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_nshape_s3 --repeat-file shape_select_nakba\targeted.json --seed 3 --ckpt-every 10 > nshape_s3_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_nshape_s3 --out-dir panel_out_n2 >> queue_n3.log 2>&1
echo [%date% %time%] N3-2 (4.8) random control, seed 3 >> queue_n3.log
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_nrand_s3 --repeat-file shape_select_nakba\random.json --seed 3 --ckpt-every 10 > nrand_s3_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_nrand_s3 --out-dir panel_out_n2 >> queue_n3.log 2>&1
echo [%date% %time%] N3-3 (4.5) in-domain NAKBA CRNN, seed 1 >> queue_n3.log
%PY% -u xai_train_aligner.py --data aligner_data_nakba.pkl --out aligner_run_nakba_s1 --seed 1 --ckpt-every 10 > nakba_s1_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_nakba_s1 --out-dir panel_out >> queue_n3.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run_nakba_s1 --out-dir khatt_full_out >> queue_n3.log 2>&1
echo [%date% %time%] N3-4 (4.1) posterior dumps, standard-split CRNN seed 1 >> queue_n3.log
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_std_s1 --out-dir panel_out --dump-posteriors posteriors >> queue_n3.log 2>&1
%PY% -u crnn_eval_panel.py --data khatt_full_panel.pkl --runs aligner_run_std_s1 --out-dir khatt_full_out --dump-posteriors posteriors >> queue_n3.log 2>&1
echo [%date% %time%] N3 ALL DONE >> queue_n3.log
