@echo off
rem Queue on the 3090 (2026-10-12): wait for the ink-crop training, then train the shape-targeted and random-control CRNNs,
rem then score every model. Launch with Win32_Process Create (see docs/research/xai-tasks-and-runbook.md).
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe
:wait
if exist aligner_run_inkcrop\model_final.pt goto go
timeout /t 60 /nobreak >nul
goto wait
:go
echo [%date% %time%] ink-crop done, scoring it >> queue.log
%PY% geom_transfer_eval.py --data inkcrop_eval.pkl --run aligner_run_inkcrop --out inkcrop_trained.json --device cuda --reps 1000 >> queue.log 2>&1
echo [%date% %time%] training targeted >> queue.log
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_shape --repeat-file shape_select\targeted.json > shape_train.log 2>&1
echo [%date% %time%] training random control >> queue.log
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_randctl --repeat-file shape_select\random.json > randctl_train.log 2>&1
echo [%date% %time%] scoring >> queue.log
%PY% geom_transfer_eval.py --data eval_data.pkl --run aligner_run --out shape_select\base.json --device cuda --reps 200 >> queue.log 2>&1
%PY% geom_transfer_eval.py --data eval_data.pkl --run aligner_run_shape --out shape_select\targeted_eval.json --device cuda --reps 200 >> queue.log 2>&1
%PY% geom_transfer_eval.py --data eval_data.pkl --run aligner_run_randctl --out shape_select\random_eval.json --device cuda --reps 200 >> queue.log 2>&1
echo [%date% %time%] ALL DONE >> queue.log
