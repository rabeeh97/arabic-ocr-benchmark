@echo off
rem Overnight queue on the 3090 (planned 2026-10-12). Starts when queue_shape_runs.bat writes ALL DONE, then runs one job at a
rem time. Progress: C:\Users\User\a-experiment\xai\aligner\queue_night.log. Launch with Win32_Process Create.
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe
:wait
findstr /c:"ALL DONE" queue.log >nul 2>&1 && goto go
timeout /t 120 /nobreak >nul
goto wait
:go
echo [%date% %time%] N1 baseline replicate (run-to-run noise) >> queue_night.log
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_rep2 > rep2_train.log 2>&1
echo [%date% %time%] N2 combined recipe: ink-cropped data + shape-targeted repeats >> queue_night.log
%PY% -u xai_train_aligner.py --data aligner_data_inkcrop.pkl --out aligner_run_combo --repeat-file shape_select\targeted.json > combo_train.log 2>&1
echo [%date% %time%] N3 scoring >> queue_night.log
%PY% geom_transfer_eval.py --data eval_data.pkl --run aligner_run_rep2 --out night\rep2_eval.json --device cuda --reps 200 >> queue_night.log 2>&1
%PY% geom_transfer_eval.py --data inkcrop_eval.pkl --run aligner_run_rep2 --out night\rep2_inkcrop.json --device cuda --reps 500 >> queue_night.log 2>&1
%PY% geom_transfer_eval.py --data inkcrop_eval.pkl --run aligner_run_combo --out night\combo_inkcrop.json --device cuda --reps 500 >> queue_night.log 2>&1
%PY% geom_transfer_eval.py --data inkcrop_eval.pkl --run aligner_run_shape --out night\shape_inkcrop.json --device cuda --reps 500 >> queue_night.log 2>&1
%PY% geom_transfer_eval.py --data inkcrop_eval.pkl --run aligner_run_randctl --out night\randctl_inkcrop.json --device cuda --reps 500 >> queue_night.log 2>&1
%PY% geom_transfer_eval.py --data transfer_eval.pkl --run aligner_run_shape --out night\shape_transfer.json --device cuda --reps 500 >> queue_night.log 2>&1
echo [%date% %time%] N4 slant sweep on ink-cropped lines (do crop and slant gains stack?) >> queue_night.log
%PY% xai_geom_tta.py --data inkcrop_eval.pkl --run aligner_run --ckpt model_final.pt --out night\tta_inkcrop.json --n 600 >> queue_night.log 2>&1
echo [%date% %time%] N5 letter novelty against the targeted model's errors >> queue_night.log
%PY% xai_letter_shapes.py --train-data aligner_data.pkl --eval-data eval_data.pkl --run aligner_run_shape --ckpt model_final.pt --out night\letters_shape --n-train 2000 >> queue_night.log 2>&1
%PY% xai_letter_shapes.py --train-data aligner_data.pkl --eval-data eval_data.pkl --run aligner_run_randctl --ckpt model_final.pt --out night\letters_randctl --n-train 2000 >> queue_night.log 2>&1
echo [%date% %time%] N6 E8 with Qwen3-1.7B (does scale fix the LLM decoder?) >> queue_night.log
%PY% -u xai_e8_spike.py --train-data aligner_data.pkl --eval-data eval_data.pkl --run aligner_run --out e8_out_1p7b --llm Qwen/Qwen3-1.7B > e8_1p7b.log 2>&1
echo [%date% %time%] NIGHT ALL DONE >> queue_night.log
