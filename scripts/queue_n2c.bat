@echo off
rem GPU night 2, CRNN stream (started by queue_n2.bat beside the 4B VLM jobs): checklist 4.8, shape targeting for NAKBA.
rem Selection on NAKBA blind half A (even ids of panel.pkl's nakba_blind rows; half B = odd ids, for evaluation) with the
rem Muharaf CRNN aligner_run; training on aligner_data.pkl with 3 extra copies of the selected lines (targeted) or of as many
rem random lines (control), as X-G11. Seed 2 tonight (seed 3 next GPU night: same commands with --seed 3 / _s3).
rem AOCR_CUDA_MEM_FRACTION caps the trainer at 30 % of the card (true peak about 4.2 GB) so the 4B VLM fits beside it.
rem Rerun-safe: selection is skipped when its output exists; training resumes from ckpt_last.pt (checkpoint every 10 steps).
rem Do not start a second copy while one runs (two trainers would write the same run folder).
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe
set AOCR_CUDA_MEM_FRACTION=0.30
echo [%date% %time%] N2C start >> queue_n2c.log
if not exist shape_select_nakba\targeted.json %PY% -u xai_shape_select.py --train-data aligner_data.pkl --eval-data panel.pkl --run aligner_run --out shape_select_nakba --dataset nakba_blind >> queue_n2c.log 2>&1
echo [%date% %time%] N2C targeted seed 2 (about 2 h beside the VLM) >> queue_n2c.log
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_nshape_s2 --repeat-file shape_select_nakba\targeted.json --seed 2 --ckpt-every 10 > nshape_s2_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_nshape_s2 --out-dir panel_out_n2 >> queue_n2c.log 2>&1
echo [%date% %time%] N2C random control seed 2 (about 2 h beside the VLM) >> queue_n2c.log
%PY% -u xai_train_aligner.py --data aligner_data.pkl --out aligner_run_nrand_s2 --repeat-file shape_select_nakba\random.json --seed 2 --ckpt-every 10 > nrand_s2_train.log 2>&1
%PY% -u crnn_eval_panel.py --data panel.pkl --runs aligner_run_nrand_s2 --out-dir panel_out_n2 >> queue_n2c.log 2>&1
echo [%date% %time%] N2C DONE >> queue_n2c.log
