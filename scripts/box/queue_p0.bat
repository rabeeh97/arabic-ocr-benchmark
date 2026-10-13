@echo off
rem Plan 2026-10-13: P0.3 harmonised panel evaluation, P2.2 standard-split CRNN, P2.1 seeds batch 1. Log: queue_p0.log
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe -u
echo [%date% %time%] Q1 panel evaluation of the 7 existing CRNNs (fp32, one line at a time) >> queue_p0.log
%PY% crnn_eval_panel.py --data panel.pkl --runs aligner_run aligner_run_rep2 aligner_run_randctl aligner_run_shape aligner_run_combo aligner_run_geomaug aligner_run_inkcrop --out-dir panel_out >> queue_p0.log 2>&1
echo [%date% %time%] Q2 standard-split CRNN seed 0 >> queue_p0.log
%PY% xai_train_aligner.py --data aligner_data_standard.pkl --out aligner_run_std_s0 --seed 0 > std_s0_train.log 2>&1
echo [%date% %time%] Q3 standard-split CRNN seed 1 >> queue_p0.log
%PY% xai_train_aligner.py --data aligner_data_standard.pkl --out aligner_run_std_s1 --seed 1 > std_s1_train.log 2>&1
echo [%date% %time%] Q4 shape-targeted seed 2 >> queue_p0.log
%PY% xai_train_aligner.py --data aligner_data.pkl --out aligner_run_shape_s2 --repeat-file shape_select\targeted.json --seed 2 > shape_s2_train.log 2>&1
echo [%date% %time%] Q5 random control seed 2 >> queue_p0.log
%PY% xai_train_aligner.py --data aligner_data.pkl --out aligner_run_randctl_s2 --repeat-file shape_select\random.json --seed 2 > randctl_s2_train.log 2>&1
echo [%date% %time%] Q6 base recipe seed 3 >> queue_p0.log
%PY% xai_train_aligner.py --data aligner_data.pkl --out aligner_run_base_s3 --seed 3 > base_s3_train.log 2>&1
echo [%date% %time%] Q7 panel evaluation of the new runs >> queue_p0.log
%PY% crnn_eval_panel.py --data panel.pkl --runs aligner_run_std_s0 aligner_run_std_s1 aligner_run_shape_s2 aligner_run_randctl_s2 aligner_run_base_s3 --out-dir panel_out >> queue_p0.log 2>&1
%PY% crnn_eval_panel.py --data panel.pkl --runs aligner_run_std_s0 aligner_run_std_s1 --ckpt model_best.pt --out-dir panel_out_best >> queue_p0.log 2>&1
echo [%date% %time%] P0 ALL DONE >> queue_p0.log
