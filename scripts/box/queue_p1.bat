@echo off
rem Plan 2026-10-13: P0.5 TTA all variants, P0.4 letter dumps, P2.1 seeds batch 2. Starts after queue_p0 writes "P0 ALL DONE". Log: queue_p1.log
cd /d C:\Users\User\a-experiment\xai\aligner
set PY=..\.venv\Scripts\python.exe -u
:wait
findstr /c:"P0 ALL DONE" queue_p0.log >NUL 2>&1 || (timeout /t 300 /nobreak >NUL & goto wait)
echo [%date% %time%] T1 slant TTA, all five shears stored, base and combo, KHATT and NAKBA (panel) >> queue_p1.log
%PY% xai_geom_tta.py --data panel.pkl --run aligner_run --ckpt model_final.pt --out panel_out\tta_base_allvar.json --n 3000 --all-variants --datasets kitab_khatt kitab_khatt__inkcrop nakba_lines nakba_lines__inkcrop nakba_blind >> queue_p1.log 2>&1
%PY% xai_geom_tta.py --data panel.pkl --run aligner_run_combo --ckpt model_final.pt --out panel_out\tta_combo_allvar.json --n 3000 --all-variants --datasets kitab_khatt kitab_khatt__inkcrop nakba_lines nakba_lines__inkcrop nakba_blind >> queue_p1.log 2>&1
echo [%date% %time%] T2 letter dumps (base, shape, random control, combo; standard split) >> queue_p1.log
%PY% -c "import pickle; b=pickle.load(open('panel.pkl','rb')); keep={'kitab_khatt','kitab_khatt__inkcrop','nakba_lines','muharaf_lines','rasam1','rasam2'}; pickle.dump({'H':64,'split':b['split'],'rows':[r for r in b['rows'] if r['dataset'] in keep]}, open('panel_letters.pkl','wb'), protocol=4)" >> queue_p1.log 2>&1
for %%R in (aligner_run aligner_run_shape aligner_run_randctl aligner_run_combo) do %PY% xai_letter_shapes.py --train-data aligner_data.pkl --eval-data panel_letters.pkl --run %%R --out letters_panel\%%R --dump-letters >> queue_p1.log 2>&1
%PY% xai_letter_shapes.py --train-data aligner_data_standard.pkl --eval-data panel_letters.pkl --run aligner_run_std_s0 --out letters_panel\aligner_run_std_s0 --dump-letters >> queue_p1.log 2>&1
echo [%date% %time%] T3 seeds batch 2 >> queue_p1.log
%PY% xai_train_aligner.py --data aligner_data.pkl --out aligner_run_shape_s3 --repeat-file shape_select\targeted.json --seed 3 > shape_s3_train.log 2>&1
%PY% xai_train_aligner.py --data aligner_data.pkl --out aligner_run_randctl_s3 --repeat-file shape_select\random.json --seed 3 > randctl_s3_train.log 2>&1
%PY% xai_train_aligner.py --data aligner_data_inkcrop.pkl --out aligner_run_combo_s2 --repeat-file shape_select\targeted.json --seed 2 > combo_s2_train.log 2>&1
%PY% xai_train_aligner.py --data aligner_data_inkcrop.pkl --out aligner_run_combo_s3 --repeat-file shape_select\targeted.json --seed 3 > combo_s3_train.log 2>&1
echo [%date% %time%] T4 panel evaluation of batch 2 >> queue_p1.log
%PY% crnn_eval_panel.py --data panel.pkl --runs aligner_run_shape_s3 aligner_run_randctl_s3 aligner_run_combo_s2 aligner_run_combo_s3 --out-dir panel_out >> queue_p1.log 2>&1
echo [%date% %time%] P1 ALL DONE >> queue_p1.log
