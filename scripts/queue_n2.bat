@echo off
rem GPU night 2 (planned 2026-10-18): checklist 7.1, 7.4, 7.3, 7.6, 7.5, 4.8, 7.2 in priority order. Live view:
rem   ssh spring-win-server-3090 "powershell Get-Content C:\Users\User\a-experiment\xai\vlm_n2\queue_n2.log -Wait -Tail 30"
rem Lives at scripts/queue_n2.bat (repo) and xai\vlm_n2\queue_n2.bat (box). Launch with Win32_Process Create (runbook section 4). Needs, in xai\vlm_n2: vlm_n2.py,
rem aocr_pkg.zip (the patched arabic_ocr_bench, 0.2b), vlm_n2_lines.pkl and icl_trials.json (scripts/vlm_n2_export.py), and the
rem base VLMs in the shared HF cache (scripts/box/n2_fetch_models.bat). The diff-LR adapter is xai\adapter (sha256 d20417e0...).
rem
rem Two streams. VLM stream (this file), model-major so each model loads once per item:
rem   4B diff-LR adapter: 7.4 reliance, 7.3 geometry, 7.6 resolution, 7.5 held-out bench; 4B base: 7.4; then 7.2 on the 4B adapter
rem   (stops 2.5 h before the deadline); then, once the CRNN stream is done, 8B base: 7.4, 7.3, 7.6 (100 lines), 7.2 (NAKBA only).
rem CRNN stream (queue_n2c.bat, started below, runs beside the 4B jobs only): 4.8 shape targeting for NAKBA, seed 2,
rem targeted and random. The 8B (about 19 GB) waits for it, because both would not fit in 24 GB.
rem Every VLM job appends one line per sample and resumes; n2_deadline.txt (start + 8 h, written once) stops jobs past it, so
rem the queue ends on time and drops from the end. Delete n2_deadline.txt before a relaunch on another day.
cd /d C:\Users\User\a-experiment\xai\vlm_n2
set PY=..\.venv\Scripts\python.exe
set HF_HUB_OFFLINE=1
set V=%PY% -u vlm_n2.py --pkg pkg --lines vlm_n2_lines.pkl --deadline-file n2_deadline.txt
echo [%date% %time%] N2 start >> queue_n2.log
if not exist n2_deadline.txt %PY% -c "import time; print(int(time.time() + 8 * 3600))" > n2_deadline.txt

echo [%date% %time%] N2-1 (7.1) setup: patched client, adapter, both base VLMs load and read 2 lines >> queue_n2.log
if not exist pkg\arabic_ocr_bench powershell -NoProfile -Command "Expand-Archive -Force aocr_pkg.zip pkg" >> queue_n2.log 2>&1
%PY% -c "import sys; sys.path.insert(0, 'pkg'); from arabic_ocr_bench.models.hf_local_client import HFLocalClient, token_logprobs, finish_reason; print('patched client ok')" >> queue_n2.log 2>&1
%V% gen --model qwen3vl4b-difflr --select kitab_khatt:stored:2 --out out\gen\qwen3vl4b-difflr.jsonl >> queue_n2.log 2>&1
%V% gen --model qwen3vl8b --select kitab_khatt:stored:2 --out out\gen\qwen3vl8b.jsonl >> queue_n2.log 2>&1

echo [%date% %time%] N2-6 (4.8) CRNN stream started beside the 4B jobs: log ..\aligner\queue_n2c.log >> queue_n2.log
start "" /b cmd /c C:\Users\User\a-experiment\xai\aligner\queue_n2c.bat

echo [%date% %time%] N2-2 (7.4) image reliance, 4B diff-LR: GT log-prob with image / blank page / no image, 800 lines (about 15 min) >> queue_n2.log
%V% reliance --model qwen3vl4b-difflr --select kitab_khatt:stored nakba_lines:stored:200 rasam2:stored:200 muharaf_lines:stored:200 --out out\reliance\qwen3vl4b-difflr.jsonl >> queue_n2.log 2>&1
echo [%date% %time%] N2-3 (7.3) geometry, 4B diff-LR: KHATT stored, ink-cropped, sheared +10.4 deg, 600 readings (about 40 min) >> queue_n2.log
%V% gen --model qwen3vl4b-difflr --select kitab_khatt:stored,inkcrop,shear10 --out out\gen\qwen3vl4b-difflr.jsonl >> queue_n2.log 2>&1
echo [%date% %time%] N2-4 (7.6) resolution, 4B diff-LR: KHATT at 60 px, plus noise SD 4 / 8 / 16, 800 readings (about 50 min) >> queue_n2.log
%V% gen --model qwen3vl4b-difflr --select kitab_khatt:h60,h60_n4,h60_n8,h60_n16 --out out\gen\qwen3vl4b-difflr.jsonl >> queue_n2.log 2>&1
echo [%date% %time%] N2-5 (7.5) E1.1 diff-LR bench: RASAM-2 372, NAKBA test 400, RASAM-1 400 (KHATT 200 done in 7.3; about 70 min) >> queue_n2.log
%V% gen --model qwen3vl4b-difflr --select rasam2:stored nakba_lines:stored rasam1:stored --out out\gen\qwen3vl4b-difflr.jsonl >> queue_n2.log 2>&1
echo [%date% %time%] N2-2b (7.4) image reliance, 4B base, same 800 lines (about 15 min) >> queue_n2.log
%V% reliance --model qwen3vl4b --select kitab_khatt:stored nakba_lines:stored:200 rasam2:stored:200 muharaf_lines:stored:200 --out out\reliance\qwen3vl4b.jsonl >> queue_n2.log 2>&1

echo [%date% %time%] N2-7 (7.2) in-context writer calibration, 4B diff-LR: NAKBA blind baseline, then 1,500 trials (stops 2.5 h before the deadline) >> queue_n2.log
%V% gen --model qwen3vl4b-difflr --select nakba_blind:stored --out out\gen\qwen3vl4b-difflr.jsonl --deadline-offset-h -2.5 >> queue_n2.log 2>&1
%V% icl --model qwen3vl4b-difflr --trials icl_trials.json --out out\icl\qwen3vl4b-difflr.jsonl --deadline-offset-h -2.5 >> queue_n2.log 2>&1

echo [%date% %time%] waiting for the CRNN stream before loading the 8B >> queue_n2.log
:wait8
findstr /c:"N2C DONE" C:\Users\User\a-experiment\xai\aligner\queue_n2c.log >nul 2>&1 && goto go8
timeout /t 120 /nobreak >nul
goto wait8
:go8
echo [%date% %time%] CRNN stream done >> queue_n2.log
echo [%date% %time%] N2-2c (7.4) image reliance, 8B base, 800 lines (about 20 min) >> queue_n2.log
%V% reliance --model qwen3vl8b --select kitab_khatt:stored nakba_lines:stored:200 rasam2:stored:200 muharaf_lines:stored:200 --out out\reliance\qwen3vl8b.jsonl >> queue_n2.log 2>&1
echo [%date% %time%] N2-3b (7.3) geometry, 8B base, 600 readings (about 60 min) >> queue_n2.log
%V% gen --model qwen3vl8b --select kitab_khatt:stored,inkcrop,shear10 --out out\gen\qwen3vl8b.jsonl >> queue_n2.log 2>&1
echo [%date% %time%] N2-4b (7.6) resolution, 8B base, KHATT lines 0-99 at 60 px and three noise levels, 400 readings (about 55 min) >> queue_n2.log
%V% gen --model qwen3vl8b --select kitab_khatt:h60,h60_n4,h60_n8,h60_n16:100 --out out\gen\qwen3vl8b.jsonl >> queue_n2.log 2>&1
echo [%date% %time%] N2-7b (7.2) in-context, 8B base, NAKBA unit only (runs only if time is left) >> queue_n2.log
%V% gen --model qwen3vl8b --select nakba_blind:stored --out out\gen\qwen3vl8b.jsonl >> queue_n2.log 2>&1
%V% icl --model qwen3vl8b --trials icl_trials.json --units nakba --out out\icl\qwen3vl8b.jsonl >> queue_n2.log 2>&1

powershell -NoProfile -Command "Get-Content C:\Users\User\a-experiment\xai\aligner\queue_n2c.log -Tail 6" >> queue_n2.log 2>&1
echo [%date% %time%] N2 ALL DONE >> queue_n2.log
