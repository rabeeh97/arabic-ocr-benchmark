@echo off
rem GPU night 2, checklist 7.1: base VLM weights into the ONE shared HF cache (C:\Users\User\.cache\huggingface, the default;
rem HF_HOME is not set on the box). Rerun-safe: hf download skips files already in the cache. Qwen models are not gated.
rem Launch detached (Win32_Process Create, see docs/research/xai-tasks-and-runbook.md section 4); log: xai\vlm_n2\fetch.log.
cd /d C:\Users\User\a-experiment\xai
if not exist vlm_n2 mkdir vlm_n2
echo [%date% %time%] fetch start >> vlm_n2\fetch.log
.venv\Scripts\hf.exe download Qwen/Qwen3-VL-4B-Instruct >> vlm_n2\fetch.log 2>&1
echo [%date% %time%] 4B exit %errorlevel% >> vlm_n2\fetch.log
.venv\Scripts\hf.exe download Qwen/Qwen3-VL-8B-Instruct >> vlm_n2\fetch.log 2>&1
echo [%date% %time%] 8B exit %errorlevel% >> vlm_n2\fetch.log
echo [%date% %time%] FETCH DONE >> vlm_n2\fetch.log
