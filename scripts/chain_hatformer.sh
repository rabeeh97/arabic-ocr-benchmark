#!/bin/zsh
# HATFormer (AraMS-28k) on the official RASAM-1 / RASAM-2 / TariMa test lines, Mac CPU, then rebuild the local leaderboard (2026-10-26).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
M=/Volumes/PortableSSD/models/hatformer-arams28k/model.safetensors
LOG=/Users/rabeehjouni/colab_staging/logs/hatformer.log
until [ -f $M ] && [ $(stat -f %z $M) -gt 1300000000 ] && ! pgrep -f "hatformer-arams28k/resolve" > /dev/null; do sleep 30; done
echo "weights $(stat -f %z $M) bytes $(date)" >> $LOG
export PYTHONPATH=src
PY=$HOME/.venvs/kraken-rasam/bin/python
rm -f results/rasam/external/hatformer_r1__test.json   # drop the 16-line smoke-test file
$PY scripts/hatformer_rasam_eval.py --limit 4 --sets r1 >> $LOG 2>&1 && rm -f results/rasam/external/hatformer_r1__test.json
$PY scripts/hatformer_rasam_eval.py >> $LOG 2>&1
$PY scripts/build_rasam_leaderboard.py >> $LOG 2>&1
echo "HATFORMER DONE $(date)" >> $LOG
