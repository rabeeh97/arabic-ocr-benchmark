#!/bin/zsh
# AMIDDA 1.0 Line 4B on the official test lines (Colab L4, approved 2026-10-26 with the floor lowered to 170), then copy the predictions into
# results/rasam/external/amidda_<set>__test.json and rebuild the local leaderboard. The supervisor stops the VM itself when the job ends.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
L=/Users/rabeehjouni/colab_staging/logs
caffeinate -i scripts/colab_job_retry.sh $L/amidda_r2tm.log configs/colab/amidda_r2tm_l4.json
J=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-amidda-r2tm-* | tail -1)/content/crnn_out/amidda
mkdir -p results/rasam/external
for s in r2 tm; do
  [ -f $J/preds_amidda_$s.json ] && cp $J/preds_amidda_$s.json results/rasam/external/amidda_${s}__test.json
done
PYTHONPATH=src $HOME/.venvs/kraken-rasam/bin/python scripts/build_rasam_leaderboard.py >> $L/amidda_r2tm.log 2>&1
echo "AMIDDA CHAIN DONE $(date)" >> $L/amidda_r2tm.log
