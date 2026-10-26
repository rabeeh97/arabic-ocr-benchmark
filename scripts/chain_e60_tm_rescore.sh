#!/bin/zsh
# RS4b, TariMa part (2026-10-26): the first LM-scoring session hits its 120-minute VM limit before the TariMa lists, so they are scored on a
# second CPU session (configs/colab/lmscore_e60tm_cpu.json); rescore TariMa from that session's scores once it has finished.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
L=/Users/rabeehjouni/colab_staging/logs
until grep -q "=== finished" $L/lmscore_e60tm.log 2>/dev/null; do sleep 60; done
S=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-lmscore-e60tm-* | tail -1)/content/crnn_out/lmscore
PYTHONPATH=src $HOME/.venvs/kraken-rasam/bin/python scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_E60_tm --tag tm >> $L/e60_rescore.log 2>&1
echo "E60 TM RESCORE DONE $(date)" >> $L/e60_rescore.log
