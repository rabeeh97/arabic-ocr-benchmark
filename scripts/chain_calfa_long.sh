#!/bin/zsh
# Queued 2026-10-27 at the user's request: continue the Calfa-architecture CRNN (15 epochs at lr 3e-4) for as long as it still improves
# (plateau lr schedule, stop after 4 evaluations = 8 epochs without a new best, at most 60 more epochs) on a Colab L4, after both
# leave-manuscript-out jobs have finished; then score it on the Mac (greedy + char LM) after the leave-manuscript-out scoring.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
L=/Users/rabeehjouni/colab_staging/logs
until grep -q "=== finished" $L/lomo_a.log 2>/dev/null && grep -q "=== finished" $L/lomo_b.log 2>/dev/null; do sleep 60; done
caffeinate -i scripts/colab_job_retry.sh $L/calfa_long.log configs/colab/calfa_long_l4.json
until grep -q "LOMO CHAIN DONE" $L/mac_chain_lomo.log 2>/dev/null || ! pgrep -f mac_chain_lomo.sh > /dev/null; do sleep 60; done
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=$L/mac_chain_calfa_long.log
R=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-calfa-long-* | tail -1)/content/crnn_out/calfa_long
if [ -f $R/model_best.pt ]; then
  echo "== calfa long $(date)" >> $LOG
  $PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h120.pkl --run $R --name CALFA_long_best --ckpt model_best.pt --tags r1 r2 tm >> $LOG 2>&1
  $PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h120_repaired.pkl --run $R --name CALFA_long_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
  for t in r1 r2 tm; do
    $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h120.pkl --run $R --name CALFA_long_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
      --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  done
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h120_repaired.pkl --run $R --name CALFA_long_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 \
    --order 8 --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
else
  echo "!!! calfa long: no model_best.pt in $R" >> $LOG
fi
echo "CALFA LONG DONE $(date)" >> $LOG
