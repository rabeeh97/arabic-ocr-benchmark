#!/bin/zsh
# Score seeds 1 and 2 of the repaired + TariMa raw-label recipe (96 px, half width) on the Mac, the same way as seed 0 in colab_post_reptm.sh:
# official RASAM-1 / RASAM-2 / TariMa tests + repaired RASAM-2, greedy and with the character LM (tuned on validation only).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
J=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-seeds-t4-* | tail -1)/content/crnn_out
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_seeds.log
echo "start $(date)" >> $LOG
for s in 1 2; do
  RUN=$J/reptm_h96_w05_s$s
  N=RAW_reptm_s$s
  $PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h96.pkl --run $RUN --name ${N}_best --ckpt model_best.pt --tags r1 r2 tm >> $LOG 2>&1
  $PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h96_repaired.pkl --run $RUN --name ${N}_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
  for t in r1 r2 tm; do
    $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96.pkl --run $RUN --name ${N}_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
      --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  done
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96_repaired.pkl --run $RUN --name ${N}_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
done
echo "POST DONE $(date)" >> $LOG
