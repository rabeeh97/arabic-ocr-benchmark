#!/bin/zsh
# N-best lists (top 10, char-LM beam search) of the best model (96 px, RASAM-1+2+TariMa) for rescoring with a pretrained Arabic LM.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-data-h96-* | tail -1)/content/crnn_out/a3tm_h96_w05
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/nbest.log
for spec in "r1 rasam_tm_h96.pkl NB_TM_r1" "r2 rasam_tm_h96.pkl NB_TM_r2" "r2 rasam_tm_h96_repaired.pkl NB_TM_r2rep"; do
  set -- ${=spec}
  $PY scripts/rasam_lm_decode.py --data $DD/$2 --run $RUN --name $3 --train-sets r1 r2 tm --val-tag $1 --test-tags $1 --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 --nbest 10 --nbest-val 300 >> $LOG 2>&1
done
echo "NBEST DONE $(date)" >> $LOG
