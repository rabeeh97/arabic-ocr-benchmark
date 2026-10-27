#!/bin/zsh
# Mac CPU: score the 5 leave-manuscript-out folds as their Colab jobs land (after the specialisation/architecture chain; one job at a time).
# Per fold: build the same hold-out pack locally, greedy + char LM on r2ho (all lines of the 3 held-out manuscripts; LM tuned on the remaining
# RASAM-2 validation lines), then the report against the in-split models on the same official test lines.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
L=/Users/rabeehjouni/colab_staging/logs
LOG=$L/mac_chain_lomo.log
echo "chain start $(date)" >> $LOG
# (the specialisation/architecture chain has finished)
for i in 4 5 1 2 3; do
  [ $i -le 3 ] && JL=lomo_a || JL=lomo_b
  until grep -q "=== finished" $L/$JL.log 2>/dev/null; do sleep 60; done
  R=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-${JL/_/-}-* | tail -1)/content/crnn_out/lomo$i
  if [ ! -f $R/model_final.pt ]; then echo "!!! fold $i: no model_final.pt in $R" >> $LOG; continue; fi
  HO=($($PY -c "import json; print(' '.join(json.load(open('results/rasam/lomo_folds.json'))['folds'][$i - 1]))"))
  P=$DD/rasam_lomo$i.pkl
  [ -f $P ] || $PY scripts/rasam_prepare_crnn_data.py $P --sets RASAM-1 RASAM-2 tarima --labels raw --height 96 \
    --repair results/rasam/repairs/rasam2_recut.pkl --holdout $HO >> $LOG 2>&1
  echo "== fold $i ($HO) $(date)" >> $LOG
  $PY scripts/rasam_crnn_eval.py --data $P --run $R --name LOMO${i}_best --ckpt model_best.pt --tags r2ho >> $LOG 2>&1
  $PY scripts/rasam_lm_decode.py --data $P --run $R --name LOMO${i}_lm --train-sets r1 r2 tm --val-tag r2 --test-tags r2ho --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  $PY scripts/rasam_lomo_report.py >> $LOG 2>&1
done
echo "LOMO CHAIN DONE $(date)" >> $LOG
