#!/bin/zsh
# Mac CPU chain (one job at a time, 8.6 GB RAM), 2026-10-26:
#  1. AraGPT2 N-best rescoring of the TariMa model's lists (scores from the lmscore Colab job)
#  2. when the 60-epoch EMA job has finished: average its last 10 epochs, score the five weight variants greedy (official tests + repaired RASAM-2),
#     and with the character LM for model_best (60 epochs), model_ema_best and model_avg10
#  3. when the standardised-label seeds job has finished: score seeds 1 and 2 like seed 0 (colab_post_reptm.sh)
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
L=/Users/rabeehjouni/colab_staging/logs
LOG=$L/mac_chain.log
echo "chain start $(date)" >> $LOG

# 1. pretrained-LM rescoring
S=/Users/rabeehjouni/colab_staging/jobs/job-lmscore-cpu-10251931/content/crnn_out/lmscore
echo "== step 1 nbest rescore $(date)" >> $LOG
$PY scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_TM_r1 --tag r1 >> $LOG 2>&1
$PY scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_TM_r2 --tag r2 >> $LOG 2>&1
$PY scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_TM_r2rep --tag r2 --val-name NB_TM_r2 >> $LOG 2>&1

lm() {  # $1 run, $2 ckpt, $3 name, $4 official pack, $5 repaired pack
  for t in r1 r2 tm; do
    $PY scripts/rasam_lm_decode.py --data $DD/$4 --run $1 --ckpt $2 --name ${3}_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
      --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  done
  $PY scripts/rasam_lm_decode.py --data $DD/$5 --run $1 --ckpt $2 --name ${3}_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
}
greedy() {  # $1 run, $2 ckpt, $3 name, $4 official pack, $5 repaired pack
  $PY scripts/rasam_crnn_eval.py --data $DD/$4 --run $1 --name ${3}_best --ckpt $2 --tags r1 r2 tm >> $LOG 2>&1
  $PY scripts/rasam_crnn_eval.py --data $DD/$5 --run $1 --name ${3}_repairedtest --ckpt $2 --tags r2 >> $LOG 2>&1
}

# 2. 60 epochs + EMA
echo "== step 2 waiting for the 60-epoch EMA job $(date)" >> $LOG
until grep -q "=== finished" $L/reptm_long_ema.log 2>/dev/null; do sleep 60; done
R=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-long-ema-* 2>/dev/null | tail -1)/content/crnn_out/reptm_h96_w05_e60_ema
if [ -f $R/model_final.pt ]; then
  echo "== step 2 scoring $R $(date)" >> $LOG
  $PY scripts/ckpt_average.py $R --glob 'model_ep*.pt' --out model_avg10.pt >> $LOG 2>&1
  for v in "model_best.pt E60" "model_final.pt E60final" "model_ema_best.pt E60ema" "model_ema_final.pt E60emafinal" "model_avg10.pt E60avg10"; do
    set -- ${=v}
    [ -f $R/$1 ] && greedy $R $1 $2 rasam_tm_h96.pkl rasam_tm_h96_repaired.pkl
  done
  for v in "model_best.pt E60" "model_ema_best.pt E60ema" "model_avg10.pt E60avg10"; do
    set -- ${=v}
    [ -f $R/$1 ] && lm $R $1 $2 rasam_tm_h96.pkl rasam_tm_h96_repaired.pkl
  done
else
  echo "!!! step 2 skipped: no model_final.pt in $R" >> $LOG
fi

# 3. standardised-label seeds 1 and 2
echo "== step 3 waiting for the standardised-label seeds job $(date)" >> $LOG
until grep -q "=== finished" $L/reptm_std_seeds.log 2>/dev/null; do sleep 60; done
J=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-std-seeds-* 2>/dev/null | tail -1)/content/crnn_out
for s in 1 2; do
  R=$J/reptm_std_h96_w05_s$s
  if [ -f $R/model_best.pt ]; then
    echo "== step 3 scoring seed $s $(date)" >> $LOG
    greedy $R model_best.pt STD_reptm_s$s rasam_tm_h96_std.pkl rasam_tm_h96_std_repaired.pkl
    lm $R model_best.pt STD_reptm_s$s rasam_tm_h96_std.pkl rasam_tm_h96_std_repaired.pkl
  else
    echo "!!! step 3 seed $s skipped: no model_best.pt in $R" >> $LOG
  fi
done
echo "CHAIN DONE $(date)" >> $LOG
