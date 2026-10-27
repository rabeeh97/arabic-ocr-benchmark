#!/bin/zsh
# Mac CPU chain (one job at a time), 2026-10-27: after HATFormer, score
#  1. the specialisation job: spec_r2 (RASAM-2 only), spec_tm (TariMa only), spec_all (control: same 10 extra epochs on all three sets),
#     each fine-tuned from the 60-epoch model; greedy + char LM on its own test set(s), plus repaired RASAM-2 where RASAM-2 is involved
#  2. the Calfa-architecture job (AMIDDA paper CRNN, 120 px, shared data): greedy + char LM on all three tests + repaired RASAM-2
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
L=/Users/rabeehjouni/colab_staging/logs
LOG=$L/mac_chain_spec_calfa.log
echo "chain start $(date)" >> $LOG
until grep -q "HATFORMER DONE" $L/hatformer.log 2>/dev/null || ! pgrep -f chain_hatformer.sh > /dev/null; do sleep 60; done

score() {  # $1 run dir, $2 name, $3 official pack, $4 repaired pack, $5.. tags
  local R=$1 N=$2 P=$3 RP=$4; shift 4
  $PY scripts/rasam_crnn_eval.py --data $DD/$P --run $R --name ${N}_best --ckpt model_best.pt --tags $@ >> $LOG 2>&1
  for t in $@; do
    $PY scripts/rasam_lm_decode.py --data $DD/$P --run $R --name ${N}_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
      --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  done
  if [[ " $@ " == *" r2 "* ]]; then
    $PY scripts/rasam_crnn_eval.py --data $DD/$RP --run $R --name ${N}_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
    $PY scripts/rasam_lm_decode.py --data $DD/$RP --run $R --name ${N}_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 --order 8 \
      --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  fi
}

# 1. specialisation
until grep -q "=== finished" $L/specialise.log 2>/dev/null; do sleep 60; done
J=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-specialise-* | tail -1)/content/crnn_out
echo "== specialisation $(date)" >> $LOG
[ -f $J/spec_r2/model_best.pt ] && score $J/spec_r2 SPEC_r2 rasam_tm_h96.pkl rasam_tm_h96_repaired.pkl r2
[ -f $J/spec_tm/model_best.pt ] && score $J/spec_tm SPEC_tm rasam_tm_h96.pkl rasam_tm_h96_repaired.pkl tm
[ -f $J/spec_all/model_best.pt ] && score $J/spec_all SPEC_all rasam_tm_h96.pkl rasam_tm_h96_repaired.pkl r1 r2 tm

# 2. architecture tests: "tall" CRNN (96 px, height / 8) and the Calfa CRNN retry (120 px, lr 3e-4; skipped if it aborted stuck on blanks)
until grep -q "=== finished" $L/tall_h96.log 2>/dev/null; do sleep 60; done
R=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-tall-h96-* | tail -1)/content/crnn_out/tall_h96_w05
echo "== tall CRNN $(date)" >> $LOG
[ -f $R/model_final.pt ] && score $R TALL_h96 rasam_tm_h96.pkl rasam_tm_h96_repaired.pkl r1 r2 tm
until grep -q "=== finished" $L/calfa_lr3e4.log 2>/dev/null; do sleep 60; done
R=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-calfa-lr3e4-* | tail -1)/content/crnn_out/calfa_h120_lr3e4
if [ -f $R/model_final.pt ]; then
  [ -f $DD/rasam_tm_h120.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_tm_h120.pkl --sets RASAM-1 RASAM-2 tarima --labels raw --height 120 >> $LOG 2>&1
  [ -f $DD/rasam_tm_h120_repaired.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_tm_h120_repaired.pkl --sets RASAM-1 RASAM-2 tarima --labels raw \
    --height 120 --repair results/rasam/repairs/rasam2_recut.pkl >> $LOG 2>&1
  echo "== calfa architecture $(date)" >> $LOG
  score $R CALFA_h120 rasam_tm_h120.pkl rasam_tm_h120_repaired.pkl r1 r2 tm
else
  echo "!!! calfa retry has no model_final.pt (aborted or failed): not scored" >> $LOG
fi
echo "CHAIN DONE $(date)" >> $LOG
