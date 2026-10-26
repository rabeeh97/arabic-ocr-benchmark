#!/bin/zsh
# RS4b (2026-10-26): 60 epochs + AraGPT2 rescoring. 10-best lists of the 60-epoch model (model_best, chosen on validation) on the Mac,
# LM scoring on a Colab CPU session, then rescoring with w, v tuned on validation (repaired RASAM-2 tuned on the official RASAM-2 validation list).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-long-ema-* | tail -1)/content/crnn_out/reptm_h96_w05_e60_ema
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
L=/Users/rabeehjouni/colab_staging/logs
LOG=$L/e60_rescore.log
echo "start $(date)" >> $LOG
for spec in "r1 rasam_tm_h96.pkl NB_E60_r1 300" "r2 rasam_tm_h96.pkl NB_E60_r2 300" "tm rasam_tm_h96.pkl NB_E60_tm 300" "r2 rasam_tm_h96_repaired.pkl NB_E60_r2rep 0"; do
  set -- ${=spec}
  $PY scripts/rasam_lm_decode.py --data $DD/$2 --run $RUN --ckpt model_best.pt --name $3 --train-sets r1 r2 tm --val-tag $1 --test-tags $1 --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 --nbest 10 --nbest-val $4 >> $LOG 2>&1
done
echo "NBEST DONE $(date)" >> $LOG
scripts/colab_job_retry.sh $L/lmscore_e60.log configs/colab/lmscore_e60_cpu.json
S=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-lmscore-e60-* | tail -1)/content/crnn_out/lmscore
$PY scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_E60_r1 --tag r1 >> $LOG 2>&1
$PY scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_E60_r2 --tag r2 >> $LOG 2>&1
$PY scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_E60_tm --tag tm >> $LOG 2>&1
$PY scripts/rasam_nbest_rescore.py --scores-dir $S --name NB_E60_r2rep --tag r2 --val-name NB_E60_r2 >> $LOG 2>&1
echo "E60 RESCORE DONE $(date)" >> $LOG
