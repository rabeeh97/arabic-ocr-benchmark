#!/bin/zsh
# Score the long Calfa-architecture run on the Mac (the scoring half of chain_calfa_long.sh; rerun on 2026-10-28 because the SSD disconnected
# overnight and the chain's working directory vanished). Greedy + char LM on RASAM-1 / RASAM-2 / TariMa and repaired RASAM-2, 120 px packs.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/score_calfa_long.log
R=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-calfa-long-* | tail -1)/content/crnn_out/calfa_long
echo "== calfa long $(date) $R" >> $LOG
$PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h120.pkl --run $R --name CALFA_long_best --ckpt model_best.pt --tags r1 r2 tm >> $LOG 2>&1
$PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h120_repaired.pkl --run $R --name CALFA_long_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
for t in r1 r2 tm; do
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h120.pkl --run $R --name CALFA_long_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
done
$PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h120_repaired.pkl --run $R --name CALFA_long_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 \
  --order 8 --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
echo "SCORE DONE $(date)" >> $LOG
