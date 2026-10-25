#!/bin/zsh
# Usage: scripts/colab_job_retry.sh LOGFILE colab_job.py-args...   e.g.  scripts/colab_job_retry.sh results/rasam/colab_runs/t4.log configs/colab/crnn_bench.json
# Re-runs colab_job.py only while it exits 75 (Colab API unreachable BEFORE any session was created). Any other exit ends the loop.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
LOG=$1; shift
mkdir -p "$(dirname "$LOG")"
for i in {1..20}; do
  echo "=== attempt $i $(date +%H:%M:%S)" >> $LOG
  python3 -u scripts/colab_job.py "$@" >> $LOG 2>&1
  rc=$?
  [ $rc -ne 75 ] && break
  sleep 120
done
echo "=== finished rc=$rc $(date +%H:%M:%S)" >> $LOG
