#!/bin/zsh
# Usage: scripts/colab_bench_retry.sh LOGFILE ARGS...   e.g.  scripts/colab_bench_retry.sh results/rasam/colab_bench/t4.log --gpu T4 --epochs 3
# Re-runs colab_bench.py while it exits 75 (Colab API unreachable before any session was created). Any other exit code ends the loop.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
LOG=$1; shift
for i in {1..20}; do
  echo "=== attempt $i $(date +%H:%M:%S)" >> $LOG
  python3 scripts/colab_bench.py "$@" >> $LOG 2>&1
  rc=$?
  [ $rc -ne 75 ] && break
  sleep 120
done
echo "=== finished rc=$rc $(date +%H:%M:%S)" >> $LOG
