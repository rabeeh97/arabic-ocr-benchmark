# Checklist 7.7: back up GPU night 2 from the 3090 to the SSD. Laptop-side command list; run by hand after "N2 ALL DONE" is in
# xai\vlm_n2\queue_n2.log, with the VPN up. Nothing here deletes anything on the box; the trim list is at the end.
set -e
B=/Volumes/PortableSSD/aocr-box-backup/xai-2026-10-18-night2
X=spring-win-server-3090:C:/Users/User/a-experiment/xai
mkdir -p "$B/vlm_n2" "$B/aligner"
# VLM readings, reliance and in-context outputs (JSONL, a few MB) and the logs
scp -r "$X/vlm_n2/out" "$B/vlm_n2/"
scp "$X/vlm_n2/queue_n2.log" "$X/vlm_n2/n2_deadline.txt" "$X/vlm_n2/fetch.log" "$B/vlm_n2/"
# 4.8 CRNN: selection, weights (no ckpt_last.pt, the 114 MB resume state), panel scores, logs
for run in aligner_run_nshape_s2 aligner_run_nrand_s2; do
  mkdir -p "$B/aligner/$run"
  scp "$X/aligner/$run/charset.json" "$X/aligner/$run/config.json" "$X/aligner/$run/heldout_cer.json" "$X/aligner/$run/model_final.pt" "$X/aligner/$run/model_best.pt" "$B/aligner/$run/"
done
scp -r "$X/aligner/shape_select_nakba" "$X/aligner/panel_out_n2" "$B/aligner/"
scp "$X/aligner/queue_n2c.log" "$X/aligner/shape_select_nakba.log" "$X/aligner/nshape_s2_train.log" "$X/aligner/nrand_s2_train.log" "$B/aligner/"
du -sh "$B"
# Into the worktree for the laptop analysis:
#   mkdir -p results/vlm_n2 && cp -r "$B"/vlm_n2/out results/vlm_n2/ && cp "$B"/aligner/panel_out_n2/*.json results/panel/out/
#   cp -r "$B"/aligner/shape_select_nakba results/n2/   (halves.json holds the NAKBA blind half-B ids for evaluation)
# Parity check for 7.5: the diff-LR readings of kitab_khatt / nakba_lines / rasam1 / rasam2 lines 0-19 should match the earlier
# smoke readings in suite-v1-artifacts/qwen3vl4b_visft_difflr/bench/smoke/<dataset>/*/predictions.jsonl.
# Trimming the box (after the backup is checked; everything is regenerable):
#   ssh spring-win-server-3090 "del C:\Users\User\a-experiment\xai\vlm_n2\vlm_n2_lines.pkl"            (205 MB; scripts/vlm_n2_export.py)
#   ckpt_last.pt in aligner_run_nshape_s2 and aligner_run_nrand_s2                                     (114 MB each)
#   Keep Qwen3-VL-8B-Instruct (16.3 GB) in the shared HF cache while 7.2 (8B) and the seed-3 runs of 4.8 are still to run;
#   afterwards: ssh spring-win-server-3090 "C:\Users\User\a-experiment\xai\.venv\Scripts\hf.exe cache rm Qwen/Qwen3-VL-8B-Instruct"
