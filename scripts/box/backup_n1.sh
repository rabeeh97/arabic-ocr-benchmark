# Checklist 4.9: back up GPU night 1 (parts A and B) from the 3090 to the SSD. Laptop-side command list; run by hand after
# "N1B ALL DONE" is in queue_n1b.log, with the VPN up. Weights only (model_final.pt, model_best.pt and the small json files);
# ckpt_last.pt (resume state, 114 MB each) is not copied. Nothing here deletes anything on the box.
set -e
B=/Volumes/PortableSSD/aocr-box-backup/xai-2026-10-18-night1
R=spring-win-server-3090:C:/Users/User/a-experiment/xai/aligner
mkdir -p "$B"
# new run folders: part A (khatt_run_s0) and part B
for run in khatt_run_s0 aligner_run_skel_s0 aligner_run_nakba_s0 aligner_run_hard_s2 aligner_run_hard_s3; do
  mkdir -p "$B/$run"
  scp "$R/$run/charset.json" "$R/$run/config.json" "$R/$run/heldout_cer.json" "$R/$run/model_final.pt" "$R/$run/model_best.pt" "$B/$run/"
done
# outputs: panel and full-KHATT scores, skeleton scores, posteriors (a few large .npz), adaptation results, hard-example selection
scp -r "$R/panel_out" "$R/panel_out_skel" "$R/khatt_full_out" "$R/khatt_full_out_skel" "$R/posteriors" "$R/adapt_out" "$B/"
mkdir -p "$B/shape_select"
scp "$R/shape_select/hard.json" "$R/shape_select/hard_losses.json" "$B/shape_select/"
# logs
scp "$R/queue_n1a.log" "$R/queue_n1b.log" "$R/khatt_s0_train.log" "$R/skel_s0_train.log" "$R/nakba_s0_train.log" "$R/hard_s2_train.log" "$R/hard_s3_train.log" \
    "$R/adapt_rasam2.log" "$R/adapt_nakba.log" "$R/adapt_rasam1.log" "$B/"
du -sh "$B"
# Then copy what the laptop analysis needs into the worktree, e.g.
#   cp "$B"/panel_out/*.json results/panel/out/ ; cp -r "$B"/adapt_out results/n1b/ ; posteriors stay in $B (large).
# Trimming the box (only after the backup is checked; all regenerable from the laptop or the box):
#   ssh spring-win-server-3090 "del C:\Users\User\a-experiment\xai\aligner\aligner_data_standard_skel.pkl"
#   ckpt_last.pt in finished run folders (khatt_run_s0, aligner_run_skel_s0, aligner_run_nakba_s0, aligner_run_hard_s2/_s3)
