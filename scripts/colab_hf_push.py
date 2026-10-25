# Runs INSIDE the Colab VM (uploaded by colab_job.py). Pushes this job's results to the private Hugging Face repo, so they survive even if the Mac
# supervisor or its connection is gone. Usage: python /content/colab_hf_push.py REPO JOB [final]
# Pushes /content/crnn_out/** (weights incl. ckpt_last.pt for resuming, configs, metrics) and the job log to runs/<JOB>/ ; never the data packs.
# Writes the outcome to /content/run/hf_push_status (read by the supervisor's status poll). The token is read from /content/.hf_token (mode 600).
import sys
import time

from huggingface_hub import HfApi

repo, job = sys.argv[1], sys.argv[2]
final = len(sys.argv) > 3 and sys.argv[3] == "final"
tok = open("/content/.hf_token").read().strip()
t = time.time()
try:
    info = HfApi(token=tok).upload_folder(
        folder_path="/content", path_in_repo=f"runs/{job}", repo_id=repo, repo_type="model",
        allow_patterns=["crnn_out/**", "run/job.log", "run/job.sh", "run/exit_code", "run/hf_push.log"],
        ignore_patterns=["**/*.pkl", "**/*.parquet"],
        commit_message=f"{job}: {'final' if final else 'periodic'} push",
    )
    msg = f"HFPUSH ok {'final' if final else 'periodic'} {time.strftime('%H:%M:%S')} {time.time() - t:.0f}s {getattr(info, 'oid', '')[:10]}"
except Exception as e:  # noqa: BLE001  a failed push is retried at the next interval; the Mac copy is the other safety net
    msg = f"HFPUSH failed {'final' if final else 'periodic'} {time.strftime('%H:%M:%S')} {type(e).__name__}: {str(e)[:150]}"
open("/content/run/hf_push_status", "w").write(msg + "\n")
print(msg, flush=True)
