import hashlib, json, os
roots = ['/content/run', '/content/crnn_out']
names = ('model_best.pt', 'model_final.pt', 'config.json', 'charset.json', 'heldout_cer.json', 'job.log', 'job.sh', 'exit_code')
all_files = False
out = []
for r in roots:
    for d, _, fs in os.walk(r):
        for n in fs:
            if n in names or n.startswith(('preds_', 'model_ema_', 'model_ep')) or n.endswith(('_metrics.json', '_overlap.json', '_timing.json')) or (all_files and n == 'ckpt_last.pt'):
                p = os.path.join(d, n)
                h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
                out.append([p, os.path.getsize(p), h])
print('MANIFEST=' + json.dumps(out))
