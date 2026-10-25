import subprocess
print(subprocess.run('tar czf /content/run_sync.tgz -C / --exclude=*.pkl --exclude=hfcache --exclude=*.parquet content/run content/crnn_out 2>/dev/null; ls -l /content/run_sync.tgz', shell=True, capture_output=True, text=True).stdout)
