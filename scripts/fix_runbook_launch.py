"""One-off: switch the runbook's detached launches from Start-Process to Win32_Process Create (survives SSH logout)."""
import re
from pathlib import Path

p = Path("docs/research/xai-tasks-and-runbook.md")
s = p.read_text(encoding="utf-8")
s = re.sub(r"powershell -NoProfile -Command \"Start-Process -NoNewWindow python -ArgumentList '([^']*)' -RedirectStandardOutput (\S+?)\"",
           lambda m: ("powershell -NoProfile -Command \"Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
                      f"@{{CommandLine='cmd /c cd /d <DIR> && ..\\.venv\\Scripts\\python.exe -u {m.group(1)} > {m.group(2)} 2>&1'}}\""), s)
note = ("\n\n> **Launching detached over SSH:** `Start-Process` does not survive the end of the SSH session on this box (Windows OpenSSH ends the "
        "session's processes). Use `Invoke-CimMethod -ClassName Win32_Process -MethodName Create`, which starts the process outside the session. "
        "Venv: `C:\\Users\\User\\a-experiment\\xai\\.venv`; `<DIR>` = `C:\\Users\\User\\a-experiment\\xai\\aligner`; `<RUN>` = `aligner_run`.")
if "Launching detached over SSH" not in s:
    s = s.replace("## 2. Runbook for the 3090", "## 2. Runbook for the 3090" + note, 1)
p.write_text(s, encoding="utf-8")
print(s.count("Invoke-CimMethod"), "launch lines use Win32_Process Create")
