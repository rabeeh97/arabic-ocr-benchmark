"""Spawn a worker that keeps running after the SSH session that launched it ends.

Windows OpenSSH puts the login in a Job with KILL_ON_JOB_CLOSE. ``DETACHED_PROCESS
| CREATE_NEW_PROCESS_GROUP`` is not enough: the child stays in that job and dies
the moment the SSH session closes. That is why ``--detach`` printed a pid, ran
while the session was open, then vanished mid-train with no traceback.

``Win32_Process.Create`` goes through WMI (a service outside the SSH job), so
the worker survives logout. Unix still uses ``start_new_session``.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def win_quote(arg: str) -> str:
    """Quote one token for cmd.exe."""
    return '"' + arg.replace('"', '""') + '"'


def _exe_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _with_unbuffered(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    if _exe_name(argv[0]) in {"python", "python.exe", "python3", "python3.exe"} and "-u" not in argv[:3]:
        return [argv[0], "-u", *argv[1:]]
    return argv


def stdout_is_regular_file() -> bool:
    """True when stdout is a plain file — the detached worker's ``>> log``."""
    try:
        return stat.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode)
    except Exception:  # noqa: BLE001
        return False


def emit_log_line(log_path: Path, line: str) -> None:
    """Write ``line`` to the log without opening a second writer on Windows.

    ``cmd.exe >> log 2>&1`` holds the file in a sharing mode that rejects
    another ``open('a')`` (PermissionError 13). Print always; re-open the
    file only when stdout is *not* already that log (foreground / piped).
    """
    print(line, flush=True)
    if stdout_is_regular_file():
        return
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_detached_cmd(
    argv: list[str], *, cwd: Path, log_path: Path, cmd_path: Path
) -> None:
    """Write a .cmd that cds, runs argv, and appends stdout/stderr to log_path."""
    argv = _with_unbuffered(argv)
    body = "\r\n".join(
        [
            "@echo off",
            f"cd /d {win_quote(str(cwd))}",
            " ".join(win_quote(a) for a in argv)
            + f" >> {win_quote(str(log_path))} 2>&1",
            "",
        ]
    )
    cmd_path.write_text(body, encoding="utf-8")


def _wmi_create_process(command_line: str, cwd: Path) -> int:
    """Create a process via WMI so it is not in the SSH Job Object.

    Command line and cwd travel through env vars so PowerShell quoting cannot
    mangle paths with spaces.
    """
    ps = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        "-Arguments @{ CommandLine = $env:AOCR_CMDLINE; CurrentDirectory = $env:AOCR_CWD }; "
        "if ($null -eq $r -or $r.ReturnValue -ne 0) { "
        "Write-Error ('Win32_Process.Create failed: ' + $(if ($r) { $r.ReturnValue } else { 'null' })); "
        "exit 1 }; "
        "$r.ProcessId"
    )
    env = os.environ.copy()
    env["AOCR_CMDLINE"] = command_line
    env["AOCR_CWD"] = str(cwd)
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if out.returncode != 0:
        raise RuntimeError(
            "WMI spawn failed (the process would die on SSH logout if we "
            f"fell back to Popen): {(out.stderr or out.stdout).strip()}"
        )
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("WMI spawn returned no ProcessId")
    return int(lines[-1])


def pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, check=False,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def pid_is_cmd(pid: int) -> bool:
    """True when *pid* is a cmd.exe wrapper (WMI ``--detach`` parent)."""
    if os.name != "nt" or not pid:
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
        capture_output=True, text=True, check=False,
    ).stdout.lower()
    return "cmd.exe" in out


def pidfile_is_other_worker(
    other: int, self_pid: int, ppid: int, *, alive: bool, is_cmd: bool,
) -> bool:
    """True when the pidfile names a live worker that is not us.

    WMI ``--detach`` writes the cmd.exe wrapper pid. The real python worker is
    often a grandchild (venv trampoline), so ``other != ppid`` and a naive
    alive-check aborts the worker as 'already running'.
    """
    return bool(other) and other not in (self_pid, ppid) and alive and not is_cmd


def spawn_surviving_logout(
    argv: list[str], *, cwd: Path, log_path: Path, pid_path: Path
) -> int:
    """Start argv with stdout/stderr appended to log_path. Return the pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        fh = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            _with_unbuffered(argv),
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        return proc.pid

    cmd_path = log_path.with_name(log_path.stem + ".detached.cmd")
    write_detached_cmd(argv, cwd=cwd, log_path=log_path, cmd_path=cmd_path)
    pid = _wmi_create_process(f"cmd.exe /c {win_quote(str(cmd_path))}", cwd)
    pid_path.write_text(str(pid), encoding="utf-8")
    return pid
