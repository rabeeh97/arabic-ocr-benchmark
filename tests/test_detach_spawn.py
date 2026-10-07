"""Detached spawn must survive Windows OpenSSH logout (not just return a pid)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _detach_spawn import spawn_surviving_logout, win_quote, write_detached_cmd


def test_win_quote_wraps_spaces_and_doubles_internal_quotes():
    assert win_quote(r"C:\no-spaces.exe") == r'"C:\no-spaces.exe"'
    assert win_quote(r"C:\Program Files\python.exe") == r'"C:\Program Files\python.exe"'
    assert win_quote(r'C:\say "hi"') == r'"C:\say ""hi"""'


def test_write_detached_cmd_cds_then_appends_stdout_to_log(tmp_path: Path):
    cmd_path = tmp_path / "detached.cmd"
    log_path = tmp_path / "run.log"
    cwd = tmp_path / "repo"
    argv = [r"C:\Python\python.exe", r"C:\repo\scripts\run.py", "--dir", r"C:\exp"]
    write_detached_cmd(argv, cwd=cwd, log_path=log_path, cmd_path=cmd_path)
    text = cmd_path.read_text(encoding="utf-8")
    assert text.startswith("@echo off")
    assert f'cd /d "{cwd}"' in text
    assert ">>" in text and str(log_path) in text
    assert "2>&1" in text
    assert "--detach" not in text
    # unbuffered so Get-Content -Wait sees tqdm / heartbeats live
    assert '"-u"' in text


def test_pidfile_ignores_wmi_cmd_wrapper():
    from _detach_spawn import pidfile_is_other_worker

    # grandchild python (venv trampoline) vs WMI cmd.exe pid in the pidfile
    assert pidfile_is_other_worker(12632, 99, 100, alive=True, is_cmd=True) is False
    assert pidfile_is_other_worker(12632, 99, 100, alive=True, is_cmd=False) is True
    assert pidfile_is_other_worker(99, 99, 50, alive=True, is_cmd=False) is False
    assert pidfile_is_other_worker(50, 99, 50, alive=True, is_cmd=False) is False
    assert pidfile_is_other_worker(12632, 99, 100, alive=False, is_cmd=False) is False
    assert pidfile_is_other_worker(0, 99, 100, alive=True, is_cmd=False) is False


def test_stdout_is_regular_file_follows_redirection(tmp_path: Path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(scripts)!r})\n"
        "from _detach_spawn import stdout_is_regular_file\n"
        "sys.exit(0 if stdout_is_regular_file() else 2)\n",
        encoding="utf-8",
    )
    redirected = tmp_path / "out.txt"
    with redirected.open("w", encoding="utf-8") as fh:
        rc = __import__("subprocess").run([sys.executable, str(probe)], stdout=fh).returncode
    assert rc == 0
    piped = __import__("subprocess").run(
        [sys.executable, str(probe)], stdout=__import__("subprocess").PIPE
    )
    assert piped.returncode == 2


def test_emit_log_line_does_not_duplicate_when_stdout_is_the_log(tmp_path: Path):
    # cmd.exe >> log holds an exclusive writer on Windows; a second open('a')
    # is PermissionError 13. Also, writing both stdout and the file would
    # duplicate every line. When stdout is already the log, print once.
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    log = tmp_path / "suite.log"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(scripts)!r})\n"
        "from pathlib import Path\n"
        "from _detach_spawn import emit_log_line\n"
        f"emit_log_line(Path({str(log)!r}), 'once')\n",
        encoding="utf-8",
    )
    with log.open("w", encoding="utf-8") as fh:
        __import__("subprocess").run([sys.executable, "-u", str(probe)], stdout=fh, check=True)
    assert log.read_text(encoding="utf-8").count("once") == 1


def test_unix_spawn_writes_pid_and_log(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("unix session-detach path")
    log_path = tmp_path / "out.log"
    pid_path = tmp_path / "out.pid"
    marker = tmp_path / "done.txt"
    argv = [
        sys.executable,
        "-c",
        f"import pathlib; pathlib.Path(r'{marker}').write_text('ok', encoding='utf-8'); print('hello', flush=True)",
    ]
    pid = spawn_surviving_logout(argv, cwd=tmp_path, log_path=log_path, pid_path=pid_path)
    assert int(pid_path.read_text(encoding="utf-8")) == pid
    deadline = time.time() + 5
    while time.time() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.read_text(encoding="utf-8") == "ok"
    deadline = time.time() + 5
    while time.time() < deadline and "hello" not in log_path.read_text(encoding="utf-8"):
        time.sleep(0.05)
    assert "hello" in log_path.read_text(encoding="utf-8")
