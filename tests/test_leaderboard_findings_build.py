import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import make_leaderboard as mk

FIX = Path(__file__).parent / "fixtures" / "mini_results"


def test_ensure_findings_builds_when_missing(tmp_path):
    import shutil

    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    doc = mk._ensure_findings(dst)
    assert "model_landscape" in doc and "meta" in doc
    assert (dst / "findings.json").exists()


def test_ensure_findings_reads_cache(tmp_path):
    import shutil

    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    (dst / "findings.json").write_text(json.dumps({"meta": {}, "cached": True}))
    now = time.time() + 10
    os.utime(dst / "findings.json", (now, now))
    doc = mk._ensure_findings(dst)
    assert doc.get("cached") is True
