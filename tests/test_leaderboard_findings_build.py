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


def test_ensure_findings_rebuilds_when_stale(tmp_path):
    import shutil

    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    # give _run_dirs something to find: a summary.csv beside each run's jsonl
    for rd in sorted(p.parent for p in dst.glob("*/predictions.jsonl")):
        (rd / "summary.csv").write_text("model,dataset,n\n")
    stale = dst / "findings.json"
    stale.write_text(json.dumps({"meta": {"stale": True}, "cached": True}))
    old = time.time() - 3600
    os.utime(stale, (old, old))
    # touch a run file so it's newer than the cache
    run = next(dst.glob("*/predictions.jsonl"))
    now = time.time() + 5
    os.utime(run, (now, now))
    doc = mk._ensure_findings(dst)
    assert "model_landscape" in doc and doc.get("cached") is not True


def test_full_page_has_both_tabs(tmp_path):
    import shutil
    import xml.etree.ElementTree as ET

    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    rows = mk.collect(dst)
    findings = mk._ensure_findings(dst)
    page = mk.render(rows, "Test board", dst, findings=findings)
    assert 'data-tab="leaderboard"' in page and 'data-tab="findings"' in page
    assert 'data-view="leaderboard"' in page and 'data-view="findings"' in page
    assert "findings-view" in page
    assert "�" not in page
    # the rendered findings-view fragment is standalone well-formed
    open_tag = '<div class="findings-view">'
    inner = page.split(open_tag, 1)[1].rsplit("</div>\n</div>\n<footer>", 1)[0]
    ET.fromstring(open_tag + inner + "</div>")


def test_render_without_findings_still_works(tmp_path):
    import shutil

    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    rows = mk.collect(dst)
    page = mk.render(rows, "Test board", dst)  # findings defaults to None
    assert "<main" in page


def test_findings_view_render_failure_degrades_but_page_still_renders(tmp_path, monkeypatch):
    """A structurally-wrong findings doc that blows up a findings renderer must
    degrade the findings tab to its empty shell, not crash the whole board."""
    import shutil

    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    rows = mk.collect(dst)

    real = mk._lf.render_findings_view
    calls = {"n": 0}

    def flaky(f):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom: golden-shape unpack failed")
        return real(f)

    monkeypatch.setattr(mk._lf, "render_findings_view", flaky)

    page = mk.render(rows, "T", dst, findings={"error_anatomy": {"sid_split": "not-a-dict"}})
    assert "<main" in page
    assert 'data-view="leaderboard"' in page
    assert 'data-view="findings"' in page
    assert 'class="findings-view"' in page  # fallback empty shell present
    assert calls["n"] == 2  # first raised, fallback call succeeded


def test_ensure_findings_rebuilds_when_cache_is_not_a_dict(tmp_path):
    """A valid-JSON but non-dict findings.json (e.g. ``[]``) must be treated
    like a decode error and rebuilt, not returned as-is."""
    import shutil

    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    for rd in sorted(p.parent for p in dst.glob("*/predictions.jsonl")):
        (rd / "summary.csv").write_text("model,dataset,n\n")
    fj = dst / "findings.json"
    fj.write_text("[]")  # valid JSON, wrong type
    now = time.time() + 10
    os.utime(fj, (now, now))  # fresher than any run file
    doc = mk._ensure_findings(dst)
    assert isinstance(doc, dict) and "model_landscape" in doc
