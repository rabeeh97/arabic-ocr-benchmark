import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def test_build_is_byte_stable_and_emits_09():
    subprocess.run([sys.executable, "scripts/build_notebooks.py"], cwd=REPO, check=True)
    nb = REPO / "notebooks" / "09_findings.ipynb"
    assert nb.exists()
    doc = json.loads(nb.read_text())
    assert doc["cells"] and all(c["cell_type"] in ("markdown", "code") for c in doc["cells"])
    # every code cell has empty outputs (committed clean)
    assert all(c.get("outputs", []) == [] for c in doc["cells"] if c["cell_type"] == "code")
    # second run makes no diff
    before = nb.read_bytes()
    subprocess.run([sys.executable, "scripts/build_notebooks.py"], cwd=REPO, check=True)
    assert nb.read_bytes() == before


def test_09_references_the_analysis_api():
    nb = json.loads((REPO / "notebooks" / "09_findings.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "from arabic_ocr_bench.analysis.findings import build" in src
