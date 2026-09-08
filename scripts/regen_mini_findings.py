"""Regenerate the golden findings fixture. Run twice; confirm no diff."""

import json
from pathlib import Path

from arabic_ocr_bench.analysis.findings import build

fix = Path("tests/fixtures/mini_results")
out = Path("tests/fixtures/mini_findings.json")
out.write_text(json.dumps(build(fix), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
print("wrote", out)
