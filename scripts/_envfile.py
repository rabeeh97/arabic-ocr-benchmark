"""Load KEY=VALUE experiment.env files (stdlib, no python-dotenv).

Per-model dirs often have no copy; the suite keeps the real token at
``<suite>/experiment.env``. Load the model file first (wins), then the
parent file to fill gaps.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines. Existing os.environ keys are left alone."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_experiment_env(root: Path) -> None:
    """Load ``<root>/experiment.env``, then ``<root>/../experiment.env``."""
    root = Path(root)
    load_env_file(root / "experiment.env")
    load_env_file(root.parent / "experiment.env")
