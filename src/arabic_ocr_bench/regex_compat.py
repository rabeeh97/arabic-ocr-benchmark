"""Make ``import transformers`` work when Windows Application Control blocks ``regex``.

The third-party ``regex`` package ships an unsigned ``_regex*.pyd``. Smart App
Control / WDAC can refuse to load it, which then fails ``import transformers``.
Stdlib ``re`` is enough for Qwen2-VL inference and LoRA SFT on this bench.
"""

from __future__ import annotations

import sys


def ensure_regex() -> None:
    try:
        import regex  # noqa: F401
    except ImportError:
        sys.modules.pop("regex", None)
        import re

        sys.modules["regex"] = re
