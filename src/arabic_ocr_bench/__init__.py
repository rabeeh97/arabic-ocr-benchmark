"""arabic-ocr-bench — Arabic cursive script recognition benchmark.

Importing the package loads a repo-root `.env` (if present) so API keys don't
need to be exported every run. Real environment variables always win.
"""

from __future__ import annotations


def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


_load_dotenv()
