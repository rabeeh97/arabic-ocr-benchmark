"""Find the newest local checkpoint so an interrupted run resumes.

Colab drops the VM without warning. Point ``output_dir`` at Google Drive and a
re-run of the same config continues from the last ``checkpoint-<step>`` written
there instead of starting over.
"""

from __future__ import annotations

import re
from pathlib import Path

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(output_dir: str | Path) -> str | None:
    """Path (as str) to the highest-step ``checkpoint-N`` under ``output_dir``, or None."""
    d = Path(output_dir)
    if not d.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in d.iterdir():
        m = _CKPT_RE.match(child.name)
        if child.is_dir() and m:
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, child)
    return str(best[1]) if best else None


def resume_from(output_dir: str | Path, *, resume: bool) -> str | None:
    """The `resume_from_checkpoint` value to hand `Trainer.train` (None to start fresh)."""
    return latest_checkpoint(output_dir) if resume else None
