"""Training / fine-tuning subsystem for arabic_ocr_bench.

Deliberately separate from the eval code. Nothing here is imported by the
benchmark runner; trained artifacts re-enter the benchmark only as ordinary
`configs/models.yaml` entries scored by the existing `bench run` path.

See docs/training-plan.md for the experiment program.
"""

from __future__ import annotations
