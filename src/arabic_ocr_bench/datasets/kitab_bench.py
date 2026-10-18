"""KITAB-Bench text-recognition adapter (line/segment-level OCR).

KITAB-Bench (arXiv:2502.14949) ships its OCR tasks as small, **ungated** HF
datasets under `ahmedheakl/arocrbench_*`, each a single `train` split with
columns `image`, `source`, and a ground-truth string that is `text` on some
subsets and `answer` on others. This one adapter serves any of them; the
concrete benchmark datasets are named in `configs/datasets.yaml`.

It is also reused for any other single-split HF `image-to-text` line dataset
that keeps its image in an `image` column and its transcription in one of
`_GT_KEYS` — e.g. `nakba_lines` (`U4RASD/omar-al-saleh-manuscripts-segments`,
NAKBA NLP 2026). Pass the full `org/name` as `subset` and set `split`.

Subsets we use (see datasets.yaml for the full list):
  arocrbench_khatt           modern handwritten Arabic (KHATT) — line images
  arocrbench_muharaf         historical handwritten Arabic (Muharaf) segments
  arocrbench_historicalbooks historical printed books
  arocrbench_hindawi         digitised printed books (Hindawi)
  arocrbench_adab            short printed segments
  arocrbench_arabicocr       printed Arabic OCR set

License / citation: per each source dataset and the KITAB-Bench paper.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterator

from ..imaging import cap_image
from .base import Sample

_GT_KEYS = ("text", "answer", "label", "ground_truth", "transcription", "gt", "target")

# HuggingFace's canonical validation split is `validation`. Older configs and
# some cached builders used `val` / `dev`. RASAM-1/2 on the Hub today only
# expose train/validation/test — a fresh venv raises
# `Unknown split "val"` (Fanar prepare, 2026-09-20).
_SPLIT_ALIASES = {
    "val": ("validation", "valid", "dev"),
    "validation": ("val", "valid", "dev"),
    "valid": ("validation", "val"),
    "dev": ("validation", "val"),
}


def resolve_hf_split(requested: str, available: list[str] | tuple[str, ...]) -> str:
    """Map a requested split name onto one the dataset actually ships."""
    names = list(available)
    if requested in names:
        return requested
    for alias in _SPLIT_ALIASES.get(requested, ()):
        if alias in names:
            return alias
    raise ValueError(f'Unknown split "{requested}". Should be one of {names}.')


def _splits_from_unknown_split_error(exc: BaseException) -> list[str] | None:
    m = re.search(r"Should be one of (\[.*\])", str(exc))
    if not m:
        return None
    try:
        names = ast.literal_eval(m.group(1))
    except (SyntaxError, ValueError):
        return None
    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        return names
    return None


class KitabBench:
    track = "line"

    def __init__(
        self,
        subset: str,
        split: str = "train",
        limit: int | None = None,
        offset: int = 0,
        indices: list[int] | None = None,
        name: str | None = None,
        max_dim: int = 1600,
    ):
        # `subset` is the HF repo id suffix, e.g. "arocrbench_khatt", or a full
        # "org/name"; `name` is the benchmark-facing dataset name (defaults to it)
        self.hf_id = subset if "/" in subset else f"ahmedheakl/{subset}"
        self.name = name or self.hf_id.split("/")[-1]
        self.split = split
        self.limit = limit
        self.offset = offset
        # explicit absolute row indices (accumulate / retry); wins over offset+limit
        self.indices = None if indices is None else list(indices)
        self.max_dim = max_dim
        self._rows = None
        self._abs: list[int] = []

    def _load(self):
        if self._rows is None:
            from datasets import load_dataset

            token = os.environ.get("HF_TOKEN")
            try:
                ds = load_dataset(self.hf_id, split=self.split, token=token)
            except ValueError as e:
                # `datasets` error: Unknown split "val". Should be one of [...].
                available = _splits_from_unknown_split_error(e)
                if available is None:
                    raise
                resolved = resolve_hf_split(self.split, available)
                if resolved == self.split:
                    raise
                ds = load_dataset(self.hf_id, split=resolved, token=token)
                self.split = resolved
            n = len(ds)
            if self.indices is not None:
                sel = [i for i in self.indices if 0 <= i < n]
            else:
                start = min(self.offset, n)
                end = n if self.limit is None else min(self.offset + self.limit, n)
                sel = list(range(start, end))
            self._abs = sel
            self._rows = list(ds.select(sel))
        return self._rows

    @staticmethod
    def _gt(row: dict) -> str:
        for k in _GT_KEYS:
            if k in row and isinstance(row[k], str):
                return row[k]
        raise KeyError(f"no ground-truth column in {list(row)} (tried {_GT_KEYS})")

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[Sample]:
        rows = self._load()
        for i, row in enumerate(rows):
            idx = self._abs[i] if i < len(self._abs) else self.offset + i
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=cap_image(row["image"].convert("RGB"), self.max_dim),
                ground_truth=self._gt(row),
                track=self.track,
                meta={
                    "source": self.hf_id,
                    "subset_source": row.get("source"),
                    "split": self.split,
                    "row": idx,
                },
            )
