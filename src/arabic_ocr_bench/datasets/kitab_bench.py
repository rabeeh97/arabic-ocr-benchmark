"""KITAB-Bench text-recognition adapter (line/segment-level OCR).

KITAB-Bench (arXiv:2502.14949) ships its OCR tasks as small, **ungated** HF
datasets under `ahmedheakl/arocrbench_*`, each a single `train` split with
columns `image`, `source`, and a ground-truth string that is `text` on some
subsets and `answer` on others. This one adapter serves any of them; the
concrete benchmark datasets are named in `configs/datasets.yaml`.

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

from collections.abc import Iterator

from ..imaging import cap_image
from .base import Sample

_GT_KEYS = ("text", "answer", "label", "ground_truth")


class KitabBench:
    track = "line"

    def __init__(
        self,
        subset: str,
        split: str = "train",
        limit: int | None = None,
        offset: int = 0,
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
        self.max_dim = max_dim
        self._rows = None

    def _load(self):
        if self._rows is None:
            from datasets import load_dataset

            ds = load_dataset(self.hf_id, split=self.split)
            end = len(ds) if self.limit is None else min(self.offset + self.limit, len(ds))
            ds = ds.select(range(min(self.offset, len(ds)), end))
            self._rows = list(ds)
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
        for i, row in enumerate(self._load()):
            idx = self.offset + i
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
