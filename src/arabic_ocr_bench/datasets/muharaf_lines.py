"""Muharaf-public line-level adapter.

HF dataset: aamijar/muharaf-public  (train 22,092 / validation 1,069 / test 1,334)
Columns: image (PIL), text (str). License CC-BY-NC-SA-4.0 (research use).

The benchmark uses the `test` split only. train / validation are reserved for a
future fine-tuning phase and must never be scored here.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

from ..imaging import cap_image
from .base import Sample

_HF_ID = "aamijar/muharaf-public"
# below this, stream instead of downloading the whole ~70 MB test parquet
_STREAM_MAX = 300


class MuharafLines:
    name = "muharaf_lines"
    track = "line"

    def __init__(
        self,
        split: str = "test",
        limit: int | None = None,
        offset: int = 0,
        indices: list[int] | None = None,
        hf_id: str = _HF_ID,
    ):
        self.split = split
        self.limit = limit
        self.offset = offset
        # explicit absolute row indices (accumulate / retry); wins over offset+limit
        self.indices = None if indices is None else list(indices)
        self.hf_id = hf_id
        self._rows = None
        self._abs: list[int] = []

    def _load(self):
        if self._rows is not None:
            return self._rows
        from datasets import load_dataset

        if self.indices is not None:
            want = sorted(i for i in self.indices if i >= 0)
            if want and want[-1] < _STREAM_MAX:
                # small slice (accumulate / retry) — stream the first N rows
                # instead of downloading + arrow-building the whole 500 MB split
                it = load_dataset(self.hf_id, split=self.split, streaming=True)
                head = list(itertools.islice(it, want[-1] + 1))
                sel = [i for i in want if i < len(head)]
                self._rows = [head[i] for i in sel]
            else:
                ds = load_dataset(self.hf_id, split=self.split)
                sel = [i for i in want if i < len(ds)]
                self._rows = list(ds.select(sel))
            self._abs = sel
            return self._rows

        window = None if self.limit is None else self.offset + self.limit
        if window is not None and window <= _STREAM_MAX:
            it = load_dataset(self.hf_id, split=self.split, streaming=True)
            self._rows = list(itertools.islice(it, self.offset, window))
        else:
            ds = load_dataset(self.hf_id, split=self.split)
            end = len(ds) if self.limit is None else min(self.offset + self.limit, len(ds))
            ds = ds.select(range(min(self.offset, len(ds)), end))
            self._rows = list(ds)
        self._abs = list(range(self.offset, self.offset + len(self._rows)))
        return self._rows

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[Sample]:
        rows = self._load()
        for i, row in enumerate(rows):
            idx = self._abs[i] if i < len(self._abs) else self.offset + i
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=cap_image(row["image"].convert("RGB"), 1024),
                ground_truth=row["text"],
                track=self.track,
                meta={"source": self.hf_id, "split": self.split, "row": idx},
            )
