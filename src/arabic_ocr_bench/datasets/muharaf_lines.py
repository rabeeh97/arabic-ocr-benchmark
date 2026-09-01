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
        hf_id: str = _HF_ID,
    ):
        self.split = split
        self.limit = limit
        self.offset = offset
        self.hf_id = hf_id
        self._rows = None

    def _load(self):
        if self._rows is not None:
            return self._rows
        from datasets import load_dataset

        window = None if self.limit is None else self.offset + self.limit
        if window is not None and window <= _STREAM_MAX:
            it = load_dataset(self.hf_id, split=self.split, streaming=True)
            self._rows = list(itertools.islice(it, self.offset, window))
        else:
            ds = load_dataset(self.hf_id, split=self.split)
            end = len(ds) if self.limit is None else min(self.offset + self.limit, len(ds))
            ds = ds.select(range(min(self.offset, len(ds)), end))
            self._rows = list(ds)
        return self._rows

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[Sample]:
        for i, row in enumerate(self._load()):
            idx = self.offset + i
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=cap_image(row["image"].convert("RGB"), 1024),
                ground_truth=row["text"],
                track=self.track,
                meta={"source": self.hf_id, "split": self.split, "row": idx},
            )
