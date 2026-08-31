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

    def __init__(self, split: str = "test", limit: int | None = None, hf_id: str = _HF_ID):
        self.split = split
        self.limit = limit
        self.hf_id = hf_id
        self._rows = None

    def _load(self):
        if self._rows is not None:
            return self._rows
        from datasets import load_dataset

        if self.limit is not None and self.limit <= _STREAM_MAX:
            it = load_dataset(self.hf_id, split=self.split, streaming=True)
            self._rows = list(itertools.islice(it, self.limit))
        else:
            ds = load_dataset(self.hf_id, split=self.split)
            if self.limit is not None:
                ds = ds.select(range(min(self.limit, len(ds))))
            self._rows = list(ds)
        return self._rows

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[Sample]:
        for idx, row in enumerate(self._load()):
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=cap_image(row["image"].convert("RGB"), 1024),
                ground_truth=row["text"],
                track=self.track,
                meta={"source": self.hf_id, "split": self.split, "row": idx},
            )
