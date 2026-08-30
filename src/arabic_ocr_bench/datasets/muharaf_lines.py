"""Muharaf-public line-level adapter.

HF dataset: aamijar/muharaf-public  (train 22,100 / validation 1,070 / test 1,330)
Columns: image (PIL), text (str). License CC-BY-NC-SA-4.0 (research use).

The benchmark uses the `test` split only. train / validation are reserved for a
future fine-tuning phase and must never be scored here.
"""

from __future__ import annotations

from typing import Iterator

from .base import Sample

_HF_ID = "aamijar/muharaf-public"


class MuharafLines:
    name = "muharaf_lines"
    track = "line"

    def __init__(self, split: str = "test", limit: int | None = None, hf_id: str = _HF_ID):
        self.split = split
        self.limit = limit
        self.hf_id = hf_id
        self._ds = None

    def _load(self):
        if self._ds is None:
            from datasets import load_dataset

            ds = load_dataset(self.hf_id, split=self.split)
            if self.limit is not None:
                ds = ds.select(range(min(self.limit, len(ds))))
            self._ds = ds
        return self._ds

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[Sample]:
        ds = self._load()
        for idx, row in enumerate(ds):
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=row["image"].convert("RGB"),
                ground_truth=row["text"],
                track=self.track,
                meta={"source": self.hf_id, "split": self.split, "row": idx},
            )
