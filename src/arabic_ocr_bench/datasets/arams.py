"""AraMS-28k-HTR adapter — real historical-manuscript line crops (Zenodo).

Source: Zenodo `10.5281/zenodo.21499649` ("AraMS-28k-HTR"), ~27,971 real,
pre-segmented handwritten-manuscript line crops. Not an HF dataset —
`scripts/fetch_arams.py` downloads and extracts it into a local directory of
`<id>.png` + `<id>.gt.txt` pairs (image + UTF-8 transcription sidecar), whose
path is passed here as `root`.

The corpus ships with no official train/test split, so this adapter carves
its own, deterministically: every id is sorted then seed-shuffled, and a
fixed-size slice is reserved as `split="test"` (the benchmark's held-out
`arams_htr` row in `configs/datasets.yaml`) with the remainder as
`split="train"` (a `data.sources` entry in a training config). The split is a
pure function of the id list plus `TEST_SEED`/`TEST_SIZE` below, so it can
never drift between the eval entry and any training source pointed at the
same `root` — there is no other guard against training/eval leakage for this
dataset (it has no `hf_id` for `check_source_not_scored` to key off).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from ..imaging import cap_image
from .base import Sample

TEST_SEED = 0
TEST_SIZE = 1800  # comparable order to nakba_lines (372) / muharaf_lines (1,334)


def _all_ids(root: Path) -> list[str]:
    return sorted(p.stem for p in root.glob("*.png"))


def _split_ids(root: Path, split: str) -> list[str]:
    ids = _all_ids(root)
    shuffled = ids[:]
    random.Random(TEST_SEED).shuffle(shuffled)
    test_ids = set(shuffled[:TEST_SIZE])
    if split == "test":
        return [i for i in ids if i in test_ids]
    if split == "train":
        return [i for i in ids if i not in test_ids]
    raise ValueError(f"AraMSHTR split must be 'train' or 'test', got {split!r}")


class AraMSHTR:
    name = "arams_htr"
    track = "line"

    def __init__(
        self,
        root: str,
        split: str = "test",
        limit: int | None = None,
        offset: int = 0,
        indices: list[int] | None = None,
        name: str | None = None,
        max_dim: int = 1024,
    ):
        self.root = Path(root)
        self.split = split
        self.limit = limit
        self.offset = offset
        # explicit absolute row indices (accumulate / retry); wins over offset+limit
        self.indices = None if indices is None else list(indices)
        self.name = name or self.__class__.name
        self.max_dim = max_dim
        self._ids: list[str] | None = None
        self._abs: list[int] = []

    def _load_ids(self) -> list[str]:
        if self._ids is not None:
            return self._ids
        all_ids = _split_ids(self.root, self.split)
        if self.indices is not None:
            sel = [i for i in self.indices if 0 <= i < len(all_ids)]
        else:
            start = min(self.offset, len(all_ids))
            end = (
                len(all_ids)
                if self.limit is None
                else min(self.offset + self.limit, len(all_ids))
            )
            sel = list(range(start, end))
        self._abs = sel
        self._ids = [all_ids[i] for i in sel]
        return self._ids

    def __len__(self) -> int:
        return len(self._load_ids())

    def __iter__(self) -> Iterator[Sample]:
        ids = self._load_ids()
        for i, item_id in enumerate(ids):
            idx = self._abs[i] if i < len(self._abs) else self.offset + i
            img = Image.open(self.root / f"{item_id}.png").convert("RGB")
            gt = (self.root / f"{item_id}.gt.txt").read_text(encoding="utf-8").strip()
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=cap_image(img, self.max_dim),
                ground_truth=gt,
                track=self.track,
                meta={
                    "source": "zenodo:10.5281/zenodo.21499649",
                    "split": self.split,
                    "row": idx,
                    "item_id": item_id,
                },
            )
