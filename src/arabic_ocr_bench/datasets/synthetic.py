"""A tiny offline dataset for pipeline validation — no network, no HF.

Ten hand-written (ASCII-safe to type, real Arabic strings) line samples with a
plain rendered image. The mock clients ignore the image; a real model will just
do poorly on the un-shaped render, which is fine — this dataset exists to prove
runner -> report -> leaderboard works end to end without downloading anything.
"""

from __future__ import annotations

from collections.abc import Iterator

from PIL import Image, ImageDraw

from .base import Sample

_LINES = [
    "بسم الله الرحمن الرحيم",
    "الحمد لله رب العالمين",
    "السلام عليكم ورحمة الله",
    "كتب الطالب الدرس في الدفتر",
    "المدرسة قريبة من البيت",
    "ذهب أحمد إلى السوق صباحا",
    "القراءة غذاء العقل والروح",
    "شمس الصباح تنير الطريق",
    "المكتبة مفتوحة يوم الجمعة",
    "الصديق وقت الضيق معروف",
]


class Synthetic:
    name = "synthetic"
    track = "line"

    def __init__(self, split: str = "test", limit: int | None = None, **_ignored):
        self.split = split
        self.limit = limit

    def _rows(self):
        rows = list(enumerate(_LINES))
        if self.limit is not None:
            rows = rows[: self.limit]
        return rows

    def __len__(self) -> int:
        return len(self._rows())

    def __iter__(self) -> Iterator[Sample]:
        for idx, text in self._rows():
            img = Image.new("RGB", (900, 80), "white")
            ImageDraw.Draw(img).text((10, 30), text, fill="black")
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=img,
                ground_truth=text,
                track=self.track,
                meta={"source": "synthetic", "split": self.split, "row": idx},
            )
