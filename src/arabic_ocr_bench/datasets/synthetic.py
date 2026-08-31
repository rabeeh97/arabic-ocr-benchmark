"""A tiny offline dataset for pipeline validation — no network, no HF.

Ten short Arabic lines rendered as clean printed text with a bundled Naskh font
(Arabic shaping + bidi applied first, so the image is genuinely legible). It is
the pipeline's *easy* anchor: a competent OCR / VLM model should score a very low
CER here, and `mock_echo` / `mock_empty` still bound it at 0 % / 100 %. Use it to
smoke-test runner -> report -> leaderboard, and to sanity-check a hosted-API key,
without downloading anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib import resources

from PIL import Image, ImageDraw, ImageFont

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

_FONT_SIZE = 44
_IMG_W, _IMG_H = 1100, 96


def _shape(text: str) -> str:
    """Presentation-form + visual bidi order, so a non-shaping renderer (Pillow
    without libraqm) still draws connected RTL Arabic. Falls back to the raw
    string if the optional deps are missing."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def _font() -> ImageFont.FreeTypeFont:
    path = resources.files("arabic_ocr_bench.assets") / "NotoNaskhArabic-Regular.ttf"
    with resources.as_file(path) as p:
        return ImageFont.truetype(str(p), _FONT_SIZE)


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
        font = _font()
        for idx, text in self._rows():
            img = Image.new("RGB", (_IMG_W, _IMG_H), "white")
            draw = ImageDraw.Draw(img)
            shaped = _shape(text)
            w = draw.textlength(shaped, font=font)
            draw.text((_IMG_W - w - 20, 20), shaped, fill="black", font=font)
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=img,
                ground_truth=text,
                track=self.track,
                meta={"source": "synthetic", "split": self.split, "row": idx},
            )
