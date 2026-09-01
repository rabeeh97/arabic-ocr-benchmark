"""Download extra Arabic display fonts for the synthetic printed-line renderer.

Fonts are NOT committed (binaries stay out of git); run this once per checkout:

    uv run python scripts/fetch_fonts.py

`training.synth` renders through `arabic_reshaper` + Pillow (no libraqm), which
needs fonts that carry the legacy **Arabic Presentation Forms** cmap. Many modern
SIL fonts (Scheherazade New, Lateef, Aref Ruqaa) dropped it and render tofu that
way, so this list is deliberately small and curated; `synth.fonts()` also
probes every face at load and skips any that fail. Without this script the
renderer still works on the one bundled Noto Naskh face.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "src" / "arabic_ocr_bench" / "assets" / "fonts"

_BASE = "https://github.com/google/fonts/raw/main/ofl"
FONTS = {
    "Amiri-Regular.ttf": f"{_BASE}/amiri/Amiri-Regular.ttf",
    "Amiri-Bold.ttf": f"{_BASE}/amiri/Amiri-Bold.ttf",
    "Amiri-Italic.ttf": f"{_BASE}/amiri/Amiri-Italic.ttf",
    "Amiri-BoldItalic.ttf": f"{_BASE}/amiri/Amiri-BoldItalic.ttf",
}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, url in FONTS.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 10_000:
            print(f"have  {name}")
            ok += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "arabic-ocr-bench"})
            data = urllib.request.urlopen(req, timeout=60).read()
            out.write_bytes(data)
            print(f"got   {name}  ({len(data) // 1024} KB)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {e}")
    print(f"\n{ok}/{len(FONTS)} fonts in {DEST}")
    print("synth.fonts() probes each at load and keeps only the ones that render.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
