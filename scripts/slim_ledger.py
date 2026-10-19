"""Shrink the ledger HTML so it loads on phones and slow links.

Re-encodes every embedded PNG/JPEG data URI as lossy WebP (kept as is when WebP is not smaller) and marks the images
loading="lazy" decoding="async". Idempotent: rerun it after any build_*_section.py run, because those embed fresh PNGs.
Usage: .geomvenv/bin/python scripts/slim_ledger.py [--ledger docs/research/paper-ledger.html] [--quality 62] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

from PIL import Image

DATA = re.compile(r'data:image/(png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)')
IMG = re.compile(r'<img\b([^>]*)>')


def to_webp(raw: bytes, quality: int) -> bytes | None:
    im = Image.open(io.BytesIO(raw))
    im.load()
    if im.mode in ('P', 'LA'):
        im = im.convert('RGBA')
    elif im.mode not in ('L', 'RGB', 'RGBA'):
        im = im.convert('RGB')
    buf = io.BytesIO()
    im.save(buf, format='WEBP', quality=quality, method=6)
    out = buf.getvalue()
    return out if len(out) < 0.9 * len(raw) else None


def slim(html: str, quality: int) -> tuple[str, dict]:
    stats = {'images': 0, 'converted': 0, 'before': 0, 'after': 0}

    def sub(m: re.Match) -> str:
        raw = base64.b64decode(m.group(2))
        stats['images'] += 1
        stats['before'] += len(m.group(2))
        out = to_webp(raw, quality)
        if out is None:
            stats['after'] += len(m.group(2))
            return m.group(0)
        b64 = base64.b64encode(out).decode()
        stats['converted'] += 1
        stats['after'] += len(b64)
        return f'data:image/webp;base64,{b64}'

    html = DATA.sub(sub, html)

    def lazy(m: re.Match) -> str:
        a = m.group(1)
        if 'data:image' not in a or 'loading=' in a:
            return m.group(0)
        return f'<img{a.rstrip("/ ")} loading="lazy" decoding="async">'

    return IMG.sub(lazy, html), stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger', type=Path, default=Path('docs/research/paper-ledger.html'))
    ap.add_argument('--quality', type=int, default=62)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    h = a.ledger.read_text(encoding='utf8')
    out, st = slim(h, a.quality)
    print(f"images {st['images']}, converted {st['converted']}; embedded {st['before'] / 1e6:.2f} -> {st['after'] / 1e6:.2f} MB; "
          f"file {len(h.encode('utf8')) / 1e6:.2f} -> {len(out.encode('utf8')) / 1e6:.2f} MB")
    if not a.dry_run:
        a.ledger.write_text(out, encoding='utf8')
        print('written', a.ledger)
