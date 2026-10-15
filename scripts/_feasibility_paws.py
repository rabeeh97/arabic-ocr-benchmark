"""Feasibility of a PAW library: do the large connected components of a line image match the label's PAWs one to one?
Large component = body (area above a fraction of the line's median body area, not dot-sized); dots and hamzas are small.
Also: lines per manuscript in RASAM (writer-adaptation units)."""
import sys
from collections import Counter

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

W = "/Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike"
sys.path.insert(0, W + "/scripts")
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from geom_data import load_lines  # noqa: E402
from geom_inkcrop_export import clean, inkcrop  # noqa: E402
from arabic_ocr_bench.characterize.text import analyze_letters  # noqa: E402


def label_paws(t: str) -> int:
    lets = analyze_letters(t)
    n = len({li.paw for li in lets})
    n += sum(1 for w in t.split() if not any(ch in "ابتثجحخدذرزسشصضطظعغفقكلمنهويىةءأإآؤئ" for ch in w))  # digit / punctuation tokens
    return n


def image_bodies(g: np.ndarray) -> int:
    g = inkcrop(g)
    try:
        ink = g < threshold_otsu(g)
    except ValueError:
        return -1
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    area = ndi.sum_labels(ink, lab, np.arange(1, n + 1))
    sl = ndi.find_objects(lab)
    h = np.array([s[0].stop - s[0].start for s in sl])
    big = area >= max(20, 0.15 * np.median(area[area >= np.percentile(area, 50)]))
    tall = h >= 0.25 * g.shape[0]
    return int((big & tall).sum() + (big & ~tall & (area > 4 * np.median(area))).sum())


for tag, n in (("muharaf_train", 400), ("khatt", 200), ("nakba", 400)):
    res = []
    for g, lab in load_lines(tag, n):
        t = clean(lab)
        if len(t) < 5:
            continue
        res.append((image_bodies(g), label_paws(t)))
    d = np.array([a - b for a, b in res if a >= 0])
    print(f"{tag:14s} lines {len(d)}  exact match {100 * np.mean(d == 0):.0f}%  within ±1 {100 * np.mean(abs(d) <= 1):.0f}%  "
          f"within ±2 {100 * np.mean(abs(d) <= 2):.0f}%  (image − label: median {np.median(d):+.0f})")

import glob, pyarrow as pa  # noqa: E401,E402
from geom_data import HF, _table  # noqa: E402
for name in ("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow", "calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow"):
    t = _table(sorted(glob.glob(str(HF / name)))[0])
    c = Counter(t.column("manuscript_name").to_pylist())
    print(name.split("/")[0], len(c), "manuscripts; lines per manuscript:", sorted(c.values(), reverse=True)[:12])
