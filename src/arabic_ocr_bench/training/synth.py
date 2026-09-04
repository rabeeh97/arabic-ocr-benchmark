"""Synthetic printed-Arabic line generation for the M1 (printed-degraded) work.

Pure PIL + numpy, no torch. A text corpus (Wikipedia / classical / Hindawi) is
cleaned into short lines, filtered against the scored ground truth, rendered with
one of several Arabic fonts, and pushed through a randomized degradation stack
(blur, resample, noise, jitter, rotation, JPEG, speckle). `SyntheticPrinted` is a
dataset adapter that plugs straight into `training.data.assemble_dataset`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import tarfile
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..datasets.base import Sample
from .contamination import _match_key

# Arabic code points as \u escapes so the source stays copy-safe (see normalize.py).
_ARABIC_RE = re.compile(
    "[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_WS_RE = re.compile(r"\s+")
# split on . ! ? U+061F(?) U+061B(;) and newline; NOT U+060C which is a comma
_SENT_SPLIT_RE = re.compile("(?<=[.!?\u061F\u061B\n])\\s+")
# harakat / tanwin / shadda / sukun + dagger alef, as in normalize.py
_HARAKAT_RE = re.compile("[\u064B-\u065F\u0670]")
_STRIP_CHARS = " \t\n\r\u061B\u060C-\u2013\u2014\u2022*\u00B7"

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_BUNDLED_FONT = _ASSETS / "NotoNaskhArabic-Regular.ttf"

_DEGRADE_LEVELS = {
    "light": {"blur": 0.7, "noise": 6.0, "jpeg": (62, 88), "rot": 0.8, "scale": (0.90, 1.0)},
    "medium": {"blur": 1.2, "noise": 12.0, "jpeg": (35, 72), "rot": 1.6, "scale": (0.72, 1.0)},
    "heavy": {"blur": 1.9, "noise": 20.0, "jpeg": (20, 55), "rot": 2.6, "scale": (0.55, 0.95)},
}


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename so a killed process never leaves a
    half-written checkpoint (works on the Drive FUSE mount too)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def arabic_ratio(s: str) -> float:
    """Share of alphabetic characters that are Arabic script (0.0 if none)."""
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if _ARABIC_RE.match(c)) / len(alpha)


def sentences(
    raw_texts: Iterable[str],
    *,
    n: int,
    seed: int,
    min_len: int = 20,
    max_len: int = 180,
    min_arabic: float = 0.6,
) -> list[str]:
    """Clean raw text blobs into a deduped, Arabic, length-bounded line sample."""
    seen: set[str] = set()
    for blob in raw_texts:
        for part in _SENT_SPLIT_RE.split(blob or ""):
            t = _WS_RE.sub(" ", part).strip(_STRIP_CHARS)
            if min_len <= len(t) <= max_len and arabic_ratio(t) >= min_arabic:
                seen.add(t)
    pool = sorted(seen)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def drop_overlap(sents: list[str], forbidden_keys: set[str]) -> list[str]:
    """Remove lines whose normalized key collides with a scored ground truth."""
    return [s for s in sents if _match_key(s) not in forbidden_keys]


def scored_text_keys(
    test_dataset_names: list[str],
    *,
    cfg_path: str = "configs/datasets.yaml",
    limit: int | None = None,
) -> set[str]:
    """All `_match_key`s of the ground truth in the named benchmark test sets."""
    from ..datasets.registry import build_dataset, load_dataset_configs

    registry = load_dataset_configs(cfg_path)
    keys: set[str] = set()
    for name in test_dataset_names:
        for s in build_dataset(name, registry[name], limit=limit):
            keys.add(_match_key(s.ground_truth))
    return keys


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
# A few Arabic Presentation Forms-B code points. The reshaper + Pillow (no
# libraqm) path can only use fonts whose cmap covers these; modern SIL / Google
# UI fonts that dropped the block render tofu and are filtered out here.
_PRESENTATION_FORMS = (0xFE8D, 0xFEDF, 0xFEE3, 0xFEEB, 0xFB50)


def _has_presentation_forms(path: Path) -> bool:
    """True if the font can be used by the reshaper+Pillow renderer.

    Uses fontTools when available (exact cmap check); otherwise trusts the file
    (the curated download list and the bundled face are known-good).
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return True
    try:
        cmap = TTFont(str(path), fontNumber=0, lazy=True).getBestCmap()
    except Exception:  # noqa: BLE001 - unreadable font -> skip it
        return False
    return sum(cp in cmap for cp in _PRESENTATION_FORMS) >= 3


def fonts() -> list[Path]:
    """Usable Arabic faces: the presentation-form-capable files under
    assets/fonts/ plus the bundled Noto Naskh (always included, deduped)."""
    d = _ASSETS / "fonts"
    found = sorted(d.glob("*.ttf")) + sorted(d.glob("*.otf")) if d.is_dir() else []
    usable = [p for p in found if _has_presentation_forms(p)]
    if _BUNDLED_FONT not in usable:
        usable.append(_BUNDLED_FONT)
    return usable


def _shape(text: str) -> str:
    """Presentation-form + visual bidi order so Pillow (no libraqm) draws RTL."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def render_line(
    text: str, font_path: str | Path | None = None, *, size: int = 40, pad: int = 12
) -> Image.Image:
    """Draw one line of Arabic as dark ink on white. Width scales with the text."""
    font = ImageFont.truetype(str(font_path or fonts()[0]), size)
    shaped = _shape(text)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    box = probe.textbbox((0, 0), shaped, font=font)
    w = (box[2] - box[0]) + 2 * pad
    h = (box[3] - box[1]) + 2 * pad
    im = Image.new("RGB", (max(w, h + 1), max(h, size // 2 + 2 * pad)), "white")
    ImageDraw.Draw(im).text(
        (pad - box[0], pad - box[1]), shaped, font=font, fill=(15, 15, 15)
    )
    return im


def degrade(img: Image.Image, rng: np.random.Generator, level: str = "medium") -> Image.Image:
    """Randomized scan-artifact stack; output keeps the input size."""
    p = _DEGRADE_LEVELS[level]
    W, H = img.size
    im = img.convert("RGB").rotate(
        float(rng.uniform(-p["rot"], p["rot"])),
        resample=Image.BILINEAR,
        fillcolor=(255, 255, 255),
    )
    s = float(rng.uniform(*p["scale"]))
    im = im.resize((max(1, int(W * s)), max(1, int(H * s))), Image.BILINEAR).resize(
        (W, H), Image.BILINEAR
    )
    im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, p["blur"]))))

    arr = np.asarray(im).astype(np.float32)
    arr *= float(rng.uniform(0.85, 1.08))
    arr += float(rng.uniform(-12, 6)) + rng.uniform(-6, 4, size=3)
    arr += rng.normal(0.0, p["noise"], size=arr.shape)
    speck = rng.random(arr.shape[:2])
    arr[speck < 0.004] = 0.0
    arr[speck > 0.997] = 255.0
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(p["jpeg"][0], p["jpeg"][1])))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --------------------------------------------------------------------------- #
# corpus loaders (lazy, network) — not exercised by unit tests
# --------------------------------------------------------------------------- #
def _stream_texts(candidates: list[tuple[str, str | None]], limit: int) -> list[str]:
    """Try each (hf_id, config) in turn; return texts from the first that loads.

    Dataset ids on the Hub move or vanish, so every corpus below is a list of
    fallbacks rather than a single id. If none resolve we return ``[]`` and let
    the caller carry on with whatever other corpora it asked for — a missing
    optional corpus must not abort a whole training-data build.
    """
    from datasets import load_dataset

    tried: list[str] = []
    for hf_id, config in candidates:
        tried.append(hf_id if config is None else f"{hf_id}:{config}")
        try:
            it = load_dataset(hf_id, config, split="train", streaming=True)
            out: list[str] = []
            for row in it:
                txt = next(
                (row[k] for k in ("text", "content", "poem_text", "poem", "body", "Text")
                 if isinstance(row.get(k), str) and row[k]),
                "",
            )
                if txt:
                    out.append(txt)
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception as e:  # noqa: BLE001 - any load failure -> try the next id
            print(f"  synth corpus: {hf_id} unavailable ({type(e).__name__}); trying next")
    print(f"  synth corpus: none of {tried} resolved — skipping this corpus")
    return []


def _load_wikipedia(limit: int) -> list[str]:
    return _stream_texts(
        [
            ("wikimedia/wikipedia", "20231101.ar"),
            ("wikipedia", "20220301.ar"),
        ],
        limit,
    )


def _load_classical(limit: int) -> list[str]:
    # Heritage / classical Arabic prose and poetry. OpenITI is the canonical
    # classical corpus; the arbml poetry sets are the lightweight fallback.
    return _stream_texts(
        [
            ("AbderrahmanSkiredj1/MLM_classical_arabic_openiti", None),
            ("moneypitts/classicalarabic", None),
            ("nehalelkaref/classical_arabic_poetry", None),
            ("arbml/ashaar", None),
        ],
        limit,
    )


def _load_hindawi(limit: int) -> list[str]:
    # Digitised printed books from the Hindawi Foundation.
    return _stream_texts(
        [
            ("alielfilali01/Hindawi-Books-dataset", None),
            ("khalidalt/hindawi", None),
            ("Abdallah4Zain/hindawi-arabic-sections", None),
        ],
        limit,
    )


_CORPUS_LOADERS = {
    "wikipedia": _load_wikipedia,
    "classical": _load_classical,
    "hindawi": _load_hindawi,
}


# --------------------------------------------------------------------------- #
# lazy image handle
# --------------------------------------------------------------------------- #
class LazyImage(Image.Image):
    """A PIL image backed by a file on disk that decodes only on first real use.

    ``assemble_dataset`` builds one row per training line, and the synthetic
    source alone can be 15k+ lines; holding every decoded bitmap in memory at
    once OOMs a Colab runtime (the "Assemble + class balance by source" cell in
    ``06_colab_prepare_data.ipynb``). When the source renders to a ``cache_dir``
    each line is written once as a PNG and represented here by its path, so the
    row lists stay tiny. The pixels are paged in lazily — one batch at a time by
    the trainer's collator, or per-sample by the notebook's preview cell.

    It subclasses ``PIL.Image.Image`` (not a bare wrapper) so it still satisfies
    ``isinstance(x, Image.Image)`` checks and ``datasets.Dataset.from_list``'s
    automatic ``Image()`` feature inference. Decoding is funnelled through the
    handful of entry points PIL routes pixel access through (``load``,
    ``convert``, ``__getstate__``) plus ``size``.
    """

    def __init__(self, path: str | Path):
        super().__init__()
        object.__setattr__(self, "_lazy_path", str(path))
        object.__setattr__(self, "_lazy_loaded", False)

    def _lazy_ensure(self) -> None:
        if self._lazy_loaded:
            return
        object.__setattr__(self, "_lazy_loaded", True)
        with Image.open(self._lazy_path) as im:
            im.load()
            # PIL's own pickle protocol — version-stable across Pillow releases.
            self.__setstate__(im.__getstate__())

    def load(self):
        self._lazy_ensure()
        return Image.Image.load(self)

    def convert(self, *args, **kwargs):
        self._lazy_ensure()
        return Image.Image.convert(self, *args, **kwargs)

    def __getstate__(self):
        self._lazy_ensure()
        return Image.Image.__getstate__(self)

    @property
    def size(self):
        self._lazy_ensure()
        return self._size

    @size.setter
    def size(self, value):
        # __setstate__ on older Pillow assigns .size directly; keep it working.
        object.__setattr__(self, "_size", value)

    def __reduce__(self):
        return (LazyImage, (self._lazy_path,))


# parameters that change a line's rendered pixels for a given row index. `n` /
# `limit` are excluded on purpose: they only set HOW MANY lines, not WHICH — so
# a 5k and a 15k config that agree on everything else render byte-identical
# images for rows 0..4999 and should share one render cache.
_RENDER_KEY_PARAMS = (
    "corpora", "texts", "seed", "degrade_level", "diacritics",
    "min_len", "max_len", "min_arabic", "drop_overlap_with", "corpus_docs", "sizes",
)


def synth_render_key(params: dict) -> str:
    """Stable short hash of the pixel-affecting synthetic-source params. Use it
    to name the render cache dir so it is shared across configs (and notebooks
    05 / 06) that produce the same images, and invalidated when they wouldn't."""
    spec = {}
    for k in _RENDER_KEY_PARAMS:
        v = params.get(k)
        if isinstance(v, tuple):
            v = list(v)
        spec[k] = v
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# dataset adapter
# --------------------------------------------------------------------------- #
class SyntheticPrinted:
    name = "synthetic_printed"
    track = "line"

    def __init__(
        self,
        *,
        corpora: list | None = None,
        texts: list[str] | None = None,
        n: int = 15000,
        seed: int = 0,
        degrade_level: str = "medium",
        diacritics: str = "mixed",  # "mixed" | "keep" | "none"
        min_len: int = 20,
        max_len: int = 180,
        min_arabic: float = 0.6,
        drop_overlap_with: list[str] | None = None,
        corpus_docs: int = 4000,
        sizes: tuple[int, ...] = (34, 38, 40, 44, 48),
        cache_dir: str | Path | None = None,
        work_dir: str | Path | None = None,
        flush_every: int = 1000,
        name: str | None = None,
        limit: int | None = None,
        **_ignored,
    ):
        self.name = name or type(self).name
        self.n = min(n, limit) if limit else n
        self.corpora = corpora or []
        self.texts = texts or []
        self.seed = seed
        self.degrade_level = degrade_level
        self.diacritics = diacritics
        self.min_len, self.max_len, self.min_arabic = min_len, max_len, min_arabic
        self.drop_overlap_with = drop_overlap_with
        self.corpus_docs = corpus_docs
        self.sizes = sizes
        # cache_dir: the durable home (on Colab, a Drive folder). work_dir: a fast
        # local scratch dir to actually render into; when both are set and differ,
        # images live in work_dir and are checkpointed to cache_dir as a single
        # tar (one sequential Drive write beats 15k tiny-file round-trips).
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.work_dir = Path(work_dir) if work_dir else None
        self.flush_every = max(1, int(flush_every))
        self._samples: list[Sample] | None = None
        self._restored = False

    def _raw(self) -> list[str]:
        raw = list(self.texts)
        for c in self.corpora:
            if isinstance(c, (list, tuple)):
                raw.extend(c)
            elif c in _CORPUS_LOADERS:
                raw.extend(_CORPUS_LOADERS[c](self.corpus_docs))
            else:
                raise KeyError(
                    f"unknown corpus {c!r}; known: {sorted(_CORPUS_LOADERS)} "
                    "(or pass a literal list of strings)"
                )
        return raw

    # --- pool: cache the (slow, network) corpus + contamination step ---------
    def _pool_key(self) -> str:
        spec = {
            "texts": self.texts,
            "corpora": [c if isinstance(c, str) else list(c) for c in self.corpora],
            "n": self.n,
            "seed": self.seed,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "min_arabic": self.min_arabic,
            "drop_overlap_with": self.drop_overlap_with,
            "corpus_docs": self.corpus_docs,
        }
        blob = json.dumps(spec, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def _pool(self) -> list[str]:
        cache_file = self.cache_dir / f"pool_{self._pool_key()}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            try:
                pool = json.loads(cache_file.read_text(encoding="utf-8"))
                if pool:
                    return pool
            except (OSError, json.JSONDecodeError):
                pass

        pool = sentences(
            self._raw(),
            n=max(self.n, 1),
            seed=self.seed,
            min_len=self.min_len,
            max_len=self.max_len,
            min_arabic=self.min_arabic,
        )
        if self.drop_overlap_with:
            pool = drop_overlap(pool, scored_text_keys(self.drop_overlap_with))
        if not pool:
            raise ValueError(
                "synthetic corpus is empty. Either every remote corpus in "
                f"{self.corpora!r} failed to load (see the warnings above — check "
                "network / dataset ids), or the length / arabic-ratio / overlap "
                "filters removed everything. Pass `texts=[...]` to supply lines "
                "directly, or widen min_len / min_arabic."
            )
        if cache_file:
            _atomic_write_text(cache_file, json.dumps(pool, ensure_ascii=False))
        return pool

    # --- image dir + tar checkpoint ----------------------------------------
    @property
    def _archive_path(self) -> Path | None:
        """Single-file checkpoint on the durable store, used only when rendering
        happens in a separate (fast, local) work_dir."""
        if self.cache_dir and self.work_dir and self.work_dir != self.cache_dir:
            return self.cache_dir / "images.tar"
        return None

    def _img_dir(self) -> Path | None:
        """Directory the PNGs live in; restores from the tar checkpoint once."""
        if not self.work_dir:
            return self.cache_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        arch = self._archive_path
        if not self._restored and arch and arch.exists():
            with tarfile.open(arch, "r") as tf:
                # filter="data" (3.12+) rejects path traversal / device files;
                # this is our own archive but it's the right default anyway.
                kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                tf.extractall(self.work_dir, **kw)
            print(f"synth: restored {len(list(self.work_dir.glob('*.png')))} "
                  f"cached lines from {arch}")
        self._restored = True
        return self.work_dir

    def _flush_archive(self, img_dir: Path) -> None:
        arch = self._archive_path
        if not arch:
            return
        arch.parent.mkdir(parents=True, exist_ok=True)
        tmp = arch.with_suffix(".tar.tmp")
        with tarfile.open(tmp, "w") as tf:
            for png in sorted(img_dir.glob("*.png")):
                tf.add(png, arcname=png.name)
        os.replace(tmp, arch)

    def _build(self) -> list[Sample]:
        if self._samples is not None:
            return self._samples
        from tqdm.auto import tqdm

        pool = self._pool()
        font_paths = fonts()
        img_dir = self._img_dir()
        out: list[Sample] = []
        rendered_since_flush = 0
        for idx in tqdm(range(self.n), desc="synth render", unit="line"):
            text = pool[idx % len(pool)]
            gt = _strip_diacritics(text, self.diacritics, seed=self.seed, idx=idx)
            sid = f"{self.name}/{self.seed}/{idx:06d}"
            img, fresh = self._render_cached(sid, gt, font_paths, idx, img_dir)
            rendered_since_flush += fresh
            if fresh and rendered_since_flush >= self.flush_every:
                self._flush_archive(img_dir)
                rendered_since_flush = 0
            out.append(
                Sample(
                    id=sid,
                    image=img,
                    ground_truth=gt,
                    track=self.track,
                    meta={"source": "synthetic_printed", "seed": self.seed, "row": idx},
                )
            )
        if rendered_since_flush:
            self._flush_archive(img_dir)
        self._samples = out
        return out

    def _render_cached(self, sid, gt, font_paths, idx, img_dir) -> tuple[Image.Image, int]:
        """Returns (image, 1 if freshly rendered else 0)."""
        if img_dir is None:
            return self._render(gt, font_paths, idx), 0
        fp = img_dir / f"{sid.replace('/', '__')}.png"
        fresh = 0
        if not fp.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
            self._render(gt, font_paths, idx).save(fp)
            fresh = 1
        # Return a lazy handle, not decoded pixels: a full run is 15k+ lines and
        # assemble_dataset keeps them all in a list — see LazyImage.
        return LazyImage(fp), fresh

    def _render(self, gt, font_paths, idx) -> Image.Image:
        prng = random.Random(f"{self.seed}:{idx}")
        font_path = font_paths[prng.randrange(len(font_paths))]
        size = self.sizes[prng.randrange(len(self.sizes))]
        return degrade(
            render_line(gt, font_path, size=size),
            np.random.default_rng([self.seed, idx]),
            self.degrade_level,
        )

    def __len__(self) -> int:
        return len(self._build())

    def __iter__(self) -> Iterator[Sample]:
        yield from self._build()


def _strip_diacritics(text: str, mode: str, *, seed: int, idx: int) -> str:
    if mode == "keep":
        return text
    if mode == "none":
        return _HARAKAT_RE.sub("", text)
    # "mixed": deterministically strip ~half
    return _HARAKAT_RE.sub("", text) if random.Random(f"{seed}:{idx}:d").random() < 0.5 else text
