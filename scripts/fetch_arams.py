"""Download + extract AraMS-28k-HTR (Zenodo) into a local `<id>.png` + `<id>.gt.txt` directory.

    python scripts/fetch_arams.py --dest data/arams_28k_htr

NOT run automatically by anything — the box operator runs this once before a
training suite that includes `arams_htr`/`arams_train` as a data source (see
`configs/datasets.yaml`'s `arams_htr` entry and
`src/arabic_ocr_bench/datasets/arams.py`). Stdlib-only (urllib + zipfile),
matching every other launcher/fetch script in this repo — no extra deps to
install before the download can even start. Idempotent: re-running skips
files already present and correctly sized.

Zenodo record: 10.5281/zenodo.21499649 ("AraMS-28k-HTR"), ~2.42 GB, real
pre-segmented historical-manuscript line crops + `.gt.txt` transcriptions.
Queries the Zenodo REST API for the record's file list rather than
hardcoding a download URL, so a version bump on Zenodo's side doesn't need a
code change here.

IMPORTANT — verify after first run: this script downloads and extracts
whatever archive(s) the record lists, but the *internal* folder layout
inside the archive (e.g. images and `.gt.txt` files nested under a
`lines/` subdirectory, or a flat directory) is not knowable without actually
fetching it. After the first real run, check that `--dest` ends up
containing `<id>.png` next to a matching `<id>.gt.txt` directly (what
`arams.py`'s adapter expects) — if the archive nests them one level deeper,
rerun with `--strip-components 1` (or adjust the constant below) rather than
patching the adapter to search subdirectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "21499649"  # the numeric suffix of 10.5281/zenodo.21499649
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
_UA = "Mozilla/5.0 (arabic-ocr-bench data fetcher)"  # avoid CDN 403s, same fix as uv's installer


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path, *, expected_size: int | None = None) -> None:
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"have  {dest.name}")
        return
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    print(f"fetch {dest.name} ...")
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    print(f"got   {dest.name}  ({dest.stat().st_size // (1 << 20)} MB)")


def _verify_checksum(path: Path, checksum: str | None) -> None:
    if not checksum or ":" not in checksum:
        return
    algo, expected = checksum.split(":", 1)
    if algo not in hashlib.algorithms_available:
        print(f"  (skip checksum: unsupported algo {algo!r})")
        return
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise SystemExit(f"checksum mismatch for {path.name}: expected {expected}, got {got}")
    print(f"  checksum ok ({algo})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data/arams_28k_htr", help="extraction target directory")
    ap.add_argument(
        "--strip-components",
        type=int,
        default=0,
        help="drop N leading path components when extracting (see module docstring)",
    )
    ap.add_argument("--skip-checksum", action="store_true", help="skip md5 verification")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    download_dir = dest / ".downloads"
    download_dir.mkdir(exist_ok=True)

    record = _get_json(ZENODO_API)
    files = record.get("files") or []
    if not files:
        print("No files listed on the Zenodo record — check ZENODO_RECORD_ID.")
        return 1

    for f in files:
        name = f["key"]
        url = f["links"]["self"]
        size = f.get("size")
        checksum = f.get("checksum")
        archive_path = download_dir / name
        _download(url, archive_path, expected_size=size)
        if not args.skip_checksum:
            _verify_checksum(archive_path, checksum)

        if zipfile.is_zipfile(archive_path):
            print(f"extract {name} -> {dest}")
            with zipfile.ZipFile(archive_path) as zf:
                for member in zf.namelist():
                    parts = Path(member).parts[args.strip_components :]
                    if not parts:
                        continue
                    out_path = dest.joinpath(*parts)
                    if member.endswith("/"):
                        out_path.mkdir(parents=True, exist_ok=True)
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, out_path.open("wb") as out:
                        out.write(src.read())
        else:
            print(f"  (not a zip — left as-is under {download_dir}, inspect manually)")

    n_png = len(list(dest.glob("*.png")))
    n_gt = len(list(dest.glob("*.gt.txt")))
    print(f"\n{dest}: {n_png} .png / {n_gt} .gt.txt files at the top level")
    if n_png == 0:
        print(
            "No .png files found directly under --dest — the archive likely nests them "
            "one or more directories deeper. Inspect the extracted tree and re-run with "
            "--strip-components N, per the module docstring."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
