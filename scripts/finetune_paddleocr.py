"""Fill in PaddleOCR recognition-finetune YAML paths. Does not train.

The actual ``tools/train.py`` invocation needs the GPU, and must wait until
the suite_v1 VLM run has released it. This script only:

  * clones PaddleOCR (optional, ``--clone``) so ``tools/train.py`` exists
  * rewrites ``configs/training/paddleocr/arabic_PP-OCRv5_mobile_rec.finetune.yaml``
    placeholders into ``<out>/rec_finetune.yaml``
  * prints the exact train command

    python scripts/finetune_paddleocr.py \\
        --labels C:\\Users\\User\\a-experiment\\paddleocr\\labels\\suite_v1 \\
        --out C:\\Users\\User\\a-experiment\\paddleocr\\finetune

Pass ``--run`` only when the 3090 is free. Refuses if another python process
already holds the GPU.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"
TEMPLATE = REPO / "configs" / "training" / "paddleocr" / "arabic_PP-OCRv5_mobile_rec.finetune.yaml"
DEFAULT_PRETRAINED = Path.home() / ".paddlex" / "official_models" / "arabic_PP-OCRv5_mobile_rec"
DEFAULT_CLONE = Path(r"C:\Users\User\a-experiment\PaddleOCR") if IS_WIN else Path.home() / "a-experiment" / "PaddleOCR"


def _gpu_busy() -> str | None:
    """Return a reason string if another process looks like it owns the GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    if out.returncode != 0:
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    mine = str(os.getpid())
    others = [ln for ln in lines if not ln.startswith(mine + ",")]
    if others:
        return "GPU already has compute apps:\n  " + "\n  ".join(others)
    return None


def write_filled_yaml(*, labels: Path, out: Path, pretrained: Path, clone: Path) -> Path:
    text = TEMPLATE.read_text(encoding="utf-8")
    dict_path = clone / "ppocr" / "utils" / "dict" / "ppocrv5_arabic_dict.txt"
    replacements = {
        "PLACEHOLDER_SAVE_DIR": str(out / "output").replace("\\", "/"),
        "PLACEHOLDER_PRETRAINED": str(pretrained).replace("\\", "/"),
        "PLACEHOLDER_DICT": str(dict_path).replace("\\", "/"),
        "PLACEHOLDER_DATA_DIR": str(labels).replace("\\", "/"),
        "PLACEHOLDER_TRAIN_LIST": str(labels / "train_list.txt").replace("\\", "/"),
        "PLACEHOLDER_VAL_LIST": str(labels / "val_list.txt").replace("\\", "/"),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "rec_finetune.yaml"
    dest.write_text(text, encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="directory from prep_paddleocr_labels.py")
    ap.add_argument("--out", required=True, help="finetune working directory")
    ap.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    ap.add_argument("--clone", default=str(DEFAULT_CLONE), help="PaddleOCR source checkout")
    ap.add_argument("--clone-repo", action="store_true", help="git clone PaddleOCR into --clone if missing")
    ap.add_argument("--run", action="store_true", help="actually invoke tools/train.py (GPU)")
    ap.add_argument("--venv", default=None, help="paddleocr_venv python")
    args = ap.parse_args(argv)

    labels = Path(args.labels).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    clone = Path(args.clone).expanduser()
    pretrained = Path(args.pretrained).expanduser()

    if args.clone_repo and not (clone / "tools" / "train.py").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", "https://github.com/PaddlePaddle/PaddleOCR.git", str(clone)]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    yaml_path = write_filled_yaml(labels=labels, out=out, pretrained=pretrained, clone=clone)
    py = Path(args.venv) if args.venv else Path(sys.executable)
    train = [
        str(py),
        str(clone / "tools" / "train.py"),
        "-c",
        str(yaml_path),
    ]
    print(f"wrote {yaml_path}")
    print("train command:")
    print(" ", " ".join(train))

    if not args.run:
        print("not launching (pass --run when the GPU is free)")
        return 0

    busy = _gpu_busy()
    if busy:
        print(busy, file=sys.stderr)
        print("refusing --run so we don't fight the occupant. retry when nvidia-smi is empty.", file=sys.stderr)
        return 2
    if not (clone / "tools" / "train.py").exists():
        print(f"missing {clone / 'tools' / 'train.py'} — pass --clone-repo", file=sys.stderr)
        return 1
    if not (labels / "train_list.txt").exists():
        print(f"missing {labels / 'train_list.txt'} — run prep_paddleocr_labels.py first", file=sys.stderr)
        return 1
    return subprocess.call(train, cwd=str(clone))


if __name__ == "__main__":
    raise SystemExit(main())
