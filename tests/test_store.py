"""Unit checks for scripts/store_build.py (run: .geomvenv/bin/python -m pytest tests/test_store.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from store_build import reversed_suspect, split_tag, vlm_family  # noqa: E402


def test_split_tag():
    assert split_tag("kitab_khatt") == ("kitab_khatt", "none")
    assert split_tag("kitab_khatt__inkcrop") == ("kitab_khatt", "inkcrop")
    assert split_tag("kitab_khatt__tta") == ("kitab_khatt", "tta_conf")
    assert split_tag("kitab_khatt__inkcrop__tta") == ("kitab_khatt", "inkcrop+tta_conf")
    assert split_tag("nakba_lines__as_muharaf_only_slant") == ("nakba_lines", "as_muharaf_only_slant")


def test_reversed_suspect():
    gt = "كثافة الهواء بها قليلة"
    assert reversed_suspect(gt, gt[::-1], native=len(gt))
    assert not reversed_suspect(gt, gt, native=0)
    assert not reversed_suspect(gt, "كثافة الهوا بها قليله", native=2)


def test_vlm_family():
    assert vlm_family("qwen3-vl-4b-bf16") == "vlm-base"
    assert vlm_family("suite-v1-ain7b-ft-bf16") == "vlm-ft"
    assert vlm_family("exp3090-qwen3vl4b-ft-bf16") == "vlm-ft"
    assert vlm_family("arabic-paddleocr-v5") == "ocr-engine"
    assert vlm_family("gemini-flash") == "api"
