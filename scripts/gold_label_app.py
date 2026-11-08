#!/usr/bin/env python
"""Start the local gold-set labeling tool.

    PYTHONPATH=src python scripts/gold_label_app.py --tranche gold/tranches/t01 [--port 8770]

Then open http://127.0.0.1:8770/ and enter your labeler ID. Labels: gold/labels/<labeler>/t01.jsonl.
"""
from arabic_ocr_bench.gold.labeler_app import main

if __name__ == "__main__":
    main()
