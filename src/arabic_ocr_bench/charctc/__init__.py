"""Character-level CTC line recogniser — first tokenizer/output-space experiment.

A small CNN-BiLSTM-CTC trained on a closed character vocabulary built from
our own ground truth. Deliberately not a VLM and not a subword decoder: the
point is to measure whether a glyph-aligned output space beats English-BPE
TrOCR (~80% CER on Muharaf here) and how it compares to the 8k-line Qwen LoRA
(~30–36% CER).
"""

from .charset import BLANK, UNK, Charset
from .decode import ctc_greedy, ids_to_text

__all__ = ["BLANK", "UNK", "Charset", "ctc_greedy", "ids_to_text"]
