"""Tiny CRNN for CPU CTC. Height-preserving-enough conv, width stride 4."""

from __future__ import annotations

import torch
from torch import nn

from arabic_ocr_bench.charctc.preprocess import WIDTH_STRIDE


class CharCRNN(nn.Module):
    """CNN + 2-layer BiLSTM + linear CTC head.

    Input: ``(B, 1, 32, W)`` with W divisible by 4.
    Output logits: ``(T, B, n_classes)`` with T = W // 4.
    """

    def __init__(self, n_classes: int, hidden: int = 128) -> None:
        super().__init__()
        if n_classes < 3:
            raise ValueError("n_classes must include blank, unk, and at least one char")
        self.n_classes = n_classes
        self.hidden = hidden
        self.cnn = nn.Sequential(
            _conv_block(1, 64, pool=(2, 2)),       # H 32->16, W /2
            _conv_block(64, 128, pool=(2, 2)),     # H 16->8,  W /4
            _conv_block(128, 128, pool=(2, 1)),    # H 8->4,   W /4
            _conv_block(128, 256, pool=(2, 1)),    # H 4->2,   W /4
            _conv_block(256, 256, pool=(2, 1)),    # H 2->1,   W /4
        )
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, 1, H, W
        feats = self.cnn(x)                 # B, 256, 1, T
        b, c, h, t = feats.shape
        if h != 1:
            feats = feats.mean(dim=2, keepdim=True)
            h = 1
        seq = feats.view(b, c * h, t).permute(2, 0, 1)  # T, B, C
        seq, _ = self.rnn(seq)
        return self.fc(seq)                 # T, B, n_classes


def _conv_block(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool, stride=pool),
    )


def time_steps_for(width: int) -> int:
    return width // WIDTH_STRIDE
