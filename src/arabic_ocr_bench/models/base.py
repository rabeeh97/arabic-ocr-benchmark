"""The uniform model-client contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass
class Prediction:
    """Result of one transcription call.

    raw_text:     the model's response, verbatim (never cleaned)
    cleaned_text: raw_text with fences / preambles / quotes stripped; what scoring uses
    latency_ms:   wall-clock time for the call
    input_tokens / output_tokens: as reported by the provider (0 if unknown)
    cost_usd:     computed by the runner from configs/models.yaml pricing (0 if free)
    error:        None on success; otherwise a short tag ("refusal", "timeout", "http_500", ...)
    finish_reason: why generation stopped, when the client knows: "stop" (end of sequence)
                  or "length" (hit the output-token cap, i.e. the reading may be truncated)
    token_logprobs: per generated token log-probability (natural log), only when the client
                  was asked for them (run_cfg / model cfg ``logprobs: true``); else None
    logprob_kind: where token_logprobs came from: "raw_logits" (model distribution),
                  "processed_scores" (after repetition penalties etc.) or "api"
    """

    raw_text: str = ""
    cleaned_text: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    finish_reason: str | None = None
    token_logprobs: list[float] | None = None
    logprob_kind: str | None = None


@runtime_checkable
class ModelClient(Protocol):
    name: str

    def transcribe(self, image: Image.Image, prompt: str,
                    max_tokens: int | None = None) -> Prediction: ...
