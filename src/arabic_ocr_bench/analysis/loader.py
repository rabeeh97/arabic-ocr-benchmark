# src/arabic_ocr_bench/analysis/loader.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PredRow:
    model: str
    dataset: str
    sample_id: str
    gt: str
    pred: str
    error: str | None
    attempts: int
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cached: bool
    scores: dict = field(default_factory=dict)


def run_dirs(results_dir: str | Path) -> list[Path]:
    p = Path(results_dir)
    if (p / "predictions.jsonl").exists():
        return [p]
    return sorted(d.parent for d in p.glob("*/predictions.jsonl"))


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def iter_pred_rows(results_dir: str | Path) -> list[PredRow]:
    merged: dict[tuple[str, str], dict] = {}
    scores: dict[tuple[str, str], dict] = {}
    for rd in run_dirs(results_dir):
        for s in _jsonl(rd / "scores.jsonl"):
            scores.setdefault((s.get("model", ""), s.get("sample_id", "")), s)
        for p in _jsonl(rd / "predictions.jsonl"):
            model, sid = p.get("model", ""), p.get("sample_id", "")
            ds = p.get("dataset", "")
            if not model or model.startswith("mock") or ds == "synthetic":
                continue
            prev = merged.get((model, sid))
            if prev is None or (prev.get("error") and not p.get("error")):
                merged[(model, sid)] = p
    out: list[PredRow] = []
    for (model, sid), p in merged.items():
        out.append(PredRow(
            model=model, dataset=p.get("dataset", ""), sample_id=sid,
            gt=p.get("ground_truth", ""), pred=p.get("cleaned_text") or "",
            error=p.get("error"), attempts=int(p.get("attempts", 1) or 1),
            latency_ms=float(p.get("latency_ms", 0.0) or 0.0),
            cost_usd=float(p.get("cost_usd", 0.0) or 0.0),
            input_tokens=int(p.get("input_tokens", 0) or 0),
            output_tokens=int(p.get("output_tokens", 0) or 0),
            cached=bool(p.get("cached", False)),
            scores=scores.get((model, sid), {}) if not p.get("error") else {},
        ))
    return out
