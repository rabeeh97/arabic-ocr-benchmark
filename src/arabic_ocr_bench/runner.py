"""Execute a benchmark run: (sample x model) -> predictions.jsonl.

Crash-safe (rows are flushed as they complete), resumable (a persistent cache
keyed by sample+model+prompt is consulted first), and bounded-concurrent.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from .datasets.registry import build_dataset, load_dataset_configs
from .models.mock_client import build_mock_prompt
from .models.registry import build_model, is_mock, load_model_configs
from .prompting import get_prompt

_TRANSIENT = ("429", "rate", "timeout", "500", "502", "503", "504", "overloaded")
_RATE_LIMITED = ("429", "rate limit", "ratelimit", "rate_limit", "quota",
                 "too many requests", "resource_exhausted", "insufficient_quota")
# Consecutive rate-limit errors that mean "this (free) model is out of quota" —
# stop hammering it and move on (its rows stay un-scored / uncached for a later run).
_RL_ABORT = 8


def _is_rate_limited(err: str | None) -> bool:
    return bool(err) and any(t in err.lower() for t in _RATE_LIMITED)


class _RateLimiter:
    """Thread-safe sliding-window limiter: at most `rpm` acquisitions per 60 s.
    Used to stay under a provider's requests-per-minute cap on a big crawl."""

    def __init__(self, rpm: float):
        self.rpm = float(rpm)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0]) + 0.01
            time.sleep(max(wait, 0.0))

# Bump when generation behaviour changes (decoding params, max_new_tokens,
# image caps, output cleaning, refusal handling) so old cached predictions miss.
CACHE_REV = "3"


def _cache_key(sample_id: str, model: str, prompt_version: str) -> str:
    return hashlib.sha1(
        f"{sample_id}|{model}|{prompt_version}|r{CACHE_REV}".encode()
    ).hexdigest()


def _cost_usd(cfg: dict, in_tok: int, out_tok: int) -> float:
    pricing = cfg.get("pricing") or {}
    return (
        in_tok / 1e6 * pricing.get("input_usd_per_1m", 0.0)
        + out_tok / 1e6 * pricing.get("output_usd_per_1m", 0.0)
    )


def _is_transient(err: str | None) -> bool:
    return bool(err) and any(t in err.lower() for t in _TRANSIENT)


_IDX_RE = re.compile(r"(\d+)\s*$")


def scored_indices(
    root: Path, model: str, dataset: str, *, exclude: Path | None = None
) -> set[int]:
    """Absolute row indices this (model, dataset) already has a NON-errored
    prediction for, across every run dir under ``root`` (except ``exclude`` —
    the in-progress run whose predictions file is mid-write). Used by accumulate
    mode so a re-run never re-spends a call that already succeeded."""
    done: set[int] = set()
    skip = exclude.resolve() if exclude else None
    for pred in root.glob("*/predictions.jsonl"):
        if skip and pred.parent.resolve() == skip:
            continue
        try:
            lines = pred.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("model") != model or r.get("dataset") != dataset or r.get("error"):
                continue
            m = _IDX_RE.search(str(r.get("sample_id", "")).split("/")[-1])
            if m:
                done.add(int(m.group(1)))
    return done


def next_indices(done: set[int], n: int) -> list[int]:
    """The next ``n`` absolute indices (from 0 up) that are not in ``done`` —
    naturally retries earlier gaps / errored rows first, then moves into fresh
    territory. The dataset adapter clamps anything past the end."""
    if n <= 0:
        return []
    return [i for i in range(len(done) + n) if i not in done][:n]


def run(run_cfg: dict, *, results_root: str | Path | None = None) -> Path:
    models_cfg = load_model_configs()
    datasets_cfg = load_dataset_configs()

    prompt_version = run_cfg.get("prompt_version", "line-v1")
    base_prompt = get_prompt(prompt_version)
    limit = run_cfg.get("limit")
    offset = int(run_cfg.get("offset", 0))
    accumulate = bool(run_cfg.get("accumulate", False))
    concurrency = int(run_cfg.get("concurrency", 4))
    max_retries = int(run_cfg.get("max_retries", 3))
    use_cache = bool(run_cfg.get("cache", True))
    show_progress = bool(run_cfg.get("progress", True))
    limiter = _RateLimiter(float(run_cfg.get("rpm", 0) or 0))  # <=0 => no limit
    rl_abort = int(run_cfg.get("rate_limit_abort", _RL_ABORT))  # 0 => never abort a model
    runner_id = str(run_cfg.get("runner_id", "") or "")
    session_id = str(run_cfg.get("session_id", "") or "")

    # A distributed runner claims an exact index range and passes it here, so
    # two Colab accounts never recompute the same rows. `indices` needs a single
    # (model, dataset) pair — it overrides accumulate's own next-indices pick.
    explicit_indices = run_cfg.get("indices")
    if explicit_indices is not None:
        if len(run_cfg["models"]) != 1 or len(run_cfg["datasets"]) != 1:
            raise ValueError("run_cfg['indices'] requires exactly one model and one dataset")
        explicit_indices = sorted(int(i) for i in explicit_indices)
        accumulate = True

    root = Path(results_root or run_cfg.get("results_dir", "results"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id, run_dir, bump = stamp, root / stamp, 1
    while run_dir.exists():  # two runs in the same second must not share a dir
        bump += 1
        run_id = f"{stamp}-{bump}"
        run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.snapshot.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    pred_path = run_dir / "predictions.jsonl"

    clients = {name: build_model(name, models_cfg[name]) for name in run_cfg["models"]}

    def _one(model_name: str, ds_name: str, sample) -> dict:
        cfg = models_cfg[model_name]
        key = _cache_key(sample.id, model_name, prompt_version)
        cache_file = cache_dir / f"{key}.json"
        if use_cache and cache_file.exists():
            row = json.loads(cache_file.read_text())
            if not row.get("error"):  # never serve a stale errored result — retry it
                row["cached"] = True
                # keep the original runner's attribution; record who re-served it
                row["served_to"] = runner_id or row.get("served_to", "")
                return row

        prompt = base_prompt
        if is_mock(cfg):
            prompt = build_mock_prompt(base_prompt, sample.ground_truth, cfg.get("mode", "empty"))

        client = clients[model_name]
        attempts = 0
        while True:
            attempts += 1
            if not is_mock(cfg):
                limiter.acquire()   # stay under the provider's requests/min cap
            t0 = time.perf_counter()
            pred = client.transcribe(sample.image, prompt)
            if pred.latency_ms == 0.0:
                pred.latency_ms = (time.perf_counter() - t0) * 1000
            if not (_is_transient(pred.error) and attempts <= max_retries):
                break
            time.sleep(min(2 ** attempts, 30))

        row = {
            "sample_id": sample.id,
            "model": model_name,
            "dataset": ds_name,
            "track": sample.track,
            "prompt_version": prompt_version,
            "ground_truth": sample.ground_truth,
            "raw_text": pred.raw_text,
            "cleaned_text": pred.cleaned_text,
            "latency_ms": round(pred.latency_ms, 2),
            "input_tokens": pred.input_tokens,
            "output_tokens": pred.output_tokens,
            "cost_usd": round(_cost_usd(cfg, pred.input_tokens, pred.output_tokens), 8),
            "error": pred.error,
            "attempts": attempts,
            "cached": False,
        }
        if runner_id:
            row["runner"] = runner_id
        if session_id:
            row["session"] = session_id
        if use_cache and pred.error is None:
            cache_file.write_text(json.dumps(row, ensure_ascii=False))
        return row

    def _chunks(it, n):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf:
            yield buf

    n_models = len(run_cfg["models"])

    # Model-major: finish every job for one model, then free it before loading
    # the next (critical for a local-model sweep on a single GPU). Samples are
    # STREAMED from a fresh adapter per model and processed in small chunks, so
    # we never hold the whole decoded dataset in RAM.
    quota_out: set[str] = set()  # models that hit their rate limit — skip their rest

    with pred_path.open("w", encoding="utf-8") as fh:
        for mi, model_name in enumerate(run_cfg["models"], 1):
            for ds_name in run_cfg["datasets"]:
                if model_name in quota_out:
                    if show_progress:
                        print(f"[{mi}/{n_models}] skip {model_name} · {ds_name} "
                              f"(rate-limited — out of quota)")
                    continue
                if accumulate:
                    if explicit_indices is not None:
                        want = explicit_indices
                    else:
                        done = scored_indices(root, model_name, ds_name, exclude=run_dir)
                        want = next_indices(done, int(limit) if limit else 10)
                    adapter = build_dataset(ds_name, datasets_cfg[ds_name], indices=want)
                else:
                    adapter = build_dataset(
                        ds_name, datasets_cfg[ds_name], limit=limit, offset=offset
                    )
                try:
                    total = len(adapter)
                except (TypeError, AttributeError):
                    total = None
                bar = tqdm(
                    total=total, unit="line", disable=not show_progress,
                    desc=f"[{mi}/{n_models}] {model_name} · {ds_name}",
                )
                seen = {"cached": 0, "err": 0, "rl_streak": 0}

                def _emit(row, bar=bar, seen=seen):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    seen["cached"] += bool(row.get("cached"))
                    seen["err"] += bool(row.get("error"))
                    if _is_rate_limited(row.get("error")) and not row.get("cached"):
                        seen["rl_streak"] += 1
                    elif not row.get("cached"):
                        seen["rl_streak"] = 0
                    bar.update(1)
                    bar.set_postfix(cached=seen["cached"], err=seen["err"], refresh=False)

                def _bail() -> bool:
                    return rl_abort > 0 and seen["rl_streak"] >= rl_abort

                if concurrency <= 1:
                    for sample in adapter:
                        _emit(_one(model_name, ds_name, sample))
                        if _bail():
                            break
                else:
                    for chunk in _chunks(adapter, max(4, concurrency * 4)):
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            rows = list(pool.map(
                                lambda s, m=model_name, d=ds_name: _one(m, d, s), chunk
                            ))
                        for row in rows:
                            _emit(row)
                        if _bail():
                            break
                bar.close()
                if _bail():
                    quota_out.add(model_name)
                    if show_progress:
                        print(f"[{mi}/{n_models}] {model_name}: {rl_abort}+ consecutive "
                              f"rate-limit errors on {ds_name} — treating as out of daily "
                              f"quota, skipping its remaining datasets this run")
            close = getattr(clients[model_name], "close", None)
            if callable(close):
                close()
            # Drop the client itself so the next model loads onto a clean GPU
            # even if close() left a reference cycle behind.
            clients[model_name] = None
            import gc

            gc.collect()

    return run_dir
