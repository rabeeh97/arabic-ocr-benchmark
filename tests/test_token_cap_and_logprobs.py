"""X6 (output-token caps) and vlm-logprobs in the benchmark runner. No network,
no model weights: LiteLLM and HF generate() are faked."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from arabic_ocr_bench import runner as runner_mod
from arabic_ocr_bench.characterize import token_cap
from arabic_ocr_bench.datasets.base import Sample
from arabic_ocr_bench.models.base import Prediction
from arabic_ocr_bench.models.hf_local_client import HFLocalClient, finish_reason
from arabic_ocr_bench.models.litellm_client import LiteLLMClient
from arabic_ocr_bench.models.registry import build_model, output_token_cap
from arabic_ocr_bench.runner import _cache_key, _key_variant, run


def _img():
    return Image.new("RGB", (20, 10), "white")


# --------------------------------------------------------------------------- #
# LiteLLM client
# --------------------------------------------------------------------------- #
def _resp(content, finish="stop", logprobs=None, out_tok=5):
    choice = SimpleNamespace(message=SimpleNamespace(content=content),
                             finish_reason=finish, logprobs=logprobs)
    return SimpleNamespace(choices=[choice],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=out_tok))


def test_litellm_reports_finish_reason_length(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: _resp("نص مقطوع", "length", out_tok=256))
    pred = LiteLLMClient("m", "openai/m", max_tokens=256).transcribe(_img(), "p")
    assert pred.error is None
    assert pred.finish_reason == "length"
    assert pred.token_logprobs is None


def test_litellm_logprobs_only_requested_when_enabled(monkeypatch):
    import litellm

    calls = []
    lp = SimpleNamespace(content=[SimpleNamespace(token="a", logprob=-0.1),
                                  {"token": "b", "logprob": -2.0}])

    def fake(**kw):
        calls.append(kw)
        return _resp("ab", "stop", logprobs=lp)

    monkeypatch.setattr(litellm, "completion", fake)
    off = LiteLLMClient("m", "openai/m").transcribe(_img(), "p")
    on = LiteLLMClient("m", "openai/m", logprobs=True).transcribe(_img(), "p")
    assert "logprobs" not in calls[0] and calls[1]["logprobs"] is True
    assert off.token_logprobs is None
    assert on.token_logprobs == [-0.1, -2.0] and on.logprob_kind == "api"


def test_litellm_missing_logprobs_in_response_is_none(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: _resp("ab"))
    pred = LiteLLMClient("m", "openai/m", logprobs=True).transcribe(_img(), "p")
    assert pred.token_logprobs is None and pred.logprob_kind is None


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_output_token_cap_matches_build_model_defaults():
    assert output_token_cap({"client": "litellm"}) == 256
    assert output_token_cap({"client": "litellm", "max_tokens": 1024}) == 1024
    assert output_token_cap({"client": "hf_local", "hf_id": "x"}) == 160
    assert output_token_cap({"client": "paddleocr"}) is None
    assert output_token_cap({"client": "mock"}) is None


def test_registry_passes_logprobs_through():
    c = build_model("m", {"client": "litellm", "litellm_id": "openai/m", "logprobs": True})
    h = build_model("h", {"client": "hf_local", "hf_id": "x", "logprobs": True})
    assert c.logprobs is True and h.logprobs is True
    assert build_model("h", {"client": "hf_local", "hf_id": "x"}).logprobs is False


# --------------------------------------------------------------------------- #
# HF local: finish reason and log-probabilities from generate()
# --------------------------------------------------------------------------- #
def test_finish_reason_rules():
    assert finish_reason([5, 6, 2], 160, {2}) == "stop"
    assert finish_reason([5] * 160, 160, {2}) == "length"
    assert finish_reason([5, 6], 160, {2}) == "stop"      # another stopping criterion
    assert finish_reason([5, 6], 160, set()) is None       # EOS unknown, under the cap


def test_gen_out_without_logprobs_needs_no_torch():
    c = HFLocalClient(name="h", hf_id="x", max_new_tokens=3)
    c._model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 9]))
    out = SimpleNamespace(sequences=[[1, 1, 7, 8, 4]], logits=None, scores=None)
    g = c._gen_out(out, 2, "txt")
    assert (g.n_tokens, g.finish_reason, g.token_logprobs) == (3, "length", None)
    g = c._gen_out(SimpleNamespace(sequences=[[1, 1, 7, 9]]), 2, "txt")
    assert (g.n_tokens, g.finish_reason) == (2, "stop")


class _FakeInputs(dict):
    def to(self, _device):
        return self


class _FakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "chat"

    def __call__(self, text, images, return_tensors):
        import torch

        return _FakeInputs(input_ids=torch.tensor([[11, 12, 13]]))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids if int(i) != 2)


class _FakeModel:
    """generate() emits tokens 4, 5, EOS(2); step logits put 0.9 / 0.5 / 0.8
    on them (after softmax) so the expected log-probabilities are known."""

    generation_config = SimpleNamespace(eos_token_id=2)

    def __init__(self):
        self.kwargs = None

    def generate(self, **kw):
        import torch

        self.kwargs = kw
        toks = [4, 5, 2]
        logits = []
        for t, p in zip(toks, (0.9, 0.5, 0.8)):
            probs = torch.full((1, 8), (1 - p) / 7)
            probs[0, t] = p
            logits.append(torch.log(probs))
        seq = torch.tensor([[11, 12, 13, *toks]])
        return SimpleNamespace(sequences=seq,
                               logits=tuple(logits) if kw.get("output_logits") else None,
                               scores=None)


def _hf_client(logprobs):
    pytest.importorskip("torch")
    c = HFLocalClient(name="h", hf_id="x", logprobs=logprobs)
    c._model, c._processor, c._device = _FakeModel(), _FakeProcessor(), "cpu"
    return c


def test_hf_transcribe_records_tokens_finish_and_logprobs():
    import math

    c = _hf_client(logprobs=True)
    pred = c.transcribe(_img(), "p")
    assert pred.error is None and pred.cleaned_text == "4 5"
    assert pred.output_tokens == 3 and pred.finish_reason == "stop"
    assert pred.logprob_kind == "raw_logits"
    assert pred.token_logprobs == pytest.approx([math.log(0.9), math.log(0.5), math.log(0.8)], abs=1e-5)
    assert c._model.kwargs["output_logits"] is True
    assert c._model.kwargs["return_dict_in_generate"] is True


def test_hf_transcribe_without_logprobs_does_not_ask_for_logits():
    c = _hf_client(logprobs=False)
    pred = c.transcribe(_img(), "p")
    assert pred.token_logprobs is None and pred.output_tokens == 3
    assert "output_logits" not in c._model.kwargs


# --------------------------------------------------------------------------- #
# runner: cache keys and row fields
# --------------------------------------------------------------------------- #
def test_cache_key_unchanged_for_existing_configs():
    old = hashlib.sha1(f"s|m|line-v2|r{runner_mod.CACHE_REV}".encode()).hexdigest()
    assert _key_variant({"client": "litellm", "max_tokens": 256}) == ""
    assert _cache_key("s", "m", "line-v2", "") == old
    assert _cache_key("s", "m", "line-v2") == old


def test_cache_key_changes_with_cache_tag_or_logprobs():
    base = _cache_key("s", "m", "line-v2", _key_variant({}))
    tagged = _cache_key("s", "m", "line-v2", _key_variant({"cache_tag": "x6"}))
    lp = _cache_key("s", "m", "line-v2", _key_variant({"logprobs": True}))
    both = _cache_key("s", "m", "line-v2", _key_variant({"cache_tag": "x6", "logprobs": True}))
    assert len({base, tagged, lp, both}) == 4


class _Dataset:
    name = "fake"
    track = "line"

    def __iter__(self):
        img = Image.new("RGB", (8, 8))
        yield Sample(id="fake/000", image=img, ground_truth="مرحبا بكم", track="line")


class _CappedClient:
    """Stands in for any API/VLM client: records whether logprobs were asked for."""

    def __init__(self, name, cfg):
        self.name = name
        self.logprobs = bool(cfg.get("logprobs"))
        self.calls = 0

    def transcribe(self, image, prompt, max_tokens=None):
        self.calls += 1
        return Prediction(raw_text="مرحبا", cleaned_text="مرحبا", output_tokens=256,
                          finish_reason="length",
                          token_logprobs=[-0.5, -1.5] if self.logprobs else None,
                          logprob_kind="api" if self.logprobs else None)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    built = {}

    def _build(name, cfg):
        built[name] = _CappedClient(name, cfg)
        return built[name]

    monkeypatch.setattr(runner_mod, "load_dataset_configs", lambda *a, **k: {"fake": {}})
    monkeypatch.setattr(runner_mod, "build_dataset", lambda *a, **k: _Dataset())
    monkeypatch.setattr(runner_mod, "build_model", _build)
    monkeypatch.setattr(runner_mod, "load_model_configs", lambda *a, **k: {
        "api": {"client": "litellm", "litellm_id": "x", "max_tokens": 256},
        "mock_empty": {"client": "mock", "mode": "empty"},
    })
    return tmp_path, built


def _rows(run_dir):
    return [json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().splitlines()]


def test_row_flags_token_cap(wired):
    root, _ = wired
    (row,) = _rows(run({"models": ["api"], "datasets": ["fake"], "prompt_version": "line-v2",
                        "progress": False}, results_root=root))
    assert row["token_cap"] == 256
    assert row["finish_reason"] == "length" and row["hit_token_cap"] is True
    assert "token_logprobs" not in row


def test_logprobs_run_flag_reaches_client_rows_and_cache_key(wired):
    root, built = wired
    cfg = {"models": ["api"], "datasets": ["fake"], "prompt_version": "line-v2", "progress": False}
    run(cfg, results_root=root)                       # plain run fills the cache
    assert built["api"].calls == 1
    (row,) = _rows(run({**cfg, "logprobs": True}, results_root=root))
    # a cached reading without log-probabilities must not be served to this run
    assert built["api"].calls == 1 and built["api"].logprobs is True
    assert row["cached"] is False
    assert row["token_logprobs"] == [-0.5, -1.5]
    assert row["mean_logprob"] == -1.0 and row["logprob_kind"] == "api"
    (again,) = _rows(run({**cfg, "logprobs": True}, results_root=root))
    assert again["cached"] is True and again["token_logprobs"] == [-0.5, -1.5]


def test_mock_rows_have_no_cap_fields(wired, monkeypatch):
    root, _ = wired
    monkeypatch.setattr(runner_mod, "build_model", build_model)  # the real registry
    (row,) = _rows(run({"models": ["mock_empty"], "datasets": ["fake"],
                        "prompt_version": "line-v1", "progress": False}, results_root=root))
    assert not {"token_cap", "finish_reason", "hit_token_cap", "token_logprobs"} & set(row)


# --------------------------------------------------------------------------- #
# stored-row report (scripts/token_cap_report.py)
# --------------------------------------------------------------------------- #
def _write_run(d, rows, override=None):
    d.mkdir(parents=True)
    (d / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    if override is not None:
        (d / "run_config.snapshot.yaml").write_text(f"max_tokens_override: {override}\n")


def test_token_cap_scan(tmp_path):
    gt = "كلمة " * 20
    cfgs = {"g": {"client": "litellm", "max_tokens": 256},
            "h": {"client": "hf_local", "hf_id": "x"}}
    _write_run(tmp_path / "results" / "r1", [
        {"model": "g", "dataset": "d", "ground_truth": gt, "cleaned_text": gt[:20], "output_tokens": 252},
        {"model": "g", "dataset": "d", "ground_truth": gt, "cleaned_text": gt * 3, "output_tokens": 256},
        {"model": "g", "dataset": "d", "ground_truth": gt, "cleaned_text": gt, "output_tokens": 40},
        {"model": "g", "dataset": "d", "ground_truth": gt, "cleaned_text": "", "error": "429"},
        {"model": "h", "dataset": "d", "ground_truth": gt, "cleaned_text": gt, "output_tokens": 0},
        {"model": "h", "dataset": "d", "ground_truth": gt, "cleaned_text": gt[:5],
         "output_tokens": 160, "hit_token_cap": True, "token_cap": 160},
    ])
    # a max_tokens_override run: 300 output tokens under a 640 cap is not at the cap
    _write_run(tmp_path / "results" / "r2", [
        {"model": "g", "dataset": "d", "ground_truth": gt, "cleaned_text": gt, "output_tokens": 300},
    ], override=640)
    _write_run(tmp_path / "results" / "smoke" / "r3", [
        {"model": "g", "dataset": "d", "ground_truth": gt, "cleaned_text": "", "output_tokens": 256},
    ])
    pairs = token_cap.scan(tmp_path, cfgs)
    g = pairs[("g", "d")]
    assert (g["n"], g["at_cap"], g["truncated"], g["runaway"]) == (4, 2, 1, 1)
    assert g["caps"] == [256, 640]
    h = pairs[("h", "d")]
    assert (h["n"], h["unknown"], h["at_cap"], h["truncated"]) == (2, 1, 1, 1)
    models = token_cap.by_model(pairs)
    assert models["g"]["share_at_cap"] == 0.5
    assert token_cap.flagged(models) == ["g", "h"]
