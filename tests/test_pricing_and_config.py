from types import SimpleNamespace

import pytest

from gerno_bot.config import Settings, parse_ensemble
from gerno_bot.pricing import (
    UnknownModelPriceError,
    cost_from_usage,
    normalise_model_id,
    price_for,
    worst_case_cost,
)


def test_opus_55_cost_with_cache():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=2000,
        cache_creation_input_tokens=10_000,
        cache_read_input_tokens=0,
        iterations=None,
    )
    # 1000*4 + 10000*5 + 2000*20 per million
    assert cost_from_usage(usage, "claude-opus-5-5") == pytest.approx(0.094)


def test_cache_read_is_cheap():
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 1_000_000}
    assert cost_from_usage(usage, "claude-opus-5-5") == pytest.approx(0.20)


def test_iterations_override_top_level_and_price_fallback_conservatively():
    usage = SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        iterations=[
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000,
             "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            {"type": "fallback_message", "model": None, "input_tokens": 1000,
             "output_tokens": 1000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ],
    )
    # Opus 5.5 input 1000*4 ; fallback priced as Opus 5: 1000*5 + 1000*25
    assert cost_from_usage(usage, "claude-opus-5-5") == pytest.approx((4000 + 5000 + 25000) / 1e6)


def test_unknown_model_refused():
    with pytest.raises(UnknownModelPriceError):
        price_for("claude-imaginary-9")


def test_price_override(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES_JSON", '{"claude-new-6": {"input": 1, "output": 2, "cache_read": 0.1}}')
    assert price_for("claude-new-6").output_per_mtok == 2


def test_openrouter_ids_normalise():
    assert normalise_model_id("anthropic/claude-opus-5.5") == "claude-opus-5-5"
    assert normalise_model_id("anthropic/claude-sonnet-5.5:batch") == "claude-sonnet-5-5"


def test_worst_case_counts_fallback():
    without = worst_case_cost("claude-opus-5-5", 10_000, 10_000, with_fallback=False)
    with_fb = worst_case_cost("claude-opus-5-5", 10_000, 10_000, with_fallback=True)
    assert without == pytest.approx((10_000 * 5 + 10_000 * 20) / 1e6)
    assert with_fb > without


def test_parse_ensemble():
    members = parse_ensemble("claude-opus-5-5@medium*3, claude-sonnet-5-5@low")
    assert [m.label for m in members] == ["claude-opus-5-5@medium"] * 3 + ["claude-sonnet-5-5@low"]
    assert [m.index for m in members] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        parse_ensemble("claude-opus-5-5@turbo")


def test_settings_defaults(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.monthly_budget_usd == 50.0
    assert s.full_ensemble[0].model == "claude-opus-5-5"
    assert len(s.full_ensemble) == 5
    assert s.llm_provider == "none"


def test_provider_auto(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x-test")
    s = Settings.from_env()
    assert s.llm_provider == "openrouter"
    assert s.use_refusal_fallbacks is False  # not offered through OpenRouter
