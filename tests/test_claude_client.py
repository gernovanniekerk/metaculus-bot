import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from gerno_bot.budget import BudgetExceededError, BudgetGuard, SpendLedger
from gerno_bot.claude_client import ClaudeClient, LlmRefusalError
from gerno_bot.fakes import FakeAnthropicSDK


def make_guard(tmp_path, cap=50.0):
    return BudgetGuard(
        ledger=SpendLedger(tmp_path / "ledger.json"),
        monthly_cap_usd=cap,
        per_question_cap_usd=1.5,
        clock=lambda: datetime(2026, 10, 6, tzinfo=timezone.utc),
    )


def call(client, model="claude-opus-5-5", effort="medium", shared="Question block " * 50):
    return asyncio.run(
        client.complete(
            model=model,
            effort=effort,
            system_text="System prompt " * 200,
            shared_text=shared,
            max_tokens=10_000,
            purpose="test",
        )
    )


def test_request_shape_caching_thinking_and_fallbacks(tmp_path):
    sdk = FakeAnthropicSDK(responder=lambda req, n: "Probability: 40%")
    client = ClaudeClient(make_guard(tmp_path), "anthropic", use_refusal_fallbacks=True, sdk_client=sdk)
    call(client)
    req = sdk.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in req and "budget_tokens" not in str(req)


def test_second_identical_prefix_reads_cache(tmp_path):
    sdk = FakeAnthropicSDK(responder=lambda req, n: "Probability: 40%")
    guard = make_guard(tmp_path)
    client = ClaudeClient(guard, "anthropic", use_refusal_fallbacks=False, sdk_client=sdk)
    first = call(client)
    second = call(client)
    assert sdk.usages[0].cache_creation_input_tokens > 0
    assert sdk.usages[1].cache_read_input_tokens > 0
    assert second.cost_usd < first.cost_usd
    assert guard.spent_this_month() == pytest.approx(first.cost_usd + second.cost_usd)


def test_haiku_gets_no_thinking_or_effort(tmp_path):
    sdk = FakeAnthropicSDK(responder=lambda req, n: "Probability: 40%")
    client = ClaudeClient(make_guard(tmp_path), "anthropic", use_refusal_fallbacks=True, sdk_client=sdk)
    call(client, model="claude-haiku-4-5", effort=None)
    req = sdk.requests[0]
    assert "thinking" not in req and "output_config" not in req and "fallbacks" not in req


def test_openrouter_model_ids_and_no_fallbacks(tmp_path):
    sdk = FakeAnthropicSDK(responder=lambda req, n: "ok")
    client = ClaudeClient(make_guard(tmp_path), "openrouter", use_refusal_fallbacks=True, sdk_client=sdk)
    call(client, model="claude-opus-5-5")
    call(client, model="claude-sonnet-5-5")
    assert [r["model"] for r in sdk.requests] == ["anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5"]
    assert all("fallbacks" not in r for r in sdk.requests)


def test_budget_refusal_means_no_api_call(tmp_path):
    sdk = FakeAnthropicSDK(responder=lambda req, n: "Probability: 40%")
    client = ClaudeClient(make_guard(tmp_path, cap=0.05), "anthropic", use_refusal_fallbacks=False, sdk_client=sdk)
    with pytest.raises(BudgetExceededError):
        call(client)
    assert sdk.requests == []


def test_unpriced_model_refused_before_call(tmp_path):
    sdk = FakeAnthropicSDK(responder=lambda req, n: "x")
    client = ClaudeClient(make_guard(tmp_path), "anthropic", use_refusal_fallbacks=False, sdk_client=sdk)
    with pytest.raises(ValueError):
        call(client, model="claude-imaginary-9")
    assert sdk.requests == []


class ErrorSDK:
    def __init__(self, exc):
        self.exc = exc

        async def create(**kw):
            raise self.exc

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def _request():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def test_bad_request_is_not_charged(tmp_path):
    guard = make_guard(tmp_path)
    exc = anthropic.BadRequestError(
        "bad", response=httpx2.Response(400, request=_request()), body=None
    )
    client = ClaudeClient(guard, "anthropic", use_refusal_fallbacks=False, sdk_client=ErrorSDK(exc))
    with pytest.raises(anthropic.BadRequestError):
        call(client)
    assert guard.spent_this_month() == 0
    assert guard.remaining_this_month() == pytest.approx(50.0)


def test_timeout_is_charged_worst_case(tmp_path):
    guard = make_guard(tmp_path)
    client = ClaudeClient(
        guard, "anthropic", use_refusal_fallbacks=False,
        sdk_client=ErrorSDK(anthropic.APITimeoutError(request=_request())),
    )
    with pytest.raises(anthropic.APITimeoutError):
        call(client)
    assert guard.spent_this_month() > 0.2  # 10k output tokens of Opus at least
    assert guard.remaining_this_month() == pytest.approx(50.0 - guard.spent_this_month())


def test_refusal_is_charged_and_raised(tmp_path):
    class RefusingSDK(FakeAnthropicSDK):
        async def _create(self, **request):
            resp = await super()._create(**request)
            resp.stop_reason = "refusal"
            resp.content = []
            return resp

    guard = make_guard(tmp_path)
    sdk = RefusingSDK(responder=lambda req, n: "")
    client = ClaudeClient(guard, "anthropic", use_refusal_fallbacks=True, sdk_client=sdk)
    with pytest.raises(LlmRefusalError):
        call(client)
    assert guard.spent_this_month() > 0


def test_rejected_fallback_beta_downgrades_instead_of_failing(tmp_path):
    guard = make_guard(tmp_path)
    sdk = FakeAnthropicSDK(responder=lambda req, n: "Probability: 40%")

    async def reject(**kw):
        raise anthropic.BadRequestError(
            "fallbacks: beta not enabled for this organization",
            response=httpx2.Response(400, request=_request()),
            body=None,
        )

    sdk.beta = SimpleNamespace(messages=SimpleNamespace(create=reject))
    client = ClaudeClient(guard, "anthropic", use_refusal_fallbacks=True, sdk_client=sdk)
    result = call(client)
    assert result.text == "Probability: 40%"
    assert client.use_refusal_fallbacks is False
    assert len(sdk.requests) == 1 and "fallbacks" not in sdk.requests[0]
    assert guard.spent_this_month() == pytest.approx(result.cost_usd)
