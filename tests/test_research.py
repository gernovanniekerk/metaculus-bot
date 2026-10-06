import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gerno_bot.budget import BudgetExceededError, BudgetGuard, SpendLedger
from gerno_bot.fakes import FakeExaHttp, FakeSource
from gerno_bot.research import AskNewsSource, ExaSource, run_research


def make_guard(tmp_path, cap=50.0):
    return BudgetGuard(
        ledger=SpendLedger(tmp_path / "ledger.json"),
        monthly_cap_usd=cap,
        per_question_cap_usd=1.5,
        clock=lambda: datetime(2026, 10, 6, tzinfo=timezone.utc),
    )


def test_exa_charges_reported_cost(tmp_path):
    guard = make_guard(tmp_path)
    http = FakeExaHttp(cost=0.011)
    text = asyncio.run(ExaSource(guard, 6, 0.02, http_post=http).search("Will X happen?"))
    assert "Background article" in text
    assert guard.spent_this_month() == pytest.approx(0.011)
    assert http.calls[0]["json"]["numResults"] == 6


def test_exa_http_error_not_charged(tmp_path):
    guard = make_guard(tmp_path)

    def post(*a, **k):
        return SimpleNamespace(status_code=401, json=lambda: {})

    with pytest.raises(RuntimeError):
        asyncio.run(ExaSource(guard, 6, 0.02, http_post=post).search("q"))
    assert guard.spent_this_month() == 0


def test_exa_missing_cost_charges_reservation(tmp_path):
    guard = make_guard(tmp_path)

    def post(*a, **k):
        return SimpleNamespace(status_code=200, json=lambda: {"results": []})

    asyncio.run(ExaSource(guard, 6, 0.02, http_post=post).search("q"))
    assert guard.spent_this_month() == pytest.approx(0.02)


def test_exa_respects_budget(tmp_path):
    guard = make_guard(tmp_path, cap=0.01)
    http = FakeExaHttp()
    with pytest.raises(BudgetExceededError):
        asyncio.run(ExaSource(guard, 6, 0.02, http_post=http).search("q"))
    assert http.calls == []


def test_one_failing_source_does_not_sink_research():
    class Broken:
        name = "Broken"

        async def search(self, q):
            raise RuntimeError("down")

    result = asyncio.run(run_research("Q?", [Broken(), FakeSource("AskNews", "news")], 10_000))
    assert result.sources_used == ["AskNews"]
    assert "## AskNews" in result.text and "Broken" not in result.text


def test_research_is_truncated_per_source():
    result = asyncio.run(run_research("Q?", [FakeSource("A", "x" * 50_000)], 6_000))
    assert len(result.text) < 7_000


def test_asknews_stops_at_monthly_limit(tmp_path):
    guard = make_guard(tmp_path)
    guard.record_asknews_calls(900)
    # Returns before touching the SDK (the network is blocked in tests anyway).
    assert asyncio.run(AskNewsSource(guard, 900, historical=False).search("q")) == ""
