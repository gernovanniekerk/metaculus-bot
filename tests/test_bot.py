"""End-to-end tests of the bot on public Metaculus questions (saved fixtures),
with fake Claude, fake search and a fake Metaculus client."""

import asyncio
import statistics
from datetime import datetime, timezone

import pytest
from forecasting_tools import BinaryQuestion, ForecastReport

from gerno_bot.budget import month_key
from gerno_bot.dry_run import build_offline_bot, run_dry_run


def run(bot, tournament="test"):
    return asyncio.run(bot.forecast_on_tournament(tournament, return_exceptions=True))


def test_every_forecast_gets_a_private_comment(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path)
    reports = run(bot)
    assert len(reports) == 5
    assert all(isinstance(r, ForecastReport) for r in reports), reports
    forecast_ids = {qid for _, qid, _ in fake.predictions}
    assert len(fake.predictions) == 5
    assert len(fake.comments) == 5
    for r in reports:
        assert r.question.id_of_question in forecast_ids
    for post_id, text, is_private in fake.comments:
        assert is_private is True
        assert text.startswith("# SUMMARY")
        assert "*Method*:" in text and "no human in the loop" in text
        assert text.count("Reasoning") >= 5  # every member's rationale is in the comment
    assert {p for p, _, _ in fake.comments} == {r.question.id_of_post for r in reports}


def test_binary_aggregate_is_median_of_members(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path, malformed_first_answer=False)
    reports = run(bot)
    binary = next(r for r in reports if isinstance(r.question, BinaryQuestion))
    comment = next(t for p, t, _ in fake.comments if p == binary.question.id_of_post)
    member_values = [
        float(line.split(":")[1].strip().rstrip("%")) / 100
        for line in comment.splitlines()
        if line.startswith("*Forecaster ")
    ]
    assert len(member_values) == 5
    assert binary.prediction == pytest.approx(statistics.median(member_values))


def test_first_member_runs_alone_then_rest_read_cache(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path, malformed_first_answer=False)
    run(bot)
    # Per question: 3 Opus + 2 Sonnet calls; 2 Opus + 1 Sonnet read the cache.
    hits = [u.cache_read_input_tokens > 0 for u in sdk.usages]
    assert sum(hits) == 15
    assert hits[0] is False  # the very first call writes the cache


def test_missing_answer_triggers_one_cheap_repair(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path, malformed_first_answer=True)
    reports = run(bot)
    assert all(isinstance(r, ForecastReport) for r in reports)
    assert sum(1 for r in sdk.requests if r["model"] == "claude-haiku-4-5") == 1
    assert any("restated by claude-haiku-4-5" in t for _, t, _ in fake.comments)


def test_budget_exhausted_posts_nothing(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path, monthly_budget_usd=0.10)
    reports = run(bot)
    assert all(isinstance(r, BaseException) for r in reports)
    assert any("left this month" in str(r) for r in reports)
    assert fake.predictions == [] and fake.comments == []
    assert sdk.requests == []
    assert guard.spent_this_month() <= 0.10


def test_spend_never_exceeds_cap_mid_run(tmp_path):
    # Enough for roughly one question: later questions must be refused, and
    # spend must stay under the cap even though calls run concurrently.
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path, monthly_budget_usd=2.0)
    reports = run(bot)
    assert guard.spent_this_month() <= 2.0
    assert any(isinstance(r, ForecastReport) for r in reports)
    assert len(fake.comments) == len(fake.predictions)


def test_economy_tier_when_ahead_of_pace(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path, malformed_first_answer=False)
    # 6 Oct with $45 of $50 already spent: well ahead of pace.
    guard.clock = lambda: datetime(2026, 10, 6, tzinfo=timezone.utc)
    guard.ledger.month(month_key(guard.clock()))["spend_usd"] = 45.0
    assert guard.choose_tier() == "economy"
    run(bot)
    models = {(r["model"], r["output_config"]["effort"]) for r in sdk.requests}
    assert models == {("claude-sonnet-5-5", "low")}


def test_skips_already_forecast_questions(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path)
    for q in fake.questions:
        q.already_forecasted = True
    assert run(bot) == []
    assert sdk.requests == []


def test_failed_comment_is_saved_and_retried(tmp_path):
    bot, sdk, fake, guard, client = build_offline_bot(tmp_path, malformed_first_answer=False)
    fake.questions = fake.questions[:1]
    fake.fail_comments = True
    reports = run(bot)
    assert isinstance(reports[0], BaseException)
    assert len(fake.predictions) == 1 and fake.comments == []
    assert client.pending_count() == 1
    fake.fail_comments = False
    assert client.retry_pending() == 0
    assert len(fake.comments) == 1


def test_dry_run_passes():
    assert run_dry_run() == 0
