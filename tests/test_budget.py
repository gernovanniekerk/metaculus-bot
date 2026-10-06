import json
from datetime import datetime, timezone

import pytest

from gerno_bot.budget import (
    BudgetExceededError,
    BudgetGuard,
    LedgerCorruptError,
    SpendLedger,
    fraction_of_month_elapsed,
)


class Clock:
    def __init__(self, moment: datetime):
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def make_guard(tmp_path, cap=50.0, per_q=1.5, when=datetime(2026, 10, 6, tzinfo=timezone.utc)):
    clock = Clock(when)
    guard = BudgetGuard(
        ledger=SpendLedger(tmp_path / "ledger.json"),
        monthly_cap_usd=cap,
        per_question_cap_usd=per_q,
        clock=clock,
    )
    return guard, clock


def test_reserve_and_settle_persists(tmp_path):
    guard, _ = make_guard(tmp_path)
    r = guard.reserve(1.0, "llm")
    assert guard.remaining_this_month() == pytest.approx(49.0)
    guard.settle(r, 0.25)
    assert guard.spent_this_month() == pytest.approx(0.25)
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["months"]["2026-10"]["spend_usd"] == pytest.approx(0.25)
    assert data["months"]["2026-10"]["llm_calls"] == 1
    # A fresh process sees the same spend.
    guard2, _ = make_guard(tmp_path)
    assert guard2.spent_this_month() == pytest.approx(0.25)


def test_cap_is_hard(tmp_path):
    guard, _ = make_guard(tmp_path, cap=1.0)
    r = guard.reserve(0.6, "llm")
    with pytest.raises(BudgetExceededError):
        guard.reserve(0.5, "llm")  # 0.6 open + 0.5 > 1.0
    guard.settle(r, 0.6)
    with pytest.raises(BudgetExceededError):
        guard.reserve(0.41, "llm")
    guard.reserve(0.4, "llm")  # exactly to the cap is allowed


def test_cancel_releases_without_charge(tmp_path):
    guard, _ = make_guard(tmp_path, cap=1.0)
    r = guard.reserve(0.9, "llm")
    guard.cancel(r)
    assert guard.spent_this_month() == 0
    guard.reserve(0.9, "llm")


def test_new_month_starts_fresh(tmp_path):
    guard, clock = make_guard(tmp_path, cap=1.0)
    guard.settle(guard.reserve(1.0, "llm"), 1.0)
    with pytest.raises(BudgetExceededError):
        guard.reserve(0.01, "llm")
    clock.moment = datetime(2026, 11, 1, 0, 5, tzinfo=timezone.utc)
    guard.reserve(0.5, "llm")
    assert guard.spent_this_month() == 0


def test_per_question_cap(tmp_path):
    guard, _ = make_guard(tmp_path, per_q=1.0)
    with guard.question_scope("q1", "full") as q:
        guard.settle(guard.reserve(0.7, "llm"), 0.7)
        with pytest.raises(BudgetExceededError):
            guard.reserve(0.4, "llm")
        assert q.spent == pytest.approx(0.7)
    # Outside the question scope the per-question cap does not apply.
    guard.reserve(0.4, "llm")
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["recent_questions"][-1]["cost_usd"] == pytest.approx(0.7)
    assert data["months"]["2026-10"]["questions_attempted"] == 1


def test_pacing_tiers(tmp_path):
    # 6 Oct: about 16% of the month elapsed; with a 25% head start the full
    # ensemble runs until spend reaches about 41% of the cap.
    guard, clock = make_guard(tmp_path, cap=50.0)
    assert guard.choose_tier() == "full"
    guard.settle(guard.reserve(19.0, "llm"), 19.0)
    assert guard.choose_tier() == "full"
    guard.settle(guard.reserve(2.0, "llm"), 2.0)
    assert guard.choose_tier() == "economy"
    clock.moment = datetime(2026, 10, 25, tzinfo=timezone.utc)
    assert guard.choose_tier() == "full"


def test_fraction_of_month():
    assert fraction_of_month_elapsed(datetime(2026, 12, 1, tzinfo=timezone.utc)) == 0
    assert fraction_of_month_elapsed(datetime(2026, 12, 16, 12, tzinfo=timezone.utc)) == pytest.approx(0.5)


def test_corrupt_ledger_fails_closed(tmp_path):
    (tmp_path / "ledger.json").write_text("{not json")
    with pytest.raises(LedgerCorruptError):
        SpendLedger(tmp_path / "ledger.json")


def test_asknews_counter(tmp_path):
    guard, _ = make_guard(tmp_path)
    guard.record_asknews_calls(1)
    guard.record_asknews_calls(5)
    assert guard.asknews_calls_this_month() == 6
