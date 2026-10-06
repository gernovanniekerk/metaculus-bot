"""Monthly spend cap, enforced in code.

How the cap holds:
- Before any paid call, the caller reserves the most that call could cost
  (all input billed as a cache write, every max_tokens output token used, plus
  a full re-run on the fallback model if refusal fallbacks are on).
- The reservation is refused if spend so far + open reservations + this one
  would pass the month's cap (or the per-question cap).
- After the call, the reservation is replaced by the cost the API reported.
  If the call failed in a way that might still have been billed, the full
  reservation is charged.
- Spend is written to a JSON ledger after every call. In GitHub Actions the
  workflow commits that file back to the repo, so the next run starts from the
  real month-to-date figure.

Months are calendar months in UTC.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

LEDGER_VERSION = 1
MAX_RECENT_QUESTIONS = 300


class BudgetExceededError(RuntimeError):
    """A call was refused because it could take spend past a cap."""


class LedgerCorruptError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def month_key(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m")


def fraction_of_month_elapsed(moment: datetime) -> float:
    moment = moment.astimezone(timezone.utc)
    start = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        next_start = start.replace(year=start.year + 1, month=1)
    else:
        next_start = start.replace(month=start.month + 1)
    return (moment - start).total_seconds() / (next_start - start).total_seconds()


def _empty_month() -> dict:
    return {
        "spend_usd": 0.0,
        "llm_usd": 0.0,
        "search_usd": 0.0,
        "llm_calls": 0,
        "search_calls": 0,
        "asknews_calls": 0,
        "questions_attempted": 0,
    }


class SpendLedger:
    """JSON file holding month-to-date spend. Small, human-readable, committed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": LEDGER_VERSION, "months": {}, "recent_questions": []}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # Fail closed: an unreadable ledger means unknown spend.
            raise LedgerCorruptError(
                f"Spend ledger {self.path} could not be read ({e}). Refusing to "
                "run; fix or restore the file from git history."
            ) from e
        if not isinstance(data, dict) or "months" not in data:
            raise LedgerCorruptError(f"Spend ledger {self.path} has no 'months' key.")
        data.setdefault("recent_questions", [])
        return data

    def month(self, key: str) -> dict:
        months = self.data["months"]
        if key not in months:
            months[key] = _empty_month()
        for k, v in _empty_month().items():
            months[key].setdefault(k, v)
        return months[key]

    def add_question_record(self, record: dict) -> None:
        recent = self.data["recent_questions"]
        recent.append(record)
        del recent[:-MAX_RECENT_QUESTIONS]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".ledger-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(payload)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


@dataclass
class QuestionSpend:
    key: str
    spent: float = 0.0
    reserved: float = 0.0
    calls: int = 0


@dataclass
class Reservation:
    amount: float
    kind: str  # "llm" | "search"
    month: str
    question: QuestionSpend | None
    settled: bool = False


_current_question: contextvars.ContextVar[QuestionSpend | None] = (
    contextvars.ContextVar("current_question_spend", default=None)
)


@dataclass
class BudgetGuard:
    ledger: SpendLedger
    monthly_cap_usd: float
    per_question_cap_usd: float
    pace_headstart: float = 0.25
    clock: Callable[[], datetime] = utc_now
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _reserved: float = 0.0

    # ----------------------------------------------------------------- queries
    def spent_this_month(self) -> float:
        with self._lock:
            return float(self.ledger.month(month_key(self.clock()))["spend_usd"])

    def remaining_this_month(self) -> float:
        with self._lock:
            spent = float(self.ledger.month(month_key(self.clock()))["spend_usd"])
            return self.monthly_cap_usd - spent - self._reserved

    def pace_allowance(self) -> float:
        """Spend the month may have reached by now while still running the full
        ensemble: the elapsed share of the month plus a head start."""
        share = min(1.0, fraction_of_month_elapsed(self.clock()) + self.pace_headstart)
        return self.monthly_cap_usd * share

    def choose_tier(self) -> str:
        """'full' while spending is on pace, 'economy' once it is ahead of pace."""
        return "full" if self.spent_this_month() < self.pace_allowance() else "economy"

    # ------------------------------------------------------------ reservations
    def reserve(self, amount: float, kind: str) -> Reservation:
        if amount < 0:
            raise ValueError("Reservation amount must be non-negative")
        question = _current_question.get()
        with self._lock:
            key = month_key(self.clock())
            month = self.ledger.month(key)
            projected = float(month["spend_usd"]) + self._reserved + amount
            if projected > self.monthly_cap_usd + 1e-9:
                raise BudgetExceededError(
                    f"Monthly cap ${self.monthly_cap_usd:.2f} would be exceeded: "
                    f"spent ${month['spend_usd']:.4f}, reserved ${self._reserved:.4f}, "
                    f"this call up to ${amount:.4f}."
                )
            if question is not None:
                # Runaway guard: a new call may start only while the question's
                # actual spend plus this call's worst case stays under the cap.
                # (Ensemble members already running are not counted, so the
                # monthly cap, which counts every open reservation, is the
                # hard limit.)
                q_projected = question.spent + amount
                if q_projected > self.per_question_cap_usd + 1e-9:
                    raise BudgetExceededError(
                        f"Per-question cap ${self.per_question_cap_usd:.2f} would be "
                        f"exceeded on {question.key}: spent ${question.spent:.4f}, "
                        f"this call up to ${amount:.4f}."
                    )
                question.reserved += amount
            self._reserved += amount
            return Reservation(
                amount=amount, kind=kind, month=key, question=question
            )

    def settle(self, reservation: Reservation, actual_cost: float) -> None:
        """Replace a reservation with the real cost and persist the ledger."""
        if reservation.settled:
            return
        question = reservation.question
        with self._lock:
            reservation.settled = True
            self._reserved = max(0.0, self._reserved - reservation.amount)
            month = self.ledger.month(reservation.month)
            month["spend_usd"] = float(month["spend_usd"]) + actual_cost
            if reservation.kind == "llm":
                month["llm_usd"] = float(month["llm_usd"]) + actual_cost
                month["llm_calls"] = int(month["llm_calls"]) + 1
            else:
                month["search_usd"] = float(month["search_usd"]) + actual_cost
                month["search_calls"] = int(month["search_calls"]) + 1
            if question is not None:
                question.reserved = max(0.0, question.reserved - reservation.amount)
                question.spent += actual_cost
                question.calls += 1
            self.ledger.save()

    def cancel(self, reservation: Reservation) -> None:
        """Release a reservation for a call that was certainly not billed."""
        if reservation.settled:
            return
        question = reservation.question
        with self._lock:
            reservation.settled = True
            self._reserved = max(0.0, self._reserved - reservation.amount)
            if question is not None:
                question.reserved = max(0.0, question.reserved - reservation.amount)

    # ---------------------------------------------------------------- counters
    def asknews_calls_this_month(self) -> int:
        with self._lock:
            return int(self.ledger.month(month_key(self.clock()))["asknews_calls"])

    def record_asknews_calls(self, n: int) -> None:
        with self._lock:
            month = self.ledger.month(month_key(self.clock()))
            month["asknews_calls"] = int(month["asknews_calls"]) + n
            self.ledger.save()

    # ------------------------------------------------------- per-question scope
    @contextlib.contextmanager
    def question_scope(self, key: str, tier: str) -> Iterator[QuestionSpend]:
        spend = QuestionSpend(key=key)
        token = _current_question.set(spend)
        started = self.clock()
        try:
            yield spend
        finally:
            _current_question.reset(token)
            with self._lock:
                month = self.ledger.month(month_key(started))
                month["questions_attempted"] = int(month["questions_attempted"]) + 1
                self.ledger.add_question_record(
                    {
                        "question": key,
                        "started_utc": started.isoformat(timespec="seconds"),
                        "tier": tier,
                        "cost_usd": round(spend.spent, 6),
                        "paid_calls": spend.calls,
                    }
                )
                self.ledger.save()
