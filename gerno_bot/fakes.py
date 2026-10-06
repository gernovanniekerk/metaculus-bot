"""Offline stand-ins for the Anthropic SDK, research sources and the Metaculus
client. Used by the tests and by `python main.py --mode dry_run`. Nothing in
here touches the network.
"""

from __future__ import annotations

import contextlib
import json
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "metaculus_posts.json"


class NetworkBlockedError(RuntimeError):
    pass


@contextlib.contextmanager
def block_network() -> Iterator[None]:
    """Make any attempt to open a network connection raise."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError(f"Network access attempted during offline run: {args[:2]}")

    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.create_connection,
            socket.getaddrinfo,
        ) = saved  # type: ignore[assignment]


def load_fixture_questions(path: Path = FIXTURE_PATH) -> list[MetaculusQuestion]:
    with path.open("r", encoding="utf-8") as f:
        posts = json.load(f)
    questions = [DataOrganizer.get_question_from_post_json(p) for p in posts]
    for q in questions:
        q.already_forecasted = False
    return questions


# --------------------------------------------------------------------- answers
def scripted_answer(question: MetaculusQuestion, variant: int) -> str:
    """A plausible write-up ending in the answer format the prompt asks for."""
    rationale = (
        "Time left: several weeks. Status quo: no change. Base rate from similar "
        "past events is moderate; recent news adds little. Main risk: an "
        "unexpected announcement before the deadline.\n\n"
    )
    shift = (variant % 5) * 0.03
    if isinstance(question, BinaryQuestion):
        return rationale + f"Probability: {round(30 + shift * 100)}%"
    if isinstance(question, MultipleChoiceQuestion):
        n = len(question.options)
        first = 0.4 + shift
        rest = (1 - first) / (n - 1)
        lines = [f"{question.options[0]}: {first * 100:.1f}%"]
        lines += [f"{o}: {rest * 100:.1f}%" for o in question.options[1:]]
        return rationale + "\n".join(lines)
    fractions = [0.15, 0.25, 0.4, 0.5, 0.65, 0.8]
    if isinstance(question, DateQuestion):
        lo, hi = question.lower_bound, question.upper_bound
        span = (hi - lo).total_seconds()
        dates = [lo + timedelta(seconds=span * min(0.95, f + shift)) for f in fractions]
        return rationale + "\n".join(
            f"Percentile {p}: {d.date().isoformat()}" for p, d in zip((10, 20, 40, 60, 80, 90), dates)
        )
    if isinstance(question, NumericQuestion):
        lo, hi = question.lower_bound, question.upper_bound
        values = [lo + (hi - lo) * min(0.95, f + shift) for f in fractions]
        return rationale + "\n".join(
            f"Percentile {p}: {v:.4f}" for p, v in zip((10, 20, 40, 60, 80, 90), values)
        )
    raise ValueError(f"No scripted answer for {type(question).__name__}")


# ------------------------------------------------------------- Anthropic SDK
def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class FakeAnthropicSDK:
    """Mimics AsyncAnthropic.messages.create / beta.messages.create.

    responder(request, call_number) -> text. Usage numbers imitate prompt
    caching: the first request with a given system+shared prefix is billed as a
    cache write, later ones as cache reads.
    """

    responder: Callable[[dict, int], str]
    requests: list[dict] = field(default_factory=list)
    usages: list[Any] = field(default_factory=list)
    _seen_prefixes: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        text = self.responder(request, len(self.requests))
        system = "".join(b["text"] for b in request.get("system", []))
        content = request["messages"][0]["content"]
        shared = content[0]["text"]
        tail = "".join(b["text"] for b in content[1:])
        prefix_key = (request["model"], system, shared)
        prefix_tokens = _tokens(system + shared)
        if prefix_key in self._seen_prefixes:
            cache_write, cache_read = 0, prefix_tokens
        else:
            self._seen_prefixes.add(prefix_key)
            cache_write, cache_read = prefix_tokens, 0
        thinking_tokens = 1500 if "thinking" in request else 0
        usage = SimpleNamespace(
            input_tokens=_tokens(tail) if tail else 0,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
            output_tokens=min(request["max_tokens"], _tokens(text) + thinking_tokens),
            iterations=None,
        )
        self.usages.append(usage)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=usage,
            stop_reason="end_turn",
            model=request["model"],
        )


def question_matching(request: dict, questions: list[MetaculusQuestion]) -> MetaculusQuestion | None:
    shared = request["messages"][0]["content"][0]["text"]
    for q in questions:
        if q.question_text in shared:
            return q
    return None


# ------------------------------------------------------------------ research
@dataclass
class FakeSource:
    name: str
    text: str
    calls: int = 0

    async def search(self, query: str) -> str:
        self.calls += 1
        return f"{self.text} (query: {query[:80]})"


class FakeExaHttp:
    """Stands in for requests.post against the Exa API."""

    def __init__(self, cost: float = 0.011) -> None:
        self.cost = cost
        self.calls: list[dict] = []

    def __call__(self, url: str, json: dict, headers: dict, timeout: int) -> Any:
        self.calls.append({"url": url, "json": json})
        payload = {
            "results": [
                {
                    "title": "Background article",
                    "url": "https://example.org/background",
                    "publishedDate": "2026-09-30T00:00:00Z",
                    "highlights": ["Relevant background text for the question."],
                }
            ],
            "costDollars": {"total": self.cost},
        }
        return SimpleNamespace(status_code=200, json=lambda: payload)


# ------------------------------------------------------------ Metaculus API
@dataclass
class FakeMetaculusClient:
    questions: list[MetaculusQuestion] = field(default_factory=list)
    predictions: list[tuple[str, int, Any]] = field(default_factory=list)
    comments: list[tuple[int, str, bool]] = field(default_factory=list)
    fail_comments: bool = False

    def get_all_open_questions_from_tournament(self, tournament_id: Any) -> list[MetaculusQuestion]:
        return list(self.questions)

    def post_binary_question_prediction(self, question_id: int, prediction_in_decimal: float) -> None:
        self.predictions.append(("binary", question_id, prediction_in_decimal))

    def post_numeric_question_prediction(self, question_id: int, cdf_values: list[float]) -> None:
        self.predictions.append(("numeric", question_id, cdf_values))

    def post_multiple_choice_question_prediction(
        self, question_id: int, options_with_probabilities: dict[str, float]
    ) -> None:
        self.predictions.append(("multiple_choice", question_id, options_with_probabilities))

    def post_question_comment(
        self, post_id: int, comment_text: str, is_private: bool = True, included_forecast: bool = True
    ) -> None:
        if self.fail_comments:
            raise RuntimeError("simulated comment failure")
        self.comments.append((post_id, comment_text, is_private))
