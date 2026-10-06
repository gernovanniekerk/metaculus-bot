"""Offline end-to-end run: `python main.py --mode dry_run`.

Runs the real bot code (research, ensemble, budget guard, parsing, aggregation,
comment building, publishing path) on five public Metaculus questions saved in
tests/fixtures, with:
- a scripted fake in place of the Anthropic API (no LLM calls, no cost),
- fake AskNews and Exa responses (no searches, no cost),
- an in-memory fake in place of Metaculus (nothing is submitted),
- every network connection blocked at the socket level, so any accidental
  real call fails loudly instead of going out,
- a throwaway spend ledger, so the real one is untouched.

The costs it prints are what the ledger WOULD have recorded at list prices
for the fake token counts. They show the metering works; they are not a
forecast of real spend.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import tempfile
from pathlib import Path

from gerno_bot.budget import BudgetGuard, SpendLedger
from gerno_bot.bot import GernoBot
from gerno_bot.claude_client import ClaudeClient
from gerno_bot.config import Settings
from gerno_bot.fakes import (
    FakeAnthropicSDK,
    FakeExaHttp,
    FakeMetaculusClient,
    FakeSource,
    block_network,
    load_fixture_questions,
    question_matching,
    scripted_answer,
)
from gerno_bot.publishing import CommentSafeClient
from gerno_bot.research import ExaSource

logger = logging.getLogger(__name__)


def build_offline_bot(
    workdir: Path,
    monthly_budget_usd: float = 50.0,
    malformed_first_answer: bool = True,
) -> tuple[GernoBot, FakeAnthropicSDK, FakeMetaculusClient, BudgetGuard, CommentSafeClient]:
    questions = load_fixture_questions()
    settings = dataclasses.replace(
        Settings.from_env(),
        llm_provider="anthropic",
        use_refusal_fallbacks=True,
        monthly_budget_usd=monthly_budget_usd,
        ledger_path=str(workdir / "spend_ledger.json"),
    )
    guard = BudgetGuard(
        ledger=SpendLedger(settings.ledger_path),
        monthly_cap_usd=settings.monthly_budget_usd,
        per_question_cap_usd=settings.max_usd_per_question,
        pace_headstart=settings.pace_headstart,
    )

    def responder(request: dict, call_number: int) -> str:
        question = question_matching(request, questions)
        if question is None:
            # Repair call: the write-up quoted in the prompt names its post.
            text = request["messages"][0]["content"][0]["text"]
            q = next((q for q in questions if f"[post {q.id_of_post}]" in text), questions[0])
            return scripted_answer(q, 0).split("\n\n", 1)[1]  # answer lines only
        if malformed_first_answer and call_number == 1:
            # Exercise the repair path once: no final answer line.
            return f"[post {question.id_of_post}] I think this is fairly unlikely but I'm not certain."
        return scripted_answer(question, call_number)

    sdk = FakeAnthropicSDK(responder=responder)
    claude = ClaudeClient(guard, "anthropic", use_refusal_fallbacks=True, sdk_client=sdk)
    sources = [
        FakeSource("AskNews", "### Latest news\n- Example headline (2026-10-05)."),
        ExaSource(
            guard,
            num_results=settings.exa_num_results,
            reserve_usd=settings.exa_reserve_usd,
            http_post=FakeExaHttp(),
        ),
    ]
    fake_metaculus = FakeMetaculusClient(questions=questions)
    client = CommentSafeClient(fake_metaculus, workdir / "pending_comments.json")
    bot = GernoBot(
        settings=settings,
        guard=guard,
        claude=claude,
        research_sources=sources,
        publish_reports_to_metaculus=True,  # goes to the in-memory fake
        metaculus_client=client,
    )
    return bot, sdk, fake_metaculus, guard, client


def run_dry_run() -> int:
    # The fake Exa source reads EXA_API_KEY for its header; use a dummy.
    os.environ.setdefault("EXA_API_KEY", "dry-run-dummy")
    with tempfile.TemporaryDirectory(prefix="metaculus-dry-run-") as tmp, block_network():
        workdir = Path(tmp)
        bot, sdk, fake_metaculus, guard, _ = build_offline_bot(workdir)
        reports = asyncio.run(bot.forecast_on_tournament("dry-run", return_exceptions=True))

        print("\n" + "=" * 78)
        print("DRY RUN (offline: fake LLM, fake search, fake Metaculus, network blocked)")
        print("=" * 78)
        ok = True
        commented = {post_id for post_id, _, _ in fake_metaculus.comments}
        for r in reports:
            if isinstance(r, BaseException):
                ok = False
                print(f"FAILED: {type(r).__name__}: {str(r)[:300]}")
                continue
            q = r.question
            has_comment = q.id_of_post in commented
            ok = ok and has_comment
            print(
                f"- {type(q).__name__:<22} post {q.id_of_post}: "
                f"{r.make_readable_prediction(r.prediction).strip()[:70].replace(chr(10), ' ')} | "
                f"comment posted: {'yes' if has_comment else 'NO'} (private), "
                f"{len(r.explanation)} chars"
            )
        cache_hits = sum(1 for u in sdk.usages if u.cache_read_input_tokens > 0)
        print(f"\nFake LLM requests: {len(sdk.requests)} "
              f"(models: {sorted({req['model'] for req in sdk.requests})}); "
              f"{cache_hits} read the shared prompt from cache")
        print(f"Forecasts posted to fake Metaculus: {len(fake_metaculus.predictions)}; "
              f"comments: {len(fake_metaculus.comments)}")
        print(f"Metered spend at list prices for the fake token counts: "
              f"${guard.spent_this_month():.4f} of ${guard.monthly_cap_usd:.2f} cap")
        print("Network connections attempted: none (socket layer blocked)")
        print("RESULT:", "PASS" if ok and len(reports) > 0 else "FAIL")
        return 0 if ok and len(reports) > 0 else 1
