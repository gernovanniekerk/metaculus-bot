"""Entry point. Based on Metaculus's metac-bot-template main.py.

Modes:
  tournament      Fall 2026 FutureEval tournament + MiniBench (what the
                  scheduled GitHub Action runs). Publishes.
  test_questions  Metaculus's bot-testing-area tournament. Publishes there.
  metaculus_cup   Metaculus Cup. Publishes. Not part of the bot prize pool.
  dry_run         Offline rehearsal: fake LLM, fake search, fake Metaculus,
                  network blocked. Spends nothing, submits nothing.
"""

import argparse
import asyncio
import logging
import sys
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import MetaculusClient  # noqa: E402

from gerno_bot.budget import BudgetGuard, LedgerCorruptError, SpendLedger  # noqa: E402
from gerno_bot.bot import GernoBot  # noqa: E402
from gerno_bot.claude_client import ClaudeClient  # noqa: E402
from gerno_bot.config import Settings  # noqa: E402
from gerno_bot.publishing import CommentSafeClient  # noqa: E402
from gerno_bot.research import AskNewsSource, ExaSource  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

PENDING_COMMENTS_PATH = "state/pending_comments.json"

TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def _tournament_id(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def build_sources(settings: Settings, guard: BudgetGuard) -> list:
    sources: list = []
    if settings.asknews_enabled:
        sources.append(
            AskNewsSource(
                guard,
                monthly_call_limit=settings.asknews_monthly_call_limit,
                historical=settings.asknews_historical,
            )
        )
    if settings.exa_enabled:
        sources.append(
            ExaSource(
                guard,
                num_results=settings.exa_num_results,
                reserve_usd=settings.exa_reserve_usd,
            )
        )
    return sources


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run Gerno's Metaculus forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions", "dry_run"],
        default="tournament",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions", "dry_run"] = args.mode

    if run_mode == "dry_run":
        from gerno_bot.dry_run import run_dry_run

        return run_dry_run()

    check_environment(strict=True)
    settings = Settings.from_env()
    try:
        ledger = SpendLedger(settings.ledger_path)
    except LedgerCorruptError as e:
        print(f"❌  {e}")
        return 1
    guard = BudgetGuard(
        ledger=ledger,
        monthly_cap_usd=settings.monthly_budget_usd,
        per_question_cap_usd=settings.max_usd_per_question,
        pace_headstart=settings.pace_headstart,
    )
    print(
        f"💵  Spend this month: ${guard.spent_this_month():.2f} of "
        f"${settings.monthly_budget_usd:.2f} cap; tier now: {guard.choose_tier()}; "
        f"provider: {settings.llm_provider}"
    )

    claude = ClaudeClient(guard, settings.llm_provider, settings.use_refusal_fallbacks)
    sources = build_sources(settings, guard)
    print(f"🔎  Research sources: {[s.name for s in sources] or 'none configured'}")
    client = CommentSafeClient(MetaculusClient(), PENDING_COMMENTS_PATH)
    still_pending = client.retry_pending()
    if still_pending:
        print(f"⚠️   {still_pending} saved comment(s) still could not be posted")

    bot = GernoBot(
        settings=settings,
        guard=guard,
        claude=claude,
        research_sources=sources,
        publish_reports_to_metaculus=True,
        metaculus_client=client,
        skip_previously_forecasted_questions=run_mode != "test_questions",
    )
    smallest = min(
        bot.minimum_question_cost(settings.full_ensemble),
        bot.minimum_question_cost(settings.economy_ensemble),
    )
    if guard.remaining_this_month() < smallest:
        print(
            f"🛑  Monthly cap reached (${guard.spent_this_month():.2f} spent). "
            "Not forecasting until next month."
        )
        return 0

    print_startup_banner(run_mode, will_publish=True)

    async def run() -> list:
        if run_mode == "tournament":
            reports = await bot.forecast_on_tournament(
                _tournament_id(settings.aib_tournament_id), return_exceptions=True
            )
            reports += await bot.forecast_on_tournament(
                _tournament_id(settings.minibench_tournament_id), return_exceptions=True
            )
            return reports
        if run_mode == "metaculus_cup":
            return await bot.forecast_on_tournament(
                MetaculusClient.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        return await bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)

    forecast_reports = asyncio.run(run())
    print_run_summary_banner(
        forecast_reports, will_publish=True, tournament_url=TOURNAMENT_URLS.get(run_mode)
    )
    print(
        f"💵  Spend this month after this run: ${guard.spent_this_month():.2f} "
        f"of ${settings.monthly_budget_usd:.2f}"
    )
    bot.log_report_summary(forecast_reports)  # raises if any question failed
    return 0


if __name__ == "__main__":
    sys.exit(main())
