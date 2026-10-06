"""Research sources: AskNews (news) and Exa (web search).

- AskNews: free for registered tournament bot makers (1k calls a month, 4k a
  season; "latest news" = 1 call, archive search = 5 calls). The bot counts
  calls in the ledger and stops using AskNews at ASKNEWS_MONTHLY_CALL_LIMIT.
  The free tier allows about one call per 10 seconds, so calls are spaced.
- Exa: paid per search (half a cent per search plus a tenth of a cent per page
  of content, per Metaculus's resources page). Every search is reserved against
  the monthly cap like an LLM call and settled with Exa's reported cost.

A source that fails or is not configured is skipped; the bot still forecasts
with whatever research it has.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import requests

from gerno_bot.budget import BudgetGuard

logger = logging.getLogger(__name__)

EXA_URL = "https://api.exa.ai/search"
ASKNEWS_MIN_SECONDS_BETWEEN_CALLS = 12.0


@dataclass
class ResearchResult:
    text: str
    sources_used: list[str]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[...truncated]"


class AskNewsSource:
    name = "AskNews"

    def __init__(self, guard: BudgetGuard, monthly_call_limit: int, historical: bool) -> None:
        self.guard = guard
        self.monthly_call_limit = monthly_call_limit
        self.historical = historical
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    def calls_needed(self) -> int:
        return 1 + (5 if self.historical else 0)

    async def _spaced(self) -> None:
        wait = ASKNEWS_MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()

    async def search(self, query: str) -> str:
        if self.guard.asknews_calls_this_month() + self.calls_needed() > self.monthly_call_limit:
            logger.warning("AskNews monthly call limit reached; skipping AskNews")
            return ""
        from asknews_sdk import AsyncAskNewsSDK

        client_id = os.getenv("ASKNEWS_CLIENT_ID") or None
        secret = os.getenv("ASKNEWS_SECRET") or None
        api_key = None if (client_id and secret) else (os.getenv("ASKNEWS_API_KEY") or None)
        sections: list[str] = []
        async with self._lock:
            async with AsyncAskNewsSDK(
                client_id=client_id,
                client_secret=secret,
                api_key=api_key,
                scopes={"news"},
            ) as ask:
                await self._spaced()
                self.guard.record_asknews_calls(1)
                latest = await ask.news.search_news(
                    query=query,
                    n_articles=8,
                    return_type="dicts",
                    strategy="latest news",
                )
                sections.append(self._format(latest.as_dicts or [], "Latest news (past 48 hours)"))
                if self.historical:
                    await self._spaced()
                    self.guard.record_asknews_calls(5)
                    older = await ask.news.search_news(
                        query=query,
                        n_articles=8,
                        return_type="dicts",
                        strategy="news knowledge",
                    )
                    sections.append(self._format(older.as_dicts or [], "Earlier news"))
        return "\n".join(s for s in sections if s)

    @staticmethod
    def _format(articles: list[Any], heading: str) -> str:
        if not articles:
            return f"### {heading}\nNo articles found.\n"
        lines = [f"### {heading}"]
        for a in sorted(articles, key=lambda x: x.pub_date, reverse=True):
            date = a.pub_date.strftime("%Y-%m-%d")
            lines.append(
                f"- **{a.eng_title}** ({a.source_id}, {date}) {a.article_url}\n  {a.summary}"
            )
        return "\n".join(lines) + "\n"


HttpPost = Callable[..., Any]


class ExaSource:
    name = "Exa"

    def __init__(
        self,
        guard: BudgetGuard,
        num_results: int,
        reserve_usd: float,
        http_post: HttpPost | None = None,
    ) -> None:
        self.guard = guard
        self.num_results = num_results
        self.reserve_usd = reserve_usd
        self._post = http_post or requests.post

    async def search(self, query: str) -> str:
        reservation = self.guard.reserve(self.reserve_usd, kind="search")
        body = {
            "query": query,
            "type": "auto",
            "numResults": self.num_results,
            "contents": {"highlights": {"maxCharacters": 1500}},
        }
        try:
            response = await asyncio.to_thread(
                self._post,
                EXA_URL,
                json=body,
                headers={"x-api-key": os.environ["EXA_API_KEY"]},
                timeout=60,
            )
        except BaseException:
            self.guard.settle(reservation, self.reserve_usd)
            raise
        if getattr(response, "status_code", 200) >= 400:
            # Rejected requests (bad key, bad request, rate limit) are not billed.
            self.guard.cancel(reservation)
            raise RuntimeError(f"Exa search failed with HTTP {response.status_code}")
        data: dict = {}
        cost = self.reserve_usd  # charged in full if Exa does not report a cost
        try:
            data = response.json()
            reported = (data.get("costDollars") or {}).get("total")
            if reported is not None:
                cost = float(reported)
        except Exception as e:
            logger.warning("Could not read Exa response: %s", e)
        self.guard.settle(reservation, cost)
        return self._format(data.get("results") or [])

    @staticmethod
    def _format(results: list[dict]) -> str:
        if not results:
            return "### Web search\nNo results found.\n"
        lines = ["### Web search"]
        for r in results:
            title = r.get("title") or "(untitled)"
            date = (r.get("publishedDate") or "")[:10] or "date unknown"
            snippet = " ... ".join(r.get("highlights") or []) or (r.get("text") or "")[:1500]
            lines.append(f"- **{title}** ({date}) {r.get('url', '')}\n  {snippet.strip()}")
        return "\n".join(lines) + "\n"


Source = Any  # anything with .name and async .search(query) -> str


async def run_research(
    question_text: str, sources: list[Source], char_limit: int
) -> ResearchResult:
    async def one(source: Source) -> tuple[str, str]:
        try:
            return source.name, await source.search(question_text)
        except Exception as e:  # one bad source must not sink the question
            logger.warning("Research source %s failed: %s", source.name, e)
            return source.name, ""

    results = await asyncio.gather(*(one(s) for s in sources))
    used = [name for name, text in results if text.strip()]
    per_source_limit = max(2000, char_limit // max(1, len(sources)))
    body = "\n".join(
        f"## {name}\n{_truncate(text, per_source_limit)}" for name, text in results if text.strip()
    )
    return ResearchResult(text=body, sources_used=used)
