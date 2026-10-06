"""The only path to Claude. Every request is reserved against the budget
first and settled with the cost the API reports afterwards.

Prompt caching: the system prompt (identical for every question) and the
question-plus-research block (identical for every ensemble member on one
question) each carry a cache breakpoint, so the second and later ensemble
members read the shared prefix from cache.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

from gerno_bot.budget import BudgetGuard
from gerno_bot.pricing import (
    cost_from_usage,
    estimate_tokens,
    load_prices,
    normalise_model_id,
    price_for,
    worst_case_cost,
)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Models that accept output_config.effort and adaptive thinking. Haiku 4.5
# takes neither, so it is called without them.
_NO_EFFORT_PREFIXES = ("claude-haiku-",)
# Models that accept the server-side `fallbacks: "default"` parameter.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}

# Errors raised before the model ran: nothing was billed. (ValueError and
# TypeError come from the SDK rejecting a request before sending it.)
_NOT_BILLED_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    anthropic.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.RateLimitError,
    anthropic.UnprocessableEntityError,
)


class LlmRefusalError(RuntimeError):
    pass


class LlmTruncatedError(RuntimeError):
    pass


@dataclass
class LlmResult:
    text: str
    model: str
    cost_usd: float
    stop_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    note: str = ""  # set when the final answer had to be restated (see bot.py)


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return dict(usage)
    return {}


class ClaudeClient:
    def __init__(
        self,
        guard: BudgetGuard,
        provider: str,
        use_refusal_fallbacks: bool,
        sdk_client: Any | None = None,
    ) -> None:
        self.guard = guard
        self.provider = provider
        self.use_refusal_fallbacks = use_refusal_fallbacks and provider == "anthropic"
        self.prices = load_prices()
        self._sdk = sdk_client if sdk_client is not None else self._make_sdk_client()

    def _make_sdk_client(self) -> Any:
        if self.provider == "anthropic":
            return anthropic.AsyncAnthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=4, timeout=600.0
            )
        if self.provider == "openrouter":
            # OpenRouter speaks the Anthropic Messages API at this base URL with
            # the OpenRouter key as a bearer token.
            return anthropic.AsyncAnthropic(
                base_url=OPENROUTER_BASE_URL,
                auth_token=os.environ["OPENROUTER_API_KEY"],
                api_key=None,
                max_retries=4,
                timeout=600.0,
            )
        raise RuntimeError(
            "No LLM key configured: set ANTHROPIC_API_KEY (or OPENROUTER_API_KEY)."
        )

    def api_model_id(self, model: str) -> str:
        if self.provider == "openrouter":
            if "/" in model:
                return model
            # claude-opus-5-5 -> anthropic/claude-opus-5.5
            head, _, tail = model.rpartition("-")
            if head and tail.isdigit() and head[-1:].isdigit():
                return f"anthropic/{head}.{tail}"
            return f"anthropic/{model}"
        return model

    async def complete(
        self,
        *,
        model: str,
        effort: str | None,
        system_text: str,
        shared_text: str,
        tail_text: str = "",
        max_tokens: int,
        purpose: str,
    ) -> LlmResult:
        """One Messages API call.

        system_text: stable across all questions (cached).
        shared_text: stable across ensemble members of one question (cached).
        tail_text: anything that varies per call (not cached).
        """
        canonical = normalise_model_id(model)
        price_for(canonical, self.prices)  # fail fast on unpriced models
        input_estimate = estimate_tokens(system_text + shared_text + tail_text)

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": shared_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if tail_text:
            user_content.append({"type": "text", "text": tail_text})
        request: dict[str, Any] = {
            "model": self.api_model_id(canonical),
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_content}],
        }
        if effort and not canonical.startswith(_NO_EFFORT_PREFIXES):
            request["thinking"] = {"type": "adaptive"}
            request["output_config"] = {"effort": effort}

        while True:
            with_fallback = self.use_refusal_fallbacks and canonical in _FALLBACK_MODELS
            reservation_amount = worst_case_cost(
                canonical, input_estimate, max_tokens, with_fallback, self.prices
            )
            reservation = self.guard.reserve(reservation_amount, kind="llm")
            try:
                if with_fallback:
                    response = await self._sdk.beta.messages.create(
                        betas=[FALLBACK_BETA], fallbacks="default", **request
                    )
                else:
                    response = await self._sdk.messages.create(**request)
                break
            except anthropic.BadRequestError as e:
                self.guard.cancel(reservation)
                if with_fallback and "fallback" in str(e).lower():
                    # The beta was rejected (e.g. not enabled for this account):
                    # carry on without it rather than failing every forecast.
                    logger.warning("Refusal fallbacks rejected (%s); continuing without them", e)
                    self.use_refusal_fallbacks = False
                    continue
                raise
            except _NOT_BILLED_ERRORS:
                self.guard.cancel(reservation)
                raise
            except BaseException:
                # Timeouts, dropped connections and 5xx may still have been billed.
                self.guard.settle(reservation, reservation_amount)
                raise

        try:
            cost = cost_from_usage(response.usage, canonical, self.prices)
        except Exception:
            logger.exception("Could not price a response; charging the reservation")
            cost = reservation_amount
        self.guard.settle(reservation, cost)

        usage = _usage_dict(getattr(response, "usage", None))
        logger.info(
            "LLM %s %s: $%.4f, in=%s cache_write=%s cache_read=%s out=%s stop=%s",
            purpose,
            canonical,
            cost,
            usage.get("input_tokens"),
            usage.get("cache_creation_input_tokens"),
            usage.get("cache_read_input_tokens"),
            usage.get("output_tokens"),
            getattr(response, "stop_reason", None),
        )
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise LlmRefusalError(f"{canonical} declined the request ({purpose})")
        text = "".join(
            getattr(block, "text", "")
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LlmTruncatedError(
                f"{canonical} returned no text (stop_reason={stop_reason}, {purpose})"
            )
        return LlmResult(
            text=text,
            model=str(getattr(response, "model", canonical)),
            cost_usd=cost,
            stop_reason=stop_reason,
            usage=usage,
        )
