"""Claude list prices and the cost arithmetic the budget cap relies on.

Prices are Anthropic first-party list prices in US$ per million tokens, taken
from Anthropic's model table (cached 25 Sep 2026). OpenRouter lists the same
per-token prices for these models (checked 6 Oct 2026), so one table serves
both providers.

A model that is not in this table cannot be used unless its price is supplied
through MODEL_PRICES_JSON. The bot refuses to run an unpriced model, because it
could not enforce the spend cap for it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute cache writes cost 1.25x base input


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float

    @property
    def cache_write_per_mtok(self) -> float:
        return self.input_per_mtok * CACHE_WRITE_MULTIPLIER


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.00, 50.00, 0.25),
    "claude-fable-5": ModelPrice(10.00, 50.00, 1.00),
    "claude-opus-5-5": ModelPrice(4.00, 20.00, 0.20),
    "claude-opus-5": ModelPrice(5.00, 25.00, 0.50),
    "claude-opus-4-8": ModelPrice(5.00, 25.00, 0.50),
    "claude-opus-4-7": ModelPrice(5.00, 25.00, 0.50),
    "claude-opus-4-6": ModelPrice(5.00, 25.00, 0.50),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00, 0.20),
    "claude-sonnet-5": ModelPrice(2.00, 10.00, 0.20),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00, 0.30),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00, 0.10),
}

# Server-side refusal fallbacks re-run a declined request on another model
# (Claude Opus 5 / Opus 4.8 for Opus 5.5). When usage does not name the model
# that ran, it is priced at this rate, the dearest of the fallback targets.
FALLBACK_PRICE_MODEL = "claude-opus-5"


class UnknownModelPriceError(ValueError):
    pass


def normalise_model_id(model: str) -> str:
    """Map provider-specific ids back to the Anthropic id used in the table.

    "anthropic/claude-opus-5.5" (OpenRouter) -> "claude-opus-5-5".
    """
    name = model.strip()
    if "/" in name:
        name = name.split("/", 1)[1]
    name = name.split(":", 1)[0]
    return name.replace(".", "-")


def load_prices() -> dict[str, ModelPrice]:
    """Default table plus any overrides from MODEL_PRICES_JSON.

    MODEL_PRICES_JSON example:
    {"claude-opus-6": {"input": 5, "output": 25, "cache_read": 0.5}}
    """
    prices = dict(DEFAULT_PRICES)
    raw = os.getenv("MODEL_PRICES_JSON", "").strip()
    if raw:
        overrides = json.loads(raw)
        for model, p in overrides.items():
            prices[normalise_model_id(model)] = ModelPrice(
                float(p["input"]), float(p["output"]), float(p["cache_read"])
            )
    return prices


def price_for(model: str, prices: dict[str, ModelPrice] | None = None) -> ModelPrice:
    table = prices if prices is not None else load_prices()
    key = normalise_model_id(model)
    if key not in table:
        raise UnknownModelPriceError(
            f"No price known for model '{model}'. Add it to MODEL_PRICES_JSON "
            "or pick a model from gerno_bot/pricing.py; the bot will not run an "
            "unpriced model because it could not enforce the spend cap."
        )
    return table[key]


def _get(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def cost_of_tokens(
    price: ModelPrice,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    return (
        input_tokens * price.input_per_mtok
        + cache_creation_input_tokens * price.cache_write_per_mtok
        + cache_read_input_tokens * price.cache_read_per_mtok
        + output_tokens * price.output_per_mtok
    ) / 1_000_000


def cost_from_usage(
    usage: Any,
    requested_model: str,
    prices: dict[str, ModelPrice] | None = None,
) -> float:
    """Dollar cost of one Messages API response.

    With refusal fallbacks the per-attempt breakdown is in usage.iterations and
    the top-level usage covers only the attempt that produced the message, so
    iterations win when present.
    """
    table = prices if prices is not None else load_prices()
    iterations: Iterable[Any] = _get(usage, "iterations", None) or []
    iterations = list(iterations)
    if iterations:
        total = 0.0
        for entry in iterations:
            entry_model = _get(entry, "model", None)
            entry_type = _get(entry, "type", "message")
            if entry_model:
                model_for_price = str(entry_model)
            elif entry_type == "fallback_message":
                model_for_price = FALLBACK_PRICE_MODEL
            else:
                model_for_price = requested_model
            total += cost_of_tokens(
                price_for(model_for_price, table),
                int(_get(entry, "input_tokens")),
                int(_get(entry, "output_tokens")),
                int(_get(entry, "cache_creation_input_tokens")),
                int(_get(entry, "cache_read_input_tokens")),
            )
        return total
    return cost_of_tokens(
        price_for(requested_model, table),
        int(_get(usage, "input_tokens")),
        int(_get(usage, "output_tokens")),
        int(_get(usage, "cache_creation_input_tokens")),
        int(_get(usage, "cache_read_input_tokens")),
    )


def estimate_tokens(text: str) -> int:
    """Deliberately generous token estimate (about 3 characters per token)."""
    return len(text) // 3 + 16


def worst_case_cost(
    model: str,
    input_tokens_estimate: int,
    max_output_tokens: int,
    with_fallback: bool,
    prices: dict[str, ModelPrice] | None = None,
) -> float:
    """Most a single request can cost: all input billed as a cache write, every
    output token used, and (if fallbacks are on) the whole request run again on
    the fallback model."""
    table = prices if prices is not None else load_prices()
    price = price_for(model, table)
    worst = cost_of_tokens(price, 0, max_output_tokens, input_tokens_estimate, 0)
    if with_fallback:
        fallback = price_for(FALLBACK_PRICE_MODEL, table)
        worst += cost_of_tokens(
            fallback, 0, max_output_tokens, input_tokens_estimate, 0
        )
    return worst
