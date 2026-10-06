"""All tunable settings, read from environment variables (GitHub secrets or
repository variables in Actions, a local .env file otherwise).

Every model id is configurable. Defaults are the current Anthropic models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Ensemble spec format: comma-separated "model@effort*count" entries.
DEFAULT_FULL_ENSEMBLE = "claude-opus-5-5@medium*3,claude-sonnet-5-5@medium*2"
DEFAULT_ECONOMY_ENSEMBLE = "claude-sonnet-5-5@low*3"
DEFAULT_PARSER_MODEL = "claude-haiku-4-5"

VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class EnsembleMember:
    model: str
    effort: str
    index: int = 0  # position within the ensemble, 0-based

    @property
    def label(self) -> str:
        return f"{self.model}@{self.effort}"


def parse_ensemble(spec: str) -> list[EnsembleMember]:
    members: list[EnsembleMember] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        count = 1
        if "*" in raw:
            raw, count_text = raw.rsplit("*", 1)
            count = int(count_text)
            if count < 1:
                raise ValueError(f"Ensemble count must be >= 1 in '{spec}'")
        if "@" in raw:
            model, effort = raw.split("@", 1)
        else:
            model, effort = raw, "medium"
        model, effort = model.strip(), effort.strip().lower()
        if effort not in VALID_EFFORTS:
            raise ValueError(f"Unknown effort '{effort}' in ensemble spec '{spec}'")
        for _ in range(count):
            members.append(EnsembleMember(model=model, effort=effort))
    if not members:
        raise ValueError(f"Ensemble spec '{spec}' has no members")
    return [
        EnsembleMember(model=m.model, effort=m.effort, index=i)
        for i, m in enumerate(members)
    ]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


def has_value(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip() and value.strip() != "REPLACE_ME")


@dataclass(frozen=True)
class Settings:
    # LLM provider: "anthropic" (ANTHROPIC_API_KEY) or "openrouter"
    # (OPENROUTER_API_KEY, which is how Metaculus hands out free credits).
    llm_provider: str
    full_ensemble: list[EnsembleMember]
    economy_ensemble: list[EnsembleMember]
    parser_model: str
    max_output_tokens: int
    use_refusal_fallbacks: bool

    monthly_budget_usd: float
    max_usd_per_question: float
    pace_headstart: float
    ledger_path: str

    asknews_enabled: bool
    asknews_monthly_call_limit: int
    asknews_historical: bool
    exa_enabled: bool
    exa_num_results: int
    exa_reserve_usd: float

    question_concurrency: int
    aib_tournament_id: str
    minibench_tournament_id: str
    research_char_limit: int = 24_000

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env("LLM_PROVIDER", "auto").lower()
        if provider == "auto":
            if has_value("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif has_value("OPENROUTER_API_KEY"):
                provider = "openrouter"
            else:
                provider = "none"
        if provider not in {"anthropic", "openrouter", "none"}:
            raise ValueError(f"LLM_PROVIDER must be anthropic, openrouter or auto, not '{provider}'")

        from forecasting_tools import MetaculusClient

        max_output_tokens = _env_int("MAX_OUTPUT_TOKENS", 10_000)
        if not 1_000 <= max_output_tokens <= 20_000:
            # Requests are non-streaming; the SDK refuses non-streaming calls
            # whose max_tokens implies more than 10 minutes (about 21k tokens).
            raise ValueError("MAX_OUTPUT_TOKENS must be between 1000 and 20000")

        return cls(
            llm_provider=provider,
            full_ensemble=parse_ensemble(_env("FULL_ENSEMBLE", DEFAULT_FULL_ENSEMBLE)),
            economy_ensemble=parse_ensemble(
                _env("ECONOMY_ENSEMBLE", DEFAULT_ECONOMY_ENSEMBLE)
            ),
            parser_model=_env("PARSER_MODEL", DEFAULT_PARSER_MODEL),
            max_output_tokens=max_output_tokens,
            # Server-side refusal fallbacks are a Claude API feature; not used
            # through OpenRouter.
            use_refusal_fallbacks=_env_bool("USE_REFUSAL_FALLBACKS", True)
            and provider == "anthropic",
            monthly_budget_usd=_env_float("MONTHLY_BUDGET_USD", 50.0),
            max_usd_per_question=_env_float("MAX_USD_PER_QUESTION", 1.50),
            pace_headstart=_env_float("PACE_HEADSTART", 0.25),
            ledger_path=_env("SPEND_LEDGER_PATH", "state/spend_ledger.json"),
            asknews_enabled=(
                (has_value("ASKNEWS_CLIENT_ID") and has_value("ASKNEWS_SECRET"))
                or has_value("ASKNEWS_API_KEY")
            ),
            asknews_monthly_call_limit=_env_int("ASKNEWS_MONTHLY_CALL_LIMIT", 900),
            asknews_historical=_env_bool("ASKNEWS_HISTORICAL", False),
            exa_enabled=has_value("EXA_API_KEY"),
            exa_num_results=_env_int("EXA_NUM_RESULTS", 6),
            exa_reserve_usd=_env_float("EXA_RESERVE_USD", 0.02),
            question_concurrency=_env_int("QUESTION_CONCURRENCY", 2),
            aib_tournament_id=_env(
                "AIB_TOURNAMENT_ID", str(MetaculusClient.CURRENT_AI_COMPETITION_ID)
            ),
            minibench_tournament_id=_env(
                "MINIBENCH_TOURNAMENT_ID", str(MetaculusClient.CURRENT_MINIBENCH_ID)
            ),
        )
