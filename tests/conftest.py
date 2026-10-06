"""Every test runs with network access blocked at the socket layer, so no test
can reach Anthropic, AskNews, Exa or Metaculus even by accident."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make settings deterministic whatever the developer's shell has set.
for name in [
    "LLM_PROVIDER",
    "FULL_ENSEMBLE",
    "ECONOMY_ENSEMBLE",
    "PARSER_MODEL",
    "MONTHLY_BUDGET_USD",
    "MAX_USD_PER_QUESTION",
    "PACE_HEADSTART",
    "MAX_OUTPUT_TOKENS",
    "MODEL_PRICES_JSON",
    "SPEND_LEDGER_PATH",
]:
    os.environ.pop(name, None)
os.environ["EXA_API_KEY"] = "test-dummy-not-a-real-key"

from gerno_bot.fakes import block_network  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network():
    with block_network():
        yield
