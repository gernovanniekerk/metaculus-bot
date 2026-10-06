import asyncio
from pathlib import Path

import yaml

from gerno_bot.dry_run import build_offline_bot
from gerno_bot.prompts import SYSTEM_PROMPT

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
SPENDING = ["run_bot_on_tournament.yaml", "test_bot.yaml", "run_bot_on_metaculus_cup.yaml"]


def load(name):
    with (WORKFLOWS / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_spending_workflows_share_one_lock_and_save_the_ledger():
    for name in SPENDING:
        wf = load(name)
        assert wf["concurrency"]["group"] == "metaculus-bot-spend", name
        assert wf["concurrency"]["cancel-in-progress"] is False, name
        assert wf["permissions"]["contents"] == "write", name
        last = wf["jobs"]["forecast_job"]["steps"][-1]
        assert last["if"] == "always()", name
        assert "commit_state.sh" in last["run"], name


def test_only_the_tournament_runs_on_a_schedule():
    on = {name: load(name)[True] for name in SPENDING}  # YAML parses `on:` as True
    assert "schedule" in on["run_bot_on_tournament.yaml"]
    assert "schedule" not in on["test_bot.yaml"]
    assert "schedule" not in on["run_bot_on_metaculus_cup.yaml"]


def test_ci_uses_no_secrets():
    text = (WORKFLOWS / "ci.yaml").read_text(encoding="utf-8")
    assert "secrets." not in text


def test_system_prompt_is_static_and_long_enough_to_cache():
    # Opus 5.5 / Sonnet 5.5 cache prefixes of 512+ tokens. English runs at
    # roughly 4 characters per token, so aim well above 2,048 characters.
    assert len(SYSTEM_PROMPT) > 2_800
    assert "{" not in SYSTEM_PROMPT  # nothing interpolated


def test_system_prompt_identical_across_questions(tmp_path):
    bot, sdk, fake, guard, _ = build_offline_bot(tmp_path, malformed_first_answer=False)
    asyncio.run(bot.forecast_on_tournament("t", return_exceptions=True))
    systems = {r["system"][0]["text"] for r in sdk.requests}
    assert systems == {SYSTEM_PROMPT}
