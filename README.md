# Metaculus forecasting bot (Gerno van Niekerk)

A fully automated entry for the Metaculus FutureEval AI forecasting bot
tournament (Fall 2026 season and MiniBench). It runs on GitHub Actions every
20 minutes, forecasts each new question once, and posts a private comment with
its research and reasoning on every question it forecasts. No human is in the
loop.

**To get it running, follow [SETUP.md](SETUP.md).**

## How it works

For each new question:

1. **Budget check.** Month-to-date spend is read from `state/spend_ledger.json`.
   While spend is on pace for the month the bot uses the full ensemble; once it
   is ahead of pace it switches to a cheaper one; at the cap it stops.
2. **Research.** AskNews (latest news) and Exa (web search), whichever are set up.
3. **Ensemble.** Five independent Claude forecasts by default: three
   `claude-opus-5-5` and two `claude-sonnet-5-5`, both at medium effort with
   adaptive thinking. The first runs alone so it writes the prompt cache; the
   other four read it (the system prompt and the question-plus-research block are
   cached).
4. **Aggregate and publish.** Median of the members (mean for multiple choice),
   posted with a private comment that contains the research and every member's
   rationale. If a comment fails to post after the forecast went in, it is saved
   in `state/pending_comments.json` and retried next run.

## The spend cap

Default US$50 a month (`MONTHLY_BUDGET_USD`), enforced in code
(`gerno_bot/budget.py`):

- Before every Claude call or Exa search the bot reserves the most that call
  could cost. If that could take the month past the cap, the call is not made.
- After the call it records the cost the API reported. A call that failed in a
  way that might still have been billed is charged at the reserved maximum.
- The ledger is written after every call and committed back to the repo by the
  workflow, so each run starts from the real month-to-date figure.
- A per-question ceiling (`MAX_USD_PER_QUESTION`, default $1.50) stops runaway
  questions.
- Prices are in `gerno_bot/pricing.py`. The bot refuses to run a model it has no
  price for.

The Anthropic Console workspace spend limit in SETUP.md is a second, independent
backstop.

## Configuration

All optional, set as GitHub repository variables (or in `.env` locally):

| Variable | Default | Meaning |
|---|---|---|
| `MONTHLY_BUDGET_USD` | 50 | Hard monthly cap for Claude + Exa spend |
| `MAX_USD_PER_QUESTION` | 1.50 | Runaway guard per question |
| `FULL_ENSEMBLE` | `claude-opus-5-5@medium*3,claude-sonnet-5-5@medium*2` | `model@effort*count`, comma separated |
| `ECONOMY_ENSEMBLE` | `claude-sonnet-5-5@low*3` | Used when spend is ahead of pace |
| `PARSER_MODEL` | `claude-haiku-4-5` | Only called if a forecast's final line can't be read |
| `PACE_HEADSTART` | 0.25 | Share of the cap the full ensemble may spend ahead of pace |
| `LLM_PROVIDER` | `auto` | `anthropic`, or `openrouter` for Metaculus free credits |
| `ASKNEWS_MONTHLY_CALL_LIMIT` | 900 | Stops using AskNews before the free quota runs out |
| `QUESTION_CONCURRENCY` | 2 | Questions worked on at once |

## Running and testing locally

```bash
poetry install
poetry run pytest                          # offline unit and end-to-end tests
poetry run python main.py --mode dry_run   # offline rehearsal: fake LLM, fake search,
                                           # fake Metaculus, network blocked, spends nothing
poetry run python main.py --mode test_questions   # REAL: bot-testing-area, costs money
```

## Origin and licence

Built from Metaculus's official template,
[Metaculus/metac-bot-template](https://github.com/Metaculus/metac-bot-template)
(commit da5de87, 1 Oct 2026). The first commit in this repo is an unmodified copy
of it. The template repository publishes no licence file (GitHub reports none as
of 6 Oct 2026); Metaculus provides it for participants to fork. It depends on
Metaculus's [forecasting-tools](https://github.com/Metaculus/forecasting-tools)
package (MIT). The fixture questions in `tests/fixtures` come from
forecasting-tools' test data (public Metaculus questions).

Changes from the template: the LLM layer is replaced with a metered Claude client
(official Anthropic SDK); research, ensemble, budget, parsing and comment-retry
code is in `gerno_bot/`; `main_with_no_framework.py` and the optional
integrations were removed so there is no code path around the spend cap; the
Metaculus Cup workflow no longer runs on a schedule.
