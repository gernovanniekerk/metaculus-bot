# Setup: what only Gerno can do

About 1.5 to 2 hours in total, plus waiting for AskNews to reply. Claude can't
do these steps: they are sign-ups, terms, payments and keys.

## Read this first: personal accounts only

**Bots built or run with company resources can't win prizes.** Metaculus counts
a bot as commercial if it is "built or operated on behalf of a for-profit entity
that has three or more founders/employees/contractors", and that "includes any
bots built using company resources". Commercial bots also get no free LLM
credits.

So every account, key, card and repo below must be **yours personally**:

- sign up with your personal email, not a work address;
- pay with your **personal** card, not any company card or account;
- use your personal GitHub account, not a company organisation;
- don't put the API bills through any company's books.

One prize bot per person: create **one** bot account only.

## Steps

**1. Metaculus accounts (10 min).** Go to
https://www.metaculus.com/futureeval/participate/ and sign up (human account).
Accept the terms. Then Settings → My Forecasting Bots → Create a Bot. Any bot
name is fine (a pseudonym is allowed; prize payouts still use your real
identity). Copy the bot's API key: that is `METACULUS_TOKEN`.

**2. Participation form (10 min).** https://forms.gle/aQdYMq9Pisrf1v7d8. Fill in
the first section (3 required questions). The same form is the application for
free LLM credits. If you apply and get them, they arrive as an OpenRouter key:
add it as `OPENROUTER_API_KEY` (see step 6) and tell me, because the OpenRouter
route is built but has not yet been run live.

**3. Anthropic API key (15 min).** At https://platform.claude.com, using a
personal account and personal card:
- buy credits (US$50 is enough for the first month);
- Settings → Workspaces → Create workspace, name it `metaculus-bot`;
- in that workspace's **Spend limits** tab, set a monthly limit of **US$55**.
  The bot stops itself at US$50; this is the backstop if anything goes wrong;
- create an API key in that workspace: that is `ANTHROPIC_API_KEY`.

**4. AskNews (10 min, then wait for a reply).** Free for registered bot makers
(1,000 calls a month; the bot uses about one per question). Make an account at
https://my.asknews.app with your personal email. Then email
contact@asknews.app (or use the Metaculus Discord) with your bot name, the email
you registered with, your name, LinkedIn, and affiliation ("individual"), asking
for the FutureEval Fall 2026 allowance. When it's active, create credentials at
https://my.asknews.app/en/settings/api-credentials: `ASKNEWS_CLIENT_ID` and
`ASKNEWS_SECRET` (or a single `ASKNEWS_API_KEY`). You must renew this every
season.

**5. Exa (5 min).** Sign up at https://exa.ai (personal account and card) and
create an API key: `EXA_API_KEY`. Not free for the tournament; Metaculus lists it
at half a cent per search plus a tenth of a cent per page, which is about
1 US cent per question here. It counts against the bot's US$50 cap.

**6. GitHub repo and secrets (15 min).**
- Create a new repo under your personal GitHub account and push this one to it
  (or tell me the repo URL and I'll push).
- **Public or private:** public is recommended. Actions minutes are free on
  public repos, Metaculus gives open-source bots about double the credits, and a
  winner has to disclose the code or method anyway. On a private repo the free
  allowance is 2,000 minutes a month and the 20-minute schedule needs about
  2,200 runs a month, so change the schedule in
  `.github/workflows/run_bot_on_tournament.yaml` to every 30 or 60 minutes.
- Settings → Secrets and variables → Actions → **New repository secret**, one
  each: `METACULUS_TOKEN`, `ANTHROPIC_API_KEY`, `ASKNEWS_CLIENT_ID`,
  `ASKNEWS_SECRET` (or `ASKNEWS_API_KEY`), `EXA_API_KEY`, and
  `OPENROUTER_API_KEY` only if you got credits. Names must match exactly.
- Optional, under the **Variables** tab: `MONTHLY_BUDGET_USD` to change the cap
  (default 50).

**7. Turn it on and test (10 min, plus about 10 min waiting).**
- Actions tab → "I understand my workflows, go ahead and enable them".
- Actions → **Test Bot** → Run workflow. This runs the offline tests, then
  forecasts on Metaculus's practice tournament (bot-testing-area). It costs real
  money: roughly US$0.30 a question, a few dollars in all (estimate).
- From then on the **Forecast on new AI tournament questions** workflow runs by
  itself every 20 minutes.

**8. Each season (30 min).** Metaculus emails a bot-maker survey after the
season's questions resolve (for Fall 2026, early 2027). It is required for
prizes, for winners and non-winners. If you win: verify identity, nationality
and residency; prizes are paid by bank transfer through Ramp.

## Check the first live run

1. **Test Bot run is green.** Open the run's log. You should see:
   - `Spend this month: $0.00 of $50.00 cap; tier now: full; provider: anthropic`
   - `Research sources: ['AskNews', 'Exa']` (if one is missing, its secret is wrong)
   - `Bot submitted N forecast(s)` with a link per question
   - lines like `LLM forecast ... cache_read=...`: from the second ensemble
     member on, `cache_read` should be above zero (prompt caching working).
2. **The forecasts and comments are on Metaculus.** Log in as yourself → your
   name → Settings → My Forecasting Bots → Switch to bot account. Open one of the
   linked questions: the bot's forecast is on the chart and its comment is under
   the **Private** tab, starting `# SUMMARY` then `*Method*: ...`.
3. **Spend was recorded.** The repo has a new commit "Update spend ledger";
   `state/spend_ledger.json` shows the month's spend. It should be close to, and
   slightly above, what the Anthropic Console shows for the `metaculus-bot`
   workspace (the bot rounds uncertain calls up).
4. **The tournament run works.** Within 20 minutes, Actions shows a "Forecast on
   new AI tournament questions" run. Its log should list questions from the Fall
   2026 tournament and MiniBench, or say there are no new questions.

If the "Save spend ledger" step fails with a permissions error: Settings →
Actions → General → Workflow permissions → **Read and write permissions**.

## Costs (estimates, not yet measured)

- Full ensemble (3 Opus 5.5 + 2 Sonnet 5.5): about US$0.25-0.30 a question.
- Economy ensemble (3 Sonnet 5.5): about US$0.05-0.10 a question.
- Expected volume: 300-400 tournament questions a season plus about 60 MiniBench
  questions a fortnight, roughly 240-280 a month. At US$50 a month most get the
  full ensemble and the rest the economy one.
- Season total: about US$120-150 (October to mid-December).

These rest on assumed answer lengths. The real cost per question shows in
`state/spend_ledger.json` after the first day; send it to me and I'll tune.

## Stopping it

Actions → "Forecast on new AI tournament questions" → ··· → Disable workflow. Or
set the `MONTHLY_BUDGET_USD` variable to `0`.
