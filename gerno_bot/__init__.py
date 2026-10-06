"""Gerno van Niekerk's Metaculus FutureEval forecasting bot.

Everything that can spend money (Claude calls, Exa searches) goes through
gerno_bot.budget.BudgetGuard, which refuses any call that could take the
month's spend past MONTHLY_BUDGET_USD.
"""
