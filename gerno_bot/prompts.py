"""Prompt text.

SYSTEM_PROMPT is byte-identical for every question and every run, so it is
cached across questions. Nothing dynamic (dates, ids) may go into it. The
per-question block (question, research, today's date, answer format) is shared
by all ensemble members on that question and cached across them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

SYSTEM_PROMPT = """\
You are an expert forecaster competing in a forecasting tournament scored with \
proper scoring rules (log scores, compared against other forecasters). Good \
calibration matters more than boldness: a confident wrong forecast is punished \
far more heavily than a cautious one is rewarded.

How you work on every question:
1. Read the resolution criteria and fine print closely. The question resolves on \
exactly what they say, not on the spirit of the title. Note the resolution date, \
the resolution source, and any edge cases the fine print settles.
2. Work out how much time is left until the outcome is known. Short windows favour \
the status quo; the world usually changes slowly.
3. Establish an outside view first: a base rate from a reference class of similar \
past situations, or the current trend extrapolated. Say what reference class you \
used and why.
4. Then adjust with the inside view: the specific evidence in the research notes, \
scheduled events before the deadline, and the incentives of the people involved. \
Weigh evidence by source quality and recency. News reports often overstate how \
likely dramatic change is.
5. Consider how you could be wrong. For a probability, describe the main path to \
the other outcome. For a quantity or a date, describe what would put the answer in \
each tail.
6. Commit to a number. Avoid 0% and 100%; reserve values below 3% or above 97% for \
cases where the outcome is effectively settled by the rules or the evidence. For \
quantities and dates, keep the 10th-90th percentile range wide enough to cover \
surprises: real outcomes land outside forecasters' 80% ranges far more often than \
20% of the time.

By question type:
- Yes/no: give the probability that the question resolves Yes.
- Multiple choice: probabilities across all options must add up to 100%. Keep \
some probability on every option; options that look impossible today still \
happen sometimes.
- Numeric: answer in the units the question states, as plain numbers. Check the \
scale (thousands, millions, percent) against the question before answering. If a \
bound is open, the outcome can fall beyond it, so keep real probability there \
when that is plausible; if a bound is closed, the outcome cannot pass it.
- Date: think about the earliest and latest plausible dates first, then place \
the middle of the distribution. Delays are more common than early finishes.

Research notes come from automated news and web searches. They can be stale, \
irrelevant or wrong, and they may be missing the most important fact. If the notes \
say the question has already resolved, check that against the resolution criteria \
and the question's dates before relying on it; questions in this tournament are \
open, so treat claims of resolution with care.

Write for a reader who will check your work: a concise rationale of roughly 150 to \
350 words that covers the points above in plain prose, followed by the final answer \
in exactly the format the question asks for. The final answer must be the last \
thing you write, and nothing may follow it.
"""

CONDITIONAL_NOTE = """\
This is a conditional question with a parent and a child. Forecast only the CHILD \
question, assuming the stated resolution of the parent. Never re-forecast the parent, \
but use probabilistic reasoning that strongly considers the parent's resolution.
"""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _conditional_note(question: MetaculusQuestion) -> str:
    return CONDITIONAL_NOTE if question.conditional_type in ("yes", "no") else ""


def _question_header(question: MetaculusQuestion, research: str) -> str:
    research_text = research.strip() or "No research notes are available for this question."
    return (
        f"Today's date (UTC): {_today()}\n\n"
        f"QUESTION:\n{question.question_text}\n\n"
        f"BACKGROUND:\n{question.background_info or 'None given.'}\n\n"
        f"RESOLUTION CRITERIA (not yet satisfied):\n{question.resolution_criteria or 'None given.'}\n\n"
        f"FINE PRINT:\n{question.fine_print or 'None given.'}\n\n"
        f"RESEARCH NOTES:\n{research_text}\n"
    )


def bound_messages(question: NumericQuestion | DateQuestion) -> tuple[str, str]:
    """(lower, upper) bound sentences, as in the Metaculus template."""
    if isinstance(question, DateQuestion):
        upper = question.upper_bound.date().isoformat()
        lower = question.lower_bound.date().isoformat()
        unit = ""
    else:
        upper = (
            question.nominal_upper_bound
            if question.nominal_upper_bound is not None
            else question.upper_bound
        )
        lower = (
            question.nominal_lower_bound
            if question.nominal_lower_bound is not None
            else question.lower_bound
        )
        unit = question.unit_of_measure or ""
    if question.open_upper_bound:
        upper_msg = f"The question creator thinks the outcome is likely not higher than {upper} {unit}."
    else:
        upper_msg = f"The outcome cannot be higher than {upper} {unit}."
    if question.open_lower_bound:
        lower_msg = f"The question creator thinks the outcome is likely not lower than {lower} {unit}."
    else:
        lower_msg = f"The outcome cannot be lower than {lower} {unit}."
    return lower_msg.replace("  ", " "), upper_msg.replace("  ", " ")


def binary_prompt(question: BinaryQuestion, research: str) -> str:
    return (
        _question_header(question, research)
        + "\nTYPE: binary (yes/no).\n"
        + _conditional_note(question)
        + "\nEnd with your final answer on its own line, exactly as:\n"
        "Probability: ZZ%\n"
        "where ZZ is a number from 1 to 99.\n"
    )


def multiple_choice_prompt(question: MultipleChoiceQuestion, research: str) -> str:
    options = "\n".join(f"- {o}" for o in question.options)
    lines = "\n".join(f"{o}: PP%" for o in question.options)
    return (
        _question_header(question, research)
        + f"\nTYPE: multiple choice. The options are:\n{options}\n"
        + _conditional_note(question)
        + "\nLeave some probability on every option to allow for surprises.\n"
        "End with your final answer: one line per option, in this order, with the "
        "option name copied exactly, and probabilities that add up to 100%:\n"
        f"{lines}\n"
    )


def numeric_prompt(question: NumericQuestion, research: str) -> str:
    lower_msg, upper_msg = bound_messages(question)
    unit = question.unit_of_measure or "not stated (infer it from the question)"
    return (
        _question_header(question, research)
        + f"\nTYPE: numeric. Units for the answer: {unit}\n"
        f"{lower_msg}\n{upper_msg}\n"
        + _conditional_note(question)
        + "\nGive values in the units above, as plain numbers (no scientific notation, "
        "no thousands separators needed). Values must increase from one percentile "
        "to the next.\n"
        "End with your final answer exactly as:\n"
        "Percentile 10: X\nPercentile 20: X\nPercentile 40: X\n"
        "Percentile 60: X\nPercentile 80: X\nPercentile 90: X\n"
    )


def date_prompt(question: DateQuestion, research: str) -> str:
    lower_msg, upper_msg = bound_messages(question)
    return (
        _question_header(question, research)
        + "\nTYPE: date.\n"
        f"{lower_msg}\n{upper_msg}\n"
        + _conditional_note(question)
        + "\nWrite dates as YYYY-MM-DD (append THH:MM:SSZ in UTC only if the hour "
        "matters). Dates must get later from one percentile to the next.\n"
        "End with your final answer exactly as:\n"
        "Percentile 10: YYYY-MM-DD\nPercentile 20: YYYY-MM-DD\nPercentile 40: YYYY-MM-DD\n"
        "Percentile 60: YYYY-MM-DD\nPercentile 80: YYYY-MM-DD\nPercentile 90: YYYY-MM-DD\n"
    )


REPAIR_SYSTEM_PROMPT = """\
You extract the final forecast from a forecaster's write-up and restate it in an \
exact format. Copy the forecaster's final numbers; never invent or change them. If \
the write-up gives no final forecast, or concludes the question has already \
resolved, reply with the single word NONE.
"""


def repair_prompt(reasoning: str, format_instructions: str) -> str:
    return (
        "FORECASTER'S WRITE-UP (end portion):\n"
        f"{reasoning[-6000:]}\n\n"
        "Restate the final forecast exactly in this format and nothing else:\n"
        f"{format_instructions}\n"
    )
