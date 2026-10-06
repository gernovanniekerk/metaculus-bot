"""The forecasting bot: Metaculus's FallTemplateBot2026 flow (forecasting-tools
ForecastBot) with the LLM, research, ensemble and budget layers replaced.

Per question:
1. Pick the tier: the full ensemble while month-to-date spend is on pace,
   the economy ensemble once it is ahead of pace (see budget.py).
2. Research once (AskNews + Exa, whichever are configured).
3. Ask each ensemble member for a forecast. The first member runs alone so it
   writes the prompt cache; the rest then run in parallel and read it.
4. Aggregate with forecasting-tools (median for binary and numeric/date,
   mean for multiple choice) and publish the forecast plus a private comment
   containing the research and every member's rationale.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from forecasting_tools import (
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DateQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    PredictionAffirmed,
    ReasonedPrediction,
    clean_indents,
)
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from gerno_bot import parsing, prompts
from gerno_bot.budget import BudgetExceededError, BudgetGuard
from gerno_bot.claude_client import ClaudeClient, LlmResult
from gerno_bot.config import EnsembleMember, Settings
from gerno_bot.pricing import estimate_tokens, worst_case_cost
from gerno_bot.research import run_research

logger = logging.getLogger(__name__)


@dataclass
class QuestionPlan:
    tier: str
    members: list[EnsembleMember]
    sources_used: list[str] = field(default_factory=list)


_current_plan: contextvars.ContextVar[QuestionPlan | None] = contextvars.ContextVar(
    "question_plan", default=None
)
_current_member: contextvars.ContextVar[EnsembleMember | None] = (
    contextvars.ContextVar("ensemble_member", default=None)
)

BINARY_FORMAT = "Probability: ZZ%"
PERCENTILE_FORMAT = "\n".join(f"Percentile {p}: X" for p in parsing.PERCENTILES)
DATE_FORMAT = "\n".join(f"Percentile {p}: YYYY-MM-DD" for p in parsing.PERCENTILES)


class GernoBot(ForecastBot):
    def __init__(
        self,
        *,
        settings: Settings,
        guard: BudgetGuard,
        claude: ClaudeClient,
        research_sources: list[Any],
        publish_reports_to_metaculus: bool,
        metaculus_client: Any | None = None,
        skip_previously_forecasted_questions: bool = True,
    ) -> None:
        self.settings = settings
        self.guard = guard
        self.claude = claude
        self.research_sources = research_sources
        self._question_semaphore = asyncio.Semaphore(max(1, settings.question_concurrency))
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=len(settings.full_ensemble),
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=None,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=None,
            enable_summarize_research=False,
            extra_metadata_in_explanation=False,
            required_successful_predictions=0.5,
            metaculus_client=metaculus_client,
        )

    # All LLM traffic goes through ClaudeClient; none through litellm.
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, Any]:
        return {}

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        return "Not generated; the full research notes are in the RESEARCH section."

    @property
    def expected_total_predictions(self) -> int:
        plan = _current_plan.get()
        return len(plan.members) if plan else super().expected_total_predictions

    # ------------------------------------------------------------ per question
    def minimum_question_cost(self, members: list[EnsembleMember]) -> float:
        """Reservation needed to start a question: research plus the first member."""
        first = members[0]
        est_input = estimate_tokens(prompts.SYSTEM_PROMPT) + self.settings.research_char_limit // 3 + 2000
        search = self.settings.exa_reserve_usd if any(
            getattr(s, "name", "") == "Exa" for s in self.research_sources
        ) else 0.0
        return search + worst_case_cost(
            first.model,
            est_input,
            self.settings.max_output_tokens,
            self.claude.use_refusal_fallbacks,
        )

    async def _run_individual_question(self, question: MetaculusQuestion):  # type: ignore[override]
        async with self._question_semaphore:
            tier = self.guard.choose_tier()
            members = (
                self.settings.full_ensemble
                if tier == "full"
                else self.settings.economy_ensemble
            )
            needed = self.minimum_question_cost(members)
            if self.guard.remaining_this_month() < needed:
                raise BudgetExceededError(
                    f"Skipping {question.page_url}: ${self.guard.remaining_this_month():.2f} "
                    f"left this month, a question needs up to ${needed:.2f} to start."
                )
            plan = QuestionPlan(tier=tier, members=list(members))
            token = _current_plan.set(plan)
            key = question.page_url or f"post-{question.id_of_post}"
            try:
                with self.guard.question_scope(key, tier):
                    return await super()._run_individual_question(question)
            finally:
                _current_plan.reset(token)

    async def run_research(self, question: MetaculusQuestion) -> str:
        if not self.research_sources:
            return ""
        result = await run_research(
            question.question_text, self.research_sources, self.settings.research_char_limit
        )
        plan = _current_plan.get()
        if plan is not None:
            plan.sources_used = result.sources_used
        logger.info("Research for %s used: %s", question.page_url, result.sources_used)
        return result.text

    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions:
        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        plan = _current_plan.get()
        members = plan.members if plan else self.settings.full_ensemble

        # First member alone: its request writes the cache the others read.
        first_ok, first_errors, _ = await self._gather_results_and_exceptions(
            [self._member_prediction(members[0], question, research)]
        )
        rest_ok, rest_errors, _ = await self._gather_results_and_exceptions(
            [self._member_prediction(m, question, research) for m in members[1:]]
        )
        predictions = first_ok + rest_ok
        errors = first_errors + rest_errors
        if errors:
            logger.warning("Ensemble errors on %s: %s", question.page_url, errors)
        if not predictions:
            raise RuntimeError(
                f"All {len(members)} ensemble members failed on {question.page_url}: {errors}"
            )
        return ResearchWithPredictions(
            research_report=research or "No research sources were available.",
            summary_report=summary_report,
            errors=errors,
            predictions=predictions,
        )

    async def _member_prediction(
        self, member: EnsembleMember, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction:
        _current_member.set(member)  # this coroutine runs in its own task/context
        return await self._make_prediction(question, research)

    # -------------------------------------------------------------- LLM calls
    async def _ask(self, question: MetaculusQuestion, shared_prompt: str) -> LlmResult:
        member = _current_member.get() or self.settings.full_ensemble[0]
        return await self.claude.complete(
            model=member.model,
            effort=member.effort,
            system_text=prompts.SYSTEM_PROMPT,
            shared_text=shared_prompt,
            max_tokens=self.settings.max_output_tokens,
            purpose=f"forecast post {question.id_of_post} member {member.index}",
        )

    async def _parse_with_repair(
        self, result: LlmResult, parse: Callable[[str], Any], format_instructions: str
    ) -> Any:
        text = result.text
        try:
            return parse(text)
        except parsing.ParseError as first_error:
            logger.warning("Parse failed (%s); trying one repair call", first_error)
            repaired = await self.claude.complete(
                model=self.settings.parser_model,
                effort=None,
                system_text=prompts.REPAIR_SYSTEM_PROMPT,
                shared_text=prompts.repair_prompt(text, format_instructions),
                max_tokens=800,
                purpose="answer repair",
            )
            if repaired.text.strip().upper().startswith("NONE"):
                raise parsing.ParseError(
                    f"No usable final answer in the write-up ({first_error})"
                ) from first_error
            value = parse(repaired.text)
            answer_lines = [ln.strip() for ln in repaired.text.splitlines() if ln.strip()]
            restated = " / ".join(answer_lines[-len(format_instructions.splitlines()):])
            result.note = (
                "(This write-up did not end in the required format. Its final answer "
                f"was restated by {self.settings.parser_model} as: {restated})"
            )
            return value

    @staticmethod
    def _rationale(result: LlmResult, member: EnsembleMember | None) -> str:
        who = f"{member.model} (effort {member.effort})" if member else result.model
        note = f"\n\n{result.note}" if result.note else ""
        return f"Model: {who}\n\n{result.text}{note}"

    # ------------------------------------------------------ question types
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        result = await self._ask(question, prompts.binary_prompt(question, research))
        value = await self._parse_with_repair(result, parsing.parse_binary, BINARY_FORMAT)
        return ReasonedPrediction(
            prediction_value=value, reasoning=self._rationale(result, _current_member.get())
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        result = await self._ask(question, prompts.multiple_choice_prompt(question, research))
        mc_format = "\n".join(f"{o}: PP%" for o in question.options)
        probs = await self._parse_with_repair(
            result,
            lambda t: parsing.parse_multiple_choice(t, list(question.options)),
            mc_format,
        )
        option_list = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=p) for name, p in probs.items()
            ]
        )
        return ReasonedPrediction(
            prediction_value=option_list,
            reasoning=self._rationale(result, _current_member.get()),
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        result = await self._ask(question, prompts.numeric_prompt(question, research))
        pairs = await self._parse_with_repair(
            result, parsing.parse_numeric_percentiles, PERCENTILE_FORMAT
        )
        distribution = NumericDistribution.from_question(
            [Percentile(percentile=p, value=v) for p, v in pairs], question
        )
        return ReasonedPrediction(
            prediction_value=distribution,
            reasoning=self._rationale(result, _current_member.get()),
        )

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        result = await self._ask(question, prompts.date_prompt(question, research))
        pairs = await self._parse_with_repair(
            result, parsing.parse_date_percentiles, DATE_FORMAT
        )
        distribution = NumericDistribution.from_question(
            [Percentile(percentile=p, value=d.timestamp()) for p, d in pairs], question
        )
        return ReasonedPrediction(
            prediction_value=distribution,
            reasoning=self._rationale(result, _current_member.get()),
        )

    # Conditional questions: unchanged from the Metaculus FallTemplateBot2026.
    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(reasoning=full_reasoning, prediction_value=full_prediction)

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction, str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            now = datetime.now(timezone.utc)
            if previous_forecast.timestamp_end is None or previous_forecast.timestamp_end > now:
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return prediction, research
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self, research: str, reasoning: ReasonedPrediction, question_type: str
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    # --------------------------------------------------------------- comment
    def _create_unified_explanation(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: Any,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        base = super()._create_unified_explanation(
            question,
            research_prediction_collections,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )
        plan = _current_plan.get()
        n_ok = sum(len(c.predictions) for c in research_prediction_collections)
        if plan is None:
            return base
        combine = (
            "mean" if isinstance(question, MultipleChoiceQuestion) else "median"
        )
        sources = ", ".join(plan.sources_used) or "none available"
        method = (
            f"*Method*: {n_ok} of {len(plan.members)} independent Claude forecasts "
            f"({', '.join(m.label for m in plan.members)}), combined by {combine}. "
            f"Research sources: {sources}. Fully automated; no human in the loop.\n"
        )
        # Keep "# SUMMARY" as the first line: forecasting-tools requires the
        # explanation to start with a heading and reads section 0 as the summary.
        heading, _, rest = base.lstrip().partition("\n")
        rest = "\n".join(
            line for line in rest.split("\n") if "extra_metadata_in_explanation" not in line
        )
        return f"{heading}\n{method}{rest}"
