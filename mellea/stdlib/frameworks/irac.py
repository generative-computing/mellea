"""IRAC (Issue, Rule, Analysis, Conclusion) legal reasoning framework."""

import asyncio
import json
from typing import Literal, TypeAlias, TypedDict

import mellea.stdlib.functional as mfuncs
from mellea.core import (
    Backend,
    CBlock,
    Component,
    Context,
    ModelOutputThunk,
    Requirement,
    TemplateRepresentation,
    ValidationResult,
)
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.components.chat import Message

Issue: TypeAlias = CBlock | ModelOutputThunk
RuleType: TypeAlias = Literal["statute", "regulation", "caselaw"]
Analysis: TypeAlias = CBlock | ModelOutputThunk
IRACAnswer: TypeAlias = CBlock | ModelOutputThunk


def _model_output_json_loads(x: str) -> dict:
    """This function can parse either actual JSON, or JSON surrounded by a docstring that looks like: ```json.*```."""
    if "```json" in x:
        # If we start and end with ```json ... ``` then return the entire middle.
        if x.startswith("```json") and x.endswith("```"):
            x_parsed = x[len("```json") : -len("```")]
            return json.loads(x_parsed)
        # Sometimes GPT-OSS-20B will start with ```json and not finish
        elif x.startswith("```json") and x.count("```") == 1:
            x_parsed = x[len("```json") :]
            return json.loads(x_parsed)
        # Sometimes we have the pattern ```json ... ``` don't start with ```json. Handle that case.
        elif x.count("```") > 1:
            x_parsed = x[x.find("```json") + len("```json") :]
            x_parsed = x_parsed.split("```")[0]
            return json.loads(x_parsed)
    return json.loads(x)


class Rule(SimpleComponent):
    """A legal rule."""

    def __init__(
        self,
        citation: str,
        rule_type: RuleType,
        rule_contents: str | None = None,
        summary: str | None = None,
    ):
        """Initialize a legal rule with citation and content."""
        super().__init__(
            citation=citation,
            rule_type=rule_type,
            rule_contents=rule_contents,
            summary=summary,
        )


class IRACQuery(Component):
    """Query component for IRAC analysis."""

    def __init__(
        self,
        scenario: str | CBlock,
        issue: Issue | None = None,
        rules: list[Rule] | None = None,
        analysis: Analysis | None = None,
        conclusion: IRACAnswer | None = None,
    ):
        """Initialize an IRAC query with scenario and optional analysis components."""
        match scenario:
            case str():
                self.scenario = CBlock(scenario)
            case CBlock():
                self.scenario = scenario
            case _:
                raise TypeError()
        self.issue = issue
        self.rules = rules
        self.analysis = analysis
        self.conclusion = conclusion

    def _parse(self, computed: ModelOutputThunk):
        return computed.value

    def parts(self):
        """Return list of non-None components in this query."""
        _parts = [self.scenario, self.issue, self.analysis, self.conclusion]
        if self.rules is not None:
            _parts.extend(self.rules)
        return [part for part in _parts if part is not None]

    def format_for_llm(self):
        """Format this query for LLM consumption."""
        return TemplateRepresentation(
            obj=self,
            template_order=["*", "IRACQuery"],
            args={
                "scenario": self.scenario,
                "issue": self.issue,
                "rules": self.rules,
                "analysis": self.analysis,
                "conclusion": self.conclusion,
            },
        )


async def identify_issue(ctx: Context, backend: Backend, scenario: CBlock) -> Issue:
    """Identify the legal issue from a given scenario."""
    query = IRACQuery(scenario=scenario)
    issue_mot, _ = await mfuncs.aact(query, ctx, backend)
    return issue_mot


# region rule discovery


class RuleFormatRequirement(Requirement):
    """Requirement for validating rule format from model output."""

    def validate(self, backend, ctx, *, format=None, model_options=None):
        """Validate that model output contains properly formatted rules."""
        output = ctx.last_output()
        parsed_rules = _model_output_json_loads(output.value)
        try:
            _rules = [Rule(rule) for rule in parsed_rules]
            return ValidationResult(result=True)
        except Exception as e:
            msg = f"{e}\n\nRules were: {output.value}"
            return ValidationResult(result=False, reason=msg)


async def discover_rule_candidates(
    ctx: Context, backend: Backend, scenario: CBlock, issue: Issue
) -> list[Rule]:
    """Find all possibly relevant rules."""
    # Simple prompt for now, but this should be expanded to a RAG pipeline.
    rule_cites_mot, _ = await mfuncs.aact(
        IRACQuery(scenario, issue), ctx, backend
    )  # , requirements=[RuleFormatRequirement("", check_only=True)])
    parsed_rules = _model_output_json_loads(str(rule_cites_mot.value))
    rules = [Rule(**rule) for rule in parsed_rules]
    return rules


# endregion


async def _construct_rule_subsets(
    ctx: Context, backend: Backend, scenario: CBlock, issue: Issue, rules: list[Rule]
) -> list[list[Rule]]:
    """Group similar rules by category."""
    raise Exception("unimplemented.")


async def analyze_issue_using_rules(
    ctx: Context, backend: Backend, scenario: CBlock, issue: Issue, rules: list[Rule]
) -> Analysis:
    """Analyze the issue in terms of the rules."""
    query = IRACQuery(scenario=scenario, issue=issue, rules=rules)
    analysis, _ = await mfuncs.aact(query, ctx, backend)
    return analysis


async def reach_conclusion(
    ctx: Context,
    backend: Backend,
    scenario: CBlock,
    issue: Issue,
    rules: list[Rule],
    analysis: Analysis,
) -> ModelOutputThunk:
    """Reach a legal conclusion based on issue, rules, and analysis."""
    query = IRACQuery(scenario=scenario, issue=issue, rules=rules, analysis=analysis)
    conclusion, _ = await mfuncs.aact(query, ctx, backend)
    return conclusion


async def summarize_irac_finding(
    ctx: Context,
    backend: Backend,
    scenario: CBlock,
    issue: Issue,
    rules: list[Rule],
    analysis: Analysis,
    conclusion: ModelOutputThunk,
):
    """Summarize the complete IRAC finding."""
    query = IRACQuery(
        scenario=scenario,
        issue=issue,
        rules=rules,
        analysis=analysis,
        conclusion=conclusion,
    )
    summary, _ = await mfuncs.aact(query, ctx, backend)
    return summary


async def summarize_irac_findings(
    conclusions: list[tuple[str, Issue, list[Rule], Analysis]],
) -> IRACAnswer:
    """Construct an IRACAnswer object and add a generated summary."""
    # use the better of the two functions below.


async def summarize_irac_findings_using_summaries(
    summaries: list[ModelOutputThunk],
) -> IRACAnswer:
    """Construct an IRACAnswer object and add a generated summary."""


async def summarize_irac_findings_using_everything(
    ctx: Context,
    backend: Backend,
    scenario: CBlock,
    issue: Issue,
    rules: list[list[Rule]],
    analyses: list[Analysis],
    conclusions: list[ModelOutputThunk],
) -> IRACAnswer:
    """Construct an IRACAnswer object and add a generated summary."""

    class IRACFinalQuery(Component):
        def __init__(
            self,
            scenario: CBlock,
            issue: Issue,
            rules: list[list[Rule]],
            analyses: list[Analysis],
            conclusions: list[ModelOutputThunk],
            summaries: list[ModelOutputThunk],
        ):
            self.scenario = scenario
            self.issue = issue
            self.rules = rules
            self.analyses = analyses
            self.conclusions = conclusions
            self.summaries = summaries

        def _parse(self, computed: ModelOutputThunk):
            return computed.value

        def parts(self):
            _parts = [self.scenario, self.issue, self.analysis, self.conclusion]
            if self.rules is not None:
                _parts.extend(self.rules)
            return _parts

        def format_for_llm(self):
            return TemplateRepresentation(
                obj=self,
                template_order=["*", "IRACFinalQuery"],
                args={
                    "scenario": self.scenario,
                    "issue": self.issue,
                    "rules": self.rules,
                    "analyses": self.analyses,
                    "conclusions": self.conclusions,
                    "summaries": self.summaries,
                },
            )

    final_answer, _ = await mfuncs.aact(
        IRACFinalQuery(
            scenario=scenario,
            issue=issue,
            rules=rules,
            analyses=analyses,
            conclusions=conclusions,
            summaries=[],
        ),
        ctx,
        backend,
    )
    return final_answer


async def irac(
    ctx: Context, backend: Backend, scenario: str | CBlock
) -> tuple[Issue, list[Rule], Analysis, ModelOutputThunk, ModelOutputThunk]:
    """Performs an [IRAC](https://en.wikipedia.org/wiki/IRAC) analysis, step-by-step."""
    scenario = CBlock(scenario) if isinstance(scenario, str) else scenario

    issue = await identify_issue(ctx, backend, scenario)
    rules = await discover_rule_candidates(ctx, backend, scenario, issue)
    analysis = await analyze_issue_using_rules(ctx, backend, scenario, issue, rules)
    conclusion = await reach_conclusion(ctx, backend, scenario, issue, rules, analysis)
    summary = await summarize_irac_finding(
        ctx, backend, scenario, issue, rules, analysis, conclusion
    )
    return issue, rules, analysis, conclusion, summary
