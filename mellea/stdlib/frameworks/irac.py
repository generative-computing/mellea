from mellea.core import Component, Backend, Context, ModelOutputThunk, CBlock, TemplateRepresentation
from mellea.stdlib.components import SimpleComponent
import mellea.stdlib.functional as mfuncs
from typing import Literal, TypedDict
import asyncio
import json

type Issue = CBlock | ModelOutputThunk
type RuleType = Literal["statute", "regulation", "caselaw"]
type Analysis = CBlock | ModelOutputThunk
type IRACAnswer = CBlock | ModelOutputThunk



def model_output_json_loads(x: str) -> dict:
    """This function can parse either actual JSON, or JSON surrounded by a docstring that looks like: ```json.*```."""
    if "```json" in x:
        # If we start and end with ```json ... ``` then return the entire middle.
        if x.startswith("```json") and x.endswith("```"):
            x_parsed = x[len("```json") : -len("```") ]
            return json.loads(x_parsed)
        # Sometimes GPT-OSS-20B will start with ```json and not finish
        elif x.startswith("```json") and x.count("```") == 1:
            x_parsed = x[len("```json") : ]
            return json.loads(x_parsed)
        # Sometimes we have the pattern ```json ... ``` don't start with ```json. Handle that case.
        elif x.count("```") > 1:
            x_parsed = x[x.find("```json") + len("```json") : ]
            x_parsed = x_parsed.split("```")[0]
            return json.loads(x_parsed)
    return json.loads(x)


class Rule(TypedDict):
    """A legal rule."""
    citation: str
    rule_type: RuleType
    rule_contents: str | None
    rule_summary : str | None


class IRACQuery(Component):
    def __init__(self, scenario: str | CBlock, issue: Issue | None = None, rules: list[Rule] | None = None, analysis: Analysis | None = None, conclusion: IRACAnswer | None = None):
        match scenario:
            case str():
                self.scenario = CBlock(scenario)
            case CBlock():
                self.scenarioq = scenario
            case _:
                raise TypeError()
        self.issue = issue
        self.rules = rules
        self.analysis = analysis
        self.conclusion = conclusion
    
    def _parse(self, computed: ModelOutputThunk):
        return computed.value        
    
    def parts(self):
        _parts = [self.scenario, self.issue, self.rules, self.analysis, self.conclusion]
        return [part for part in _parts if part is not None]
    
    def format_for_llm(self):
        return TemplateRepresentation(
            obj=self,
            template_order=["*", "IRACQuery"],
            args={
                "scenario": self.scenario,
                "issue": self.issue,
                "rules": self.rules,
                "analysis": self.analysis,
                "conclusion": self.conclusion
            }
        )


async def identifiy_issue(ctx: Context, backend: Backend, scenario: CBlock) -> Issue:
    query = IRACQuery(scenario=scenario)
    issue_mot, _ = await mfuncs.aact(query, ctx, backend)
    return issue_mot


async def discover_rule_candidates(ctx: Context, backend: Backend, scenario: CBlock, issue: Issue) -> list[Rule]:
    """Find all possibly relevant rules."""
    # Simple prompt for now, but this should be expanded to a RAG pipeline.
    rule_cites_mot, _ = await mfuncs.aact(IRACQuery(scenario, issue), ctx, backend)
    rules = model_output_json_loads(rule_cites_mot.value)
    return rules


async def construct_rule_subsets(ctx: Context, backend: Backend, scenario: CBlock, issue: Issue, rules: list[Rule]) -> list[list[Rule]]:
    """Group similar rules by category."""
    return [[rule] for rule in rules] # TODO-nrf: do pairwise relevance checks or something.


async def analyze_issue_using_rules(ctx: Context, backend: Backend, scenario: CBlock, issue: Issue, rules: list[Rule]) -> Analysis:
    """Analyze the issue in terms of the rules."""
    query = IRACQuery(scenario=scenario, issue=issue, rules=rules)
    analysis = await mfuncs.aact(query, ctx, backend)


async def reach_conclusion(ctx: Context, backend: Backend, scenario: CBlock, issue: Issue, rules: list[Rule], analysis: Analysis):
    ...


async def summarize_irac_findings(conclusions: list[tuple[str, Issue, list[Rule], Analysis]]) -> IRACAnswer:
    """Construct an IRACAnswer object and add a generated summary."""
    ...


async def irac(ctx: Context, backend: Backend, scenario: str | CBlock) -> IRACAnswer:
    scenario = CBlock(scenario) if isinstance(scenario, str) else scenario

    issue = identifiy_issue(ctx, backend, scenario)
    rule_candidates = discover_rule_candidates(ctx, backend, scenario, issue)
    rule_subsets = construct_rule_subsets(ctx, backend, scenario, issue, rule_candidates)
    conclusions = []
    for subset in rule_subsets:
        analysis = analyze_issue_using_rules(ctx, backend, scenario, issue, subset)
        conclusion = reach_conclusion(ctx, backend, scenario, issue, subset, analysis)
        conclusions.append((scenario, issue, subset, analysis, conclusion))
    final_answer = summarize_irac_findings(conclusions)
    return final_answer
