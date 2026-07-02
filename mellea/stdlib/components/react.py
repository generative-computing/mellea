"""Components that implement the ReACT (Reason + Act) agentic pattern.

Provides `ReactInitiator`, which primes the model with a goal and a tool list, and
`ReactThought`, which signals a thinking step. Also exports the
`MELLEA_FINALIZER_TOOL` sentinel string used to signal loop termination. These
components are consumed by `mellea.stdlib.frameworks.react`, which orchestrates the
reasoning-acting cycle until the model invokes `final_answer` or the step budget
is exhausted.
"""

import inspect
from typing import Generic

from mellea.backends.tools import MelleaTool
from mellea.core.backend import BaseModelSubclass
from mellea.core.base import (
    AbstractMelleaTool,
    CBlock,
    Component,
    ModelOutputThunk,
    TemplateRepresentation,
)
from mellea.core.utils import MelleaLogger

MELLEA_FINALIZER_TOOL = "final_answer"
"""Used in the react loop to symbolize the loop is done."""


# Note: must leave answer type as str. Otherwise, must set it during the format reconfiguration done to the tool in format_for_llm.
def _mellea_finalize_tool(answer: str) -> str:
    """Finalizer function that signals the end of the react loop and takes the final answer."""
    return answer


def pin_react_initiator(components: list[Component | CBlock]) -> int:
    """A ``PinPredicate`` that pins everything up to and including the first ``ReactInitiator``.

    Plug it into any compactor in :mod:`mellea.stdlib.context` that takes a
    ``pin_predicate`` (e.g. :class:`WindowCompactor`,
    :class:`ThresholdCompactor`'s inner compactor) so the react goal and
    tool registration survive compaction:

        from mellea.stdlib.context import ChatContext, WindowCompactor
        from mellea.stdlib.components.react import pin_react_initiator

        ctx = ChatContext(
            compactor=WindowCompactor(size=5, pin_predicate=pin_react_initiator),
        )
        result, _ = await react(goal=..., context=ctx, ...)

    Returns ``0`` when no ``ReactInitiator`` is found, so a context that
    has not yet been seeded with a react goal compacts as if there were
    no prefix.
    """
    for i, c in enumerate(components):
        if isinstance(c, ReactInitiator):
            return i + 1
    return 0


def react_summary_prompt(
    goal: str | None = None, max_tokens_hint: int | None = None
) -> str:
    """Build a research-flavoured summary prompt for :class:`LLMSummarizeCompactor`.

    Returns a template with a ``{conversation}`` placeholder that
    :class:`LLMSummarizeCompactor` fills in at compaction time. Pass the
    react goal via ``goal=`` to anchor the summarisation around the
    objective; with ``goal=None`` the ``GOAL:`` line is omitted.

    Pass ``max_tokens_hint=N`` to inject a soft length-cap bullet
    ("Be at most ~N tokens") into the summarizer's instructions. The hint
    is a plan-time anchor for the model — combine it with a hard
    ``max_tokens`` API arg on the summarizer's LLM call to enforce.
    ``max_tokens_hint=None`` (default) or non-positive values omit the
    bullet, so the prompt is byte-identical to the un-hinted form.

    Curly braces in ``goal`` are escaped so :meth:`str.format` (used by the
    compactor) preserves them as literal characters.

    Example::

        from mellea.stdlib.components.react import (
            pin_react_initiator,
            react_summary_prompt,
        )
        from mellea.stdlib.context import LLMSummarizeCompactor

        compactor = LLMSummarizeCompactor(
            default_backend=my_backend,
            keep_n=5,
            pin_predicate=pin_react_initiator,
            prompt_template=react_summary_prompt(
                goal="find papers on X",
                max_tokens_hint=2000,
            ),
        )
    """
    if goal is not None:
        # Escape braces so .format() in the compactor keeps them literal.
        safe_goal = goal.replace("{", "{{").replace("}", "}}")
        goal_block = f"GOAL: {safe_goal}\n\n"
    else:
        goal_block = ""
    if max_tokens_hint is not None and max_tokens_hint > 0:
        # Rough heuristic: ~0.75 words per token for English research text.
        words_estimate = int(max_tokens_hint * 0.75)
        length_bullet = (
            f"- Be at most ~{max_tokens_hint} tokens (roughly "
            f"{words_estimate} words). Prioritize density: drop redundant "
            "or ancillary detail.\n"
        )
    else:
        length_bullet = ""
    return (
        "You are summarizing research progress to maintain context "
        "within token limits.\n\n"
        f"{goal_block}"
        "Provide a comprehensive summary of the research context below. "
        "Your summary should:\n"
        "- Preserve ALL specific facts, numbers, names, URLs, and search "
        "queries found\n"
        "- Note which tools were called and what results were obtained\n"
        "- Highlight key findings and any dead ends encountered\n"
        "- Be structured clearly so the research can continue seamlessly\n"
        f"{length_bullet}"
        "\nContext to summarize:\n{conversation}"
    )


class ReactInitiator(Component[str]):
    """`ReactInitiator` is used at the start of the ReACT loop to prime the model.

    Args:
        goal (str): The objective of the react loop.
        tools (list[AbstractMelleaTool] | None): Tools available to the agent.
            `None` is treated as an empty list.

    Attributes:
        goal (CBlock): The objective of the react loop wrapped as a content block.
        tools (list[AbstractMelleaTool]): The tools made available to the react agent.
    """

    def __init__(self, goal: str, tools: list[AbstractMelleaTool] | None):
        """Initialize ReactInitiator with a goal string and optional list of available tools."""
        self.goal = CBlock(goal)
        self.tools = tools or []

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return the constituent parts of this component.

        Returns:
            list[Component | CBlock | ModelOutputThunk]: A list containing the goal content block.
        """
        return [self.goal]

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the `Component` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` whose tools always includes a finalizer tool.
        """
        tools = {tool.name: tool for tool in self.tools}

        if tools.get(MELLEA_FINALIZER_TOOL, None) is not None:
            MelleaLogger.get_logger().warning(
                f"overriding user tool '{MELLEA_FINALIZER_TOOL}' in react call; this tool name is required for internal use"
            )

        finalizer_tool = MelleaTool.from_callable(
            _mellea_finalize_tool, MELLEA_FINALIZER_TOOL
        )
        tools[MELLEA_FINALIZER_TOOL] = finalizer_tool

        return TemplateRepresentation(
            obj=self,
            args={
                "goal": self.goal,
                "finalizer_tool_name": tools[MELLEA_FINALIZER_TOOL].name,
            },
            tools=tools,
            template_order=["*", "ReactInitiator"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Returns the value of the ModelOutputThunk unchanged."""
        return computed.value if computed.value is not None else ""


class ReactThought(Component[str]):
    """ReactThought signals that a thinking step should be done."""

    def __init__(self):
        """ReactThought signals that a thinking step should be done."""

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Return the constituent parts of this component.

        `ReactThought` has no sub-components; it solely triggers a thinking step.

        Returns:
            list[Component | CBlock | ModelOutputThunk]: Always an empty list.
        """
        return []

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Returns the value of the ModelOutputThunk unchanged."""
        return computed.value if computed.value is not None else ""

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the `Component` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` whose tools always includes a finalizer tool.
        """
        return TemplateRepresentation(
            obj=self, args={}, template_order=["*", "ReactThought"]
        )
