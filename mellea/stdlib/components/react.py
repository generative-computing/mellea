"""Components that implement the ReACT (Reason + Act) agentic pattern.

Provides ``ReactInitiator``, which primes the model with a goal and a tool list, and
``ReactThought``, which signals a thinking step. Also exports the
``MELLEA_FINALIZER_TOOL`` sentinel string used to signal loop termination. These
components are consumed by ``mellea.stdlib.frameworks.react``, which orchestrates the
reasoning-acting cycle until the model invokes ``final_answer`` or the step budget
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


class ReactInitiator(Component[str]):
    """`ReactInitiator` is used at the start of the ReACT loop to prime the model.

    Args:
        goal (str): The objective of the react loop.
        tools (list[AbstractMelleaTool] | None): Tools available to the agent.
            ``None`` is treated as an empty list.

    Attributes:
        goal (CBlock): The objective of the react loop wrapped as a content block.
        tools (list[AbstractMelleaTool]): The tools made available to the react agent.
    """

    def __init__(self, goal: str, tools: list[AbstractMelleaTool] | None):
        """Initialize ReactInitiator with a goal string and optional list of available tools."""
        self.goal = CBlock(goal)
        self.tools = tools or []

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this component.

        Returns:
            list[Component | CBlock]: A list containing the goal content block.
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
            _mellea_finalize_tool, MELLEA_FINALIZER_TOOL, is_internal=True
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

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this component.

        ``ReactThought`` has no sub-components; it solely triggers a thinking step.

        Returns:
            list[Component | CBlock]: Always an empty list.
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
