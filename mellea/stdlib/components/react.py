"""Components used for ReACT."""

from collections.abc import Callable
from typing import Generic

from mellea.core.backend import BaseModelSubclass
from mellea.core.base import CBlock, Component, ModelOutputThunk, TemplateRepresentation
from mellea.stdlib.components.chat import Message

# TODO: JAL. Change this to a nice name like finalize_answer
MELLEA_FINALIZER_TOOL = "__mellea_finalize_tool"
"""Used in the react loop to symbolize the loop is done."""

class ReactInitiator(Component[str], Generic[BaseModelSubclass]):
    """`ReactInitiator` is used at the start of the ReACT loop to prime the model."""

    def __init__(
            self,
            goal: str,
            tools: list[Callable] | None, # TODO: JAL. change this to the new tool class.
            # role: Message.Role = "system", # TODO: JAL. Once components support this, add this back. Ensure this works properly with tools, etc...
            format: type[BaseModelSubclass] | None = None
    ):
        """`ReactInitiator` is used at the start of the ReACT loop to prime the model.

        Args:
            goal: the objective of the react loop
            tools: a list of tools that are available to the react agent
            format: the format/schema of the expected answer
        """
        self.goal = CBlock(goal)
        self.tools = tools or []
        self.format = format

    def parts(self) -> list[Component | CBlock]:
        """The set of all the constituent parts of the `Component`."""
        return [self.goal]

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the `Component` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` whose tools always includes a finalizer tool.
        """
        tools = {
            tool.__name__: tool for tool in self.tools
        }

        # TODO: JAL. Make the naming nicer?
        if self.format is None:
            def __mellea_finalize_tool(input: str) -> str:  # type: ignore[reportRedeclaration]
                """Finalizer function that signals the end of the react loop."""
                return input
        else:
            def __mellea_finalize_tool(input: BaseModelSubclass) -> BaseModelSubclass:
                """Finalizer function that signals the end of the react loop."""
                return input

        tools[MELLEA_FINALIZER_TOOL] = __mellea_finalize_tool

        return TemplateRepresentation(
            obj=self,
            args={
                "goal": self.goal,
                "finalizer_tool_name": tools[MELLEA_FINALIZER_TOOL].__name__
            },
            tools=tools,
            template_order=["*", "ReactInitiator"]
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Returns the value of the ModelOutputThunk unchanged."""
        return computed.value if computed.value is not None else ""

class ReactThought(Component[str]):
    """ReactThought signals that a thinking step should be done."""

    def __init__(self):
        """ReactThought signals that a thinking step should be done."""

    def parts(self) -> list[Component | CBlock]:
        """Component has no parts."""
        return []

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Returns the value of the ModelOutputThunk unchanged."""
        return computed.value if computed.value is not None else ""

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the `Component` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` whose tools always includes a finalizer tool.
        """
        return TemplateRepresentation(
            obj=self,
            args={},
            template_order=["*", "ReactThought"]
        )

class ReactAction(Component[str]):
    """ReactAction signals that an action should be taken."""

    def __init__(self):
        """ReactAction signals that an action should be taken."""

    def parts(self) -> list[Component | CBlock]:
        """Component has no parts."""
        return []

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Returns the value of the ModelOutputThunk unchanged."""
        return computed.value if computed.value is not None else ""

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the `Component` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` whose tools always includes a finalizer tool.
        """
        return TemplateRepresentation(
            obj=self,
            args={},
            template_order=["*", "ReactAction"]
        )
