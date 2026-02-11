"""Convert LangChain tools to Mellea format."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from mellea.backends.tools import MelleaTool

LANGCHAIN_AVAILABLE = False
BaseTool = None
MelleaTool = None

try:
    from langchain_core.tools import BaseTool as _BaseTool  # type: ignore  # noqa: F401
    from mellea.backends.tools import MelleaTool as _MelleaTool  # type: ignore

    BaseTool = _BaseTool
    MelleaTool = _MelleaTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


def langchain_to_mellea_tools(tools: list[Any]) -> list[Any]:
    """Convert LangChain tools to Mellea tool format.

    Args:
        tools: List of LangChain tools or Mellea tools

    Returns:
        List of Mellea-compatible tool objects

    Note:
        This function uses Mellea's built-in MelleaTool.from_langchain()
        method to convert LangChain tools. If a tool is already a Mellea
        tool, it is passed through unchanged.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not installed. Please install it with: pip install langchain-core"
        )

    mellea_tools = []

    for tool in tools:
        # Check if it's a LangChain tool by checking for the class name
        if (
            hasattr(tool, "__class__")
            and tool.__class__.__name__ == "BaseTool"
            or (
                hasattr(tool.__class__, "__mro__")
                and any(c.__name__ == "BaseTool" for c in tool.__class__.__mro__)
            )
        ):
            # Convert LangChain tool to Mellea format using built-in converter
            mellea_tool = MelleaTool.from_langchain(tool)  # type: ignore
            mellea_tools.append(mellea_tool)
        else:
            # Assume it's already a Mellea tool or compatible format
            mellea_tools.append(tool)

    return mellea_tools
