"""Module for working with Requirements."""

from typing import TYPE_CHECKING

# Import from core for ergonomics.
from ...core import Requirement, ValidationResult, default_output_to_bool
from .md import as_markdown_list, is_markdown_list, is_markdown_table
from .python_reqs import PythonExecutionReq
from .requirement import (
    ALoraRequirement,
    LLMaJRequirement,
    check,
    req,
    reqify,
    requirement_check_to_bool,
    simple_validate,
)
from .tool_reqs import tool_arg_validator, uses_tool

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from .rag import HallucinationRequirement

__all__ = [
    "ALoraRequirement",
    "HallucinationRequirement",
    "LLMaJRequirement",
    "PythonExecutionReq",
    "Requirement",
    "ValidationResult",
    "as_markdown_list",
    "check",
    "default_output_to_bool",
    "is_markdown_list",
    "is_markdown_table",
    "req",
    "reqify",
    "requirement_check_to_bool",
    "simple_validate",
    "tool_arg_validator",
    "uses_tool",
]


def __getattr__(name: str):
    """Lazy import for HallucinationRequirement to avoid circular imports."""
    if name == "HallucinationRequirement":
        from .rag import HallucinationRequirement

        return HallucinationRequirement
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
