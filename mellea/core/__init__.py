"""Core abstractions for the mellea library.

This package defines the fundamental interfaces and data structures on which every
other layer of mellea is built: the ``Backend``, ``Formatter``, and
``SamplingStrategy`` protocols; the ``Component``, ``CBlock``, ``Context``, and
``ModelOutputThunk`` data types that flow through the inference pipeline; and
``Requirement`` / ``ValidationResult`` for constrained generation. Start here when
building a new backend, formatter, or sampling strategy, or when you need the type
definitions shared across the library.
"""

from .backend import Backend, BaseModelSubclass, generate_walk
from .base import (
    C,
    CBlock,
    Component,
    ComponentParseError,
    ComputedModelOutputThunk,
    Context,
    ContextTurn,
    GenerateLog,
    GenerateType,
    ImageBlock,
    ModelOutputThunk,
    ModelToolCall,
    S,
    TemplateRepresentation,
    blockify,
)
from .formatter import Formatter
from .requirement import Requirement, ValidationResult, default_output_to_bool
from .sampling import SamplingResult, SamplingStrategy
from .utils import MelleaLogger, clear_log_context, log_context, set_log_context

__all__ = [
    "Backend",
    "BaseModelSubclass",
    "C",
    "CBlock",
    "Component",
    "ComponentParseError",
    "ComputedModelOutputThunk",
    "Context",
    "ContextTurn",
    "Formatter",
    "GenerateLog",
    "GenerateType",
    "ImageBlock",
    "MelleaLogger",
    "ModelOutputThunk",
    "ModelToolCall",
    "Requirement",
    "S",
    "SamplingResult",
    "SamplingStrategy",
    "TemplateRepresentation",
    "ValidationResult",
    "blockify",
    "clear_log_context",
    "default_output_to_bool",
    "generate_walk",
    "log_context",
    "set_log_context",
]
