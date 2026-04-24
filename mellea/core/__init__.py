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
from .requirement import (
    PartialValidationResult,
    Requirement,
    ValidationResult,
    default_output_to_bool,
)
from .sampling import SamplingResult, SamplingStrategy
from .utils import FancyLogger

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
    "FancyLogger",
    "Formatter",
    "GenerateLog",
    "GenerateType",
    "ImageBlock",
    "ModelOutputThunk",
    "ModelToolCall",
    "PartialValidationResult",
    "Requirement",
    "S",
    "SamplingResult",
    "SamplingStrategy",
    "TemplateRepresentation",
    "ValidationResult",
    "blockify",
    "default_output_to_bool",
    "generate_walk",
]
