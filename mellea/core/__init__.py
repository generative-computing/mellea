# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core abstractions for the mellea library.

This package defines the fundamental interfaces and data structures on which every
other layer of mellea is built: the `Backend`, `Formatter`, and
`SamplingStrategy` protocols; the `Component`, `CBlock`, `Context`, and
`ModelOutputThunk` data types that flow through the inference pipeline; and
`Requirement` / `ValidationResult` / `PartialValidationResult` for constrained generation. Start here when
building a new backend, formatter, or sampling strategy, or when you need the type
definitions shared across the library.
"""

from .backend import Backend, BaseModelSubclass, generate_walk
from .base import (
    AudioBlock,
    AudioUrlBlock,
    C,
    CBlock,
    Component,
    ComponentParseError,
    ComputedModelOutputThunk,
    Context,
    ContextTurn,
    GenerateLog,
    GenerateType,
    GenerationMetadata,
    ImageBlock,
    ImageUrlBlock,
    ModelOutputThunk,
    ModelToolCall,
    RawProviderResponse,
    S,
    Span,
    TemplateRepresentation,
    blockify,
    get_audio_from_component,
    get_images_from_component,
    make_image_block,
)
from .chunking import (
    Chunker,
    ChunkingStrategy,
    ParagraphChunking,
    SentenceChunking,
    WordChunking,
    resolve_chunking_strategy,
)
from .formatter import Formatter
from .requirement import (
    PartialValidationResult,
    PartialValidationSummary,
    Requirement,
    ValidationResult,
    default_output_to_bool,
)
from .sampling import SampleActionType, SamplingResult, SamplingStrategy
from .utils import MelleaLogger, clear_log_context, log_context, set_log_context


def __getattr__(name: str) -> object:
    if name == "FancyLogger":
        import warnings

        warnings.warn(
            "FancyLogger has been renamed to MelleaLogger and will be removed in a future release. "
            "Update your imports to use mellea.core.MelleaLogger.",
            DeprecationWarning,
            stacklevel=2,
        )
        return MelleaLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AudioBlock",
    "AudioUrlBlock",
    "Backend",
    "BaseModelSubclass",
    "C",
    "CBlock",
    "Chunker",
    "ChunkingStrategy",
    "Component",
    "ComponentParseError",
    "ComputedModelOutputThunk",
    "Context",
    "ContextTurn",
    "Formatter",
    "GenerateLog",
    "GenerateType",
    "GenerationMetadata",
    "ImageBlock",
    "ImageUrlBlock",
    "MelleaLogger",
    "ModelOutputThunk",
    "ModelToolCall",
    "ParagraphChunking",
    "PartialValidationResult",
    "PartialValidationSummary",
    "RawProviderResponse",
    "Requirement",
    "S",
    "SampleActionType",
    "SamplingResult",
    "SamplingStrategy",
    "SentenceChunking",
    "Span",
    "TemplateRepresentation",
    "ValidationResult",
    "WordChunking",
    "blockify",
    "clear_log_context",
    "default_output_to_bool",
    "generate_walk",
    "get_audio_from_component",
    "get_images_from_component",
    "log_context",
    "make_image_block",
    "resolve_chunking_strategy",
    "set_log_context",
]
