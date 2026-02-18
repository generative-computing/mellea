"""Generation pipeline hook payloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mellea.plugins.base import MelleaBasePayload


class GenerationPreCallPayload(MelleaBasePayload):
    """Payload for ``generation_pre_call`` — before LLM backend call."""

    action: Any = None  # Component | CBlock
    context: Any = None  # Context
    formatted_prompt: str | list[dict[str, Any]] = ""
    model_options: dict[str, Any] = {}
    tools: dict[str, Callable] | None = None
    format: Any = None  # type | None


class GenerationPostCallPayload(MelleaBasePayload):
    """Payload for ``generation_post_call`` — after LLM response received."""

    prompt: str | list[dict[str, Any]] = ""
    raw_response: dict[str, Any] = {}
    processed_output: str = ""
    model_output: Any = None  # ModelOutputThunk
    token_usage: dict[str, Any] | None = None
    latency_ms: int = 0
    finish_reason: str = ""


class GenerationStreamChunkPayload(MelleaBasePayload):
    """Payload for ``generation_stream_chunk`` — for each streaming chunk."""

    chunk: str = ""
    accumulated: str = ""
    chunk_index: int = 0
    is_final: bool = False
