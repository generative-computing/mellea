"""Generation pipeline hook payloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mellea.plugins.base import MelleaBasePayload


class GenerationPreCallPayload(MelleaBasePayload):
    """Payload for ``generation_pre_call`` — before LLM backend call."""

    action: Any = None  # Component | CBlock
    context: Any = None  # Context
    model_options: dict[str, Any] = {}
    tools: dict[str, Callable] | None = None
    format: Any = None  # type | None


class GenerationPostCallPayload(MelleaBasePayload):
    """Payload for ``generation_post_call`` — after LLM response received."""

    prompt: str | list[dict[str, Any]] = ""
    model_output: Any = None  # ModelOutputThunk
    latency_ms: int = 0


class GenerationStreamChunkPayload(MelleaBasePayload):
    """Payload for ``generation_stream_chunk`` — for each streaming chunk."""

    chunk: str = ""
    accumulated: str = ""
    chunk_index: int = 0
    is_final: bool = False
