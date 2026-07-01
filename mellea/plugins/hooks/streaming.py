"""Streaming pipeline hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class StreamingStartPayload(MelleaBasePayload):
    """Payload for `streaming_start` — before a `stream_with_chunking` run starts.

    Attributes:
        streaming_id: UUID correlating start/event/end hooks for a single
            `stream_with_chunking` run.
        has_requirements: `True` when the orchestrator was given at least one
            `Requirement` to validate against.
        requirement_count: Number of `Requirement` instances supplied.
        chunking_strategy: Class name of the resolved `ChunkingStrategy`.
    """

    streaming_id: str = ""
    has_requirements: bool = False
    requirement_count: int = 0
    chunking_strategy: str = ""


class StreamingEventPayload(MelleaBasePayload):
    """Payload for `streaming_event` — fired once per `StreamEvent`.

    Attributes:
        streaming_id: UUID correlating with the matching `streaming_start`.
        event: The `StreamEvent` subclass instance for this event.
        requirements: For a `QuickCheckEvent`, the active `Requirement` instances
            corresponding to the event's results, in the same order; empty for
            other event types.
    """

    streaming_id: str = ""
    event: Any = None
    requirements: list[Any] = []


class StreamingEndPayload(MelleaBasePayload):
    """Payload for `streaming_end` — when `stream_with_chunking` finishes.

    Fires on every completing path: natural completion, validation-fail
    early-exit, and an unhandled exception. `success` and `exception`
    together distinguish the outcomes.

    Attributes:
        streaming_id: UUID correlating with the matching `streaming_start`.
        success: `True` only when the run completed with no streaming
            validation failure and no exception.
        failure_reason: Human-readable reason when `success` is `False` and no
            exception was raised (validation-fail early-exit).
        exception: The exception raised by the orchestrator, when one was.
        model: Model identifier from the underlying generation, when known.
        provider: Provider name from the underlying generation, when known.
        full_text_length: Length of the accumulated text at orchestrator exit.
    """

    streaming_id: str = ""
    success: bool = True
    failure_reason: str | None = None
    exception: BaseException | None = None
    model: str | None = None
    provider: str | None = None
    full_text_length: int = 0


class StreamingOrchestrationStartPayload(MelleaBasePayload):
    """Payload for `streaming_orchestration_start` — on the orchestration task, before the stream is drained.

    Attributes:
        streaming_id: UUID correlating with the matching `streaming_start`.
    """

    streaming_id: str = ""


class StreamingOrchestrationEndPayload(MelleaBasePayload):
    """Payload for `streaming_orchestration_end` — on the orchestration task, after the stream is drained.

    Fires on the same task as `streaming_orchestration_start`.

    Attributes:
        streaming_id: UUID correlating with the matching
            `streaming_orchestration_start`.
    """

    streaming_id: str = ""
