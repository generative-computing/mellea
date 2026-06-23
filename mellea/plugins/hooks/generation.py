"""Generation pipeline hook payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mellea.plugins.base import MelleaBasePayload


class GenerationPreCallPayload(MelleaBasePayload):
    """Payload for `generation_pre_call` — before LLM backend call.

    Attributes:
        action: The `Component` or `CBlock` about to be sent to the backend.

        context: The `Context` being used for this generation call.

        model_options: Dict of model options (writable — plugins may adjust temperature, etc.).
        format: Optional `BaseModel` subclass for constrained decoding (writable).
        tool_calls: Whether tool calls are enabled for this generation (writable).
        generation_id: Mellea-side hook correlation ID, distinct from the
            provider-assigned `GenerationMetadata.response_id`. `None` when
            the firing site does not generate one.
    """

    action: Any = None
    context: Any = None
    model_options: dict[str, Any] = {}
    format: Any = None
    tool_calls: bool = False
    generation_id: str | None = None


class GenerationPostCallPayload(MelleaBasePayload):
    """Payload for `generation_post_call` — fires once the model output is fully computed.

    For lazy `ModelOutputThunk` objects this hook fires inside
    `ModelOutputThunk.astream` after `post_process` completes, so
    `model_output.value` is guaranteed to be available. For already-computed
    thunks (e.g. cached responses) it fires before `generate_from_context`
    returns.

    Attributes:
        prompt: The formatted prompt sent to the backend (str or list of message dicts).
        model_output: The fully-computed `ModelOutputThunk`.
        latency_ms: Elapsed milliseconds from the `generate_from_context` call
            to when the value was fully materialized.
        generation_id: Mellea-side hook correlation ID matching the
            corresponding pre_call payload, distinct from the provider-assigned
            `GenerationMetadata.response_id`. `None` when the firing site did
            not generate one.
    """

    prompt: str | list[dict[str, Any]] = ""
    model_output: Any = None
    latency_ms: float = 0.0
    generation_id: str | None = None


class GenerationErrorPayload(MelleaBasePayload):
    """Payload for `generation_error` — fires when the LLM backend raises an exception.

    This hook fires inside `ModelOutputThunk.astream` just before the exception
    is re-raised, giving plugins a chance to observe (but not suppress) the error.

    Attributes:
        exception: The exception raised by the backend.
        model_output: The `ModelOutputThunk` at the time of the error. `model`
            and `provider` are set when the backend set them early (before the
            async task); otherwise they are `None`.
        generation_id: Mellea-side hook correlation ID matching the
            corresponding pre_call payload, distinct from the provider-assigned
            `GenerationMetadata.response_id`. `None` when the firing site did
            not generate one.
    """

    exception: BaseException
    model_output: Any = None
    generation_id: str | None = None


class GenerationBatchPreCallPayload(MelleaBasePayload):
    """Payload for `generation_batch_pre_call` — fires once before a batch generation request.

    Carries the action sequence being sent in the batch alongside batch-level
    fields (`model`, `provider`, `num_actions`) describing the single
    API call.

    Attributes:
        actions: The action sequence being sent in the batch.
        generation_id: Correlation identifier set by the firing backend; matches
            the corresponding `generation_batch_post_call` /
            `generation_batch_error` payloads for the same call.
        model_options: Dict of model options (writable — plugins may adjust temperature, etc.).
        format: Optional structured-output format applied to the batch (writable).
        tool_calls: Whether tool calling is enabled (writable; typically `False` for raw).
        num_actions: Convenience copy of `len(actions)`.
        model: Model identifier the backend is calling.
        provider: Provider name (e.g. `"openai"`, `"ollama"`).
    """

    actions: Sequence[Any] = ()
    generation_id: str | None = None
    model_options: dict[str, Any] = {}
    format: Any = None
    tool_calls: bool = False
    num_actions: int = 0
    model: str | None = None
    provider: str | None = None


class GenerationBatchPostCallPayload(MelleaBasePayload):
    """Payload for `generation_batch_post_call` — fires once after a batch generation succeeds.

    Carries the list of `ModelOutputThunk` instances produced by the batch
    alongside batch-level fields (`usage`, `model`, `provider`,
    `latency_ms`) describing the single API call.

    Attributes:
        generation_id: Correlation identifier from the matching pre_call.
        model_outputs: The list of `ModelOutputThunk` instances built from
            the API response, in batch order.
        usage: Aggregate token-usage dict (OpenAI-shape) for the whole batch.
        model: Model identifier from the call.
        provider: Provider name.
        latency_ms: Wall-clock duration of the API call in milliseconds.
    """

    generation_id: str | None = None
    model_outputs: list[Any] = []
    usage: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None
    latency_ms: float = 0.0


class GenerationBatchErrorPayload(MelleaBasePayload):
    """Payload for `generation_batch_error` — fires once when a batch generation request fails.

    Carries the exception alongside batch-level fields (`model`,
    `provider`, `latency_ms`) describing the failed API call. No
    `ModelOutputThunk` instances are present.

    Attributes:
        generation_id: Correlation identifier from the matching pre_call.
        exception: The exception raised by the backend.
        model: Model identifier from the call.
        provider: Provider name.
        latency_ms: Wall-clock time-until-error in milliseconds.
    """

    generation_id: str | None = None
    exception: BaseException
    model: str | None = None
    provider: str | None = None
    latency_ms: float = 0.0
