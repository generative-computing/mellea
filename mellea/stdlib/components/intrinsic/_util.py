# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for intrinsic convenience wrappers."""

from typing import cast

from ....backends import ModelOption
from ....backends._options import resolve_model_options
from ....backends.adapters import AdapterMixin, AdapterType
from ....core import Backend
from ....stdlib import functional as mfuncs
from ...components import Document
from ...context import ChatContext
from .intrinsic import Intrinsic


def _resolve_question(
    question: str | None, context: ChatContext, backend: Backend | None = None
) -> tuple[str, ChatContext]:
    """Return `(question_text, context_to_use)`.

    When *question* is not `None`, returns it with *context* unchanged.
    When `None`, extracts the text from the last turn's `model_input`
    and rewinds *context* to before that element.

    Supports `Message` (via `.content`), `CBlock` (via `.value`),
    and generic `Component` types (via `TemplateFormatter.print()`).
    """
    if question is not None:
        return question, context
    from ....core import CBlock, Component, ModelOutputThunk
    from ..chat import Message

    turn = context.last_turn()
    if turn is None or turn.model_input is None:
        raise ValueError(
            "question is None and context has no last turn with model input"
        )

    model_input = turn.model_input
    if isinstance(model_input, Message):
        text = model_input.content
    elif isinstance(model_input, (CBlock, ModelOutputThunk)):
        if model_input.value is None:
            raise ValueError(
                "question is None and last turn model_input CBlock has no value"
            )
        text = model_input.value
    elif isinstance(model_input, Component):
        formatter = getattr(backend, "formatter", None)
        if formatter is not None:
            text = formatter.print(model_input)
        else:
            from ....formatters import TemplateFormatter

            text = TemplateFormatter(model_id="default").print(model_input)
    else:
        raise ValueError(
            f"question is None but last turn model_input is "
            f"{type(model_input).__name__}, which is not a supported type"
        )

    rewound = context.previous_node
    if rewound is None:
        raise ValueError("Cannot rewind context past the root node")
    return text, rewound  # type: ignore[return-value]


def _extract_last_response(context: ChatContext) -> tuple[str, ChatContext]:
    """Extract the last assistant response text and the context preceding it.

    Returns `(response_text, prev_ctx)` where *prev_ctx* is *context* rewound
    to before the last assistant turn. Handles both session-generated contexts
    (last turn is a `ModelOutputThunk`) and manually-constructed contexts
    (last turn is an assistant `Message`).

    Args:
        context: Chat context whose last element is an assistant response.

    Returns:
        Tuple of the assistant response text and the rewound context.

    Raises:
        ValueError: If *context* is empty, if the last element is not an
            assistant response, if the response has not been computed yet,
            or if there is no preceding node.
    """
    from ....core import ModelOutputThunk
    from ..chat import Message

    turn = context.last_turn()
    if turn is None:
        raise ValueError("Context is empty; cannot extract an assistant response.")

    if turn.output is None:
        raise ValueError(
            "Cannot extract assistant response: the last context element is "
            "not an assistant response."
        )

    if isinstance(turn.output, ModelOutputThunk):
        # Session-generated response stored as a ModelOutputThunk.
        if turn.output.value is None:
            raise ValueError(
                "Cannot extract assistant response: it has not been computed yet. "
                "Await the response before calling this adapter function."
            )
        response_text: str = turn.output.value
    elif isinstance(turn.output, Message):
        # Manually-added assistant Message (e.g. built from test fixtures).
        response_text = turn.output.content
    else:
        raise ValueError(
            f"Cannot extract assistant response: output is of type "
            f"{type(turn.output).__name__}, not ModelOutputThunk or Message."
        )

    prev_ctx = context.previous_node
    if prev_ctx is None:
        raise ValueError(
            "Context has no previous node; cannot rewind past the assistant turn."
        )

    return response_text, cast(ChatContext, prev_ctx)


def _resolve_response(
    response: str | None, context: ChatContext
) -> tuple[str, ChatContext]:
    """Return `(response_text, context_to_use)`.

    When *response* is not `None`, returns it with *context* unchanged.
    When `None`, delegates to `_extract_last_response` to pull the
    text from the last assistant turn and rewind the context.
    """
    if response is not None:
        return response, context
    return _extract_last_response(context)


def _assert_context_forwards_history(intrinsic_name: str, context: ChatContext) -> None:
    """Guard against contexts that forward no history to the model.

    Adapter functions evaluate over a conversation, so the context must
    linearize to at least one message. A `SimpleContext` (the `start_session`
    default) always drops history — its `view_for_generation()` returns `[]`
    even after messages are added — which downstream causes an opaque
    `IndexError` inside `apply_chat_template` (issue #937). Fail early here
    with an actionable message instead.

    Args:
        intrinsic_name: Capability name, included in the error message.
        context: The context that will be forwarded to the backend.

    Raises:
        ValueError: When `context.view_for_generation()` is empty or `None`.
    """
    if not context.view_for_generation():
        raise ValueError(
            f"Intrinsic '{intrinsic_name}' received a context that forwards no "
            "history to the model. Adapter functions evaluate over a "
            "conversation, so the context must contain at least one message. "
            "This usually means a SimpleContext was passed (the default for "
            "`start_session`), which does not retain history. Use a ChatContext "
            'instead, e.g. `start_session(..., context_type="chat")` or '
            '`start_backend(..., context_type="chat")`.'
        )


def call_intrinsic(
    intrinsic_name: str,
    context: ChatContext,
    backend: AdapterMixin,
    /,
    kwargs: dict | None = None,
    model_options: dict | None = None,
) -> dict[str, object]:
    """Invoke an adapter function via the backend, returning parsed and validated JSON output.

    Uses `AdapterMixin.resolve_adapter` to find or lazily register the adapter, then
    executes via `mfuncs.act`. The resolved adapter's own
    `IOContract.parse` method is called on the raw output string before returning — the
    output contract always travels with the adapter that produced it, rather than as a
    separate argument a caller could mismatch. For adapters Mellea constructs (the lazy
    shims) that contract comes from the `mellea.backends.adapters.io_contracts` registry;
    for an adapter a user registered under the same capability, the user's own
    `io_contract` wins. The contract validates required fields and
    raises `AdapterSchemaMismatchError` on contract-breaking deltas; forward-compatible
    additions (extra optional fields) do not raise.

    Args:
        intrinsic_name (str): Capability name of the adapter function
            (e.g. `"answerability"`).
        context (ChatContext): The current conversation context.
        backend (AdapterMixin): A backend that supports adapter functions.
        kwargs (dict | None): Extra keyword arguments forwarded to the
            adapter function's input template.
        model_options (dict | None): Model options that override defaults.
            Adapter functions default to `TEMPERATURE: 0.0` for deterministic
            output; pass `TEMPERATURE` here to override it.

    Returns:
        dict[str, object]: Parsed and validated JSON output from the adapter function.

    Raises:
        ValueError: When *context* forwards no history to the model (e.g. a
            `SimpleContext` was passed), when the model output is `None` or is
            not valid JSON, or when well-formed JSON is rejected for shape by
            the resolved adapter's contract (wrong top-level type, or a
            non-object element of an array-shaped contract).
        AdapterSchemaMismatchError: When the model output is missing a field required
            by the resolved adapter's output contract.
    """
    _assert_context_forwards_history(intrinsic_name, context)

    # Resolve (finding or lazily registering) the adapter now, rather than merely
    # ensuring it is registered and discarding the result: its io_contract is what
    # parses the raw output below.
    adapter = backend.resolve_adapter(intrinsic_name)

    # Adapter activation is the backend's responsibility — the HF backend acquires
    # its generation lock and activates the adapter inside adapter_scope(), driven
    # from _generate_intrinsic_with_adapter_scope immediately before generation.
    # Activating here (outside that lock) would race with concurrent async requests.
    intrinsic = Intrinsic(
        intrinsic_name,
        intrinsic_kwargs=kwargs,
        adapter_types=(AdapterType.ALORA, AdapterType.LORA),
    )

    resolved_opts = resolve_model_options(
        backend_defaults={},
        remap={},
        helper_defaults={ModelOption.TEMPERATURE: 0.0},
        call_options=model_options,
    )

    model_output_thunk, _ = mfuncs.act(
        intrinsic,
        context,
        backend,
        model_options=resolved_opts,
        tool_calls=True,
        strategy=None,
    )

    assert model_output_thunk.is_computed()
    result_str = model_output_thunk.value
    if result_str is None:
        raise ValueError("Model output is None.")
    return adapter.io_contract.parse(result_str)
