"""Shared utilities for intrinsic convenience wrappers."""

import json
from typing import cast

from ....backends import ModelOption
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
    from ....core import CBlock, Component
    from ..chat import Message

    turn = context.last_turn()
    if turn is None or turn.model_input is None:
        raise ValueError(
            "question is None and context has no last turn with model input"
        )

    model_input = turn.model_input
    if isinstance(model_input, Message):
        text = model_input.content
    elif isinstance(model_input, CBlock):
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
    from ..chat import Message

    turn = context.last_turn()
    if turn is None:
        raise ValueError("Context is empty; cannot extract an assistant response.")

    if turn.output is not None and turn.output.value is not None:
        # Session-generated response stored as a ModelOutputThunk.
        # Only the text value is preserved; thunk metadata is intentionally dropped.
        response_text: str = turn.output.value
        prev_ctx = context.previous_node
    elif turn.output is not None and turn.output.value is None:
        raise ValueError(
            "Cannot extract assistant response: it has not been computed yet. "
            "Await the response before calling this adapter function."
        )
    elif (
        turn.model_input is not None
        and isinstance(turn.model_input, Message)
        and turn.model_input.role == "assistant"
    ):
        # Manually-added assistant Message (e.g. built from test fixtures).
        response_text = turn.model_input.content
        prev_ctx = context.previous_node
    else:
        raise ValueError(
            "Cannot extract assistant response: the last context element is "
            "not an assistant response."
        )

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


def call_intrinsic(
    intrinsic_name: str,
    context: ChatContext,
    backend: AdapterMixin,
    /,
    kwargs: dict | None = None,
    model_options: dict | None = None,
):
    """Invoke an adapter function via the backend, returning parsed JSON output.

    Uses `AdapterMixin.resolve_adapter` to find or lazily register the adapter,
    then executes via `mfuncs.act`.

    Args:
        intrinsic_name (str): Capability name of the adapter function
            (e.g. `"answerability"`).
        context (ChatContext): The current conversation context.
        backend (AdapterMixin): A backend that supports adapter functions.
        kwargs (dict | None): Extra keyword arguments forwarded to the
            adapter function's input template.
        model_options (dict | None): Model options that override defaults.

    Returns:
        dict: Parsed JSON output from the adapter function.
    """
    # Ensure the adapter is registered; resolve_adapter creates it if absent.
    backend.resolve_adapter(intrinsic_name)

    # Adapter activation is the backend's responsibility — the HF backend acquires
    # its generation lock and sets the active adapter inside _generate_with_adapter_lock,
    # immediately before generation.  Activating here (outside that lock) would race
    # with concurrent async requests.
    intrinsic = Intrinsic(
        intrinsic_name,
        intrinsic_kwargs=kwargs,
        adapter_types=(AdapterType.ALORA, AdapterType.LORA),
    )

    default_opts: dict = {ModelOption.TEMPERATURE: 0.0}
    if model_options is not None:
        default_opts.update(model_options)

    model_output_thunk, _ = mfuncs.act(
        intrinsic,
        context,
        backend,
        model_options=default_opts,
        tool_calls=True,
        strategy=None,
    )

    assert model_output_thunk.is_computed()
    result_str = model_output_thunk.value
    if result_str is None:
        raise ValueError("Model output is None.")
    return json.loads(result_str)
