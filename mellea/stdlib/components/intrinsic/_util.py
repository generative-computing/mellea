"""Shared utilities for intrinsic convenience wrappers."""

import collections.abc
import json
import warnings

from ....backends import ModelOption
from ....backends.adapters import AdapterMixin, AdapterType, IntrinsicAdapter
from ....core import Backend
from ....stdlib import functional as mfuncs
from ...components import Document
from ...context import ChatContext
from .intrinsic import Intrinsic


def _coerce_documents(
    documents: collections.abc.Iterable[str | Document], *, auto_doc_id: bool = False
) -> list[Document]:
    """Convert an iterable of strings or Documents into a list of Documents.

    Args:
        documents: Strings, Document objects, or a mix.
        auto_doc_id: When True, assign sequential string doc_id values
            ("0", "1", ...) to Documents created from strings and warn about
            existing Document objects that have no ``doc_id`` set.
    """
    result: list[Document] = []
    for i, d in enumerate(documents):
        if isinstance(d, str):
            doc_id = str(i) if auto_doc_id else None
            result.append(Document(text=d, doc_id=doc_id))
        else:
            if auto_doc_id and d.doc_id is None:
                warnings.warn(
                    f"Document at index {i} has no doc_id; results may omit "
                    "document identification. Set doc_id on the Document or "
                    "pass a plain string to auto-generate one.",
                    UserWarning,
                    stacklevel=3,
                )
            result.append(d)
    return result


def _coerce_document(document: str | Document) -> Document:
    """Convert a single string or Document into a Document."""
    if isinstance(document, str):
        return Document(text=document)
    return document


def _resolve_question(
    question: str | None, context: ChatContext, backend: Backend | None = None
) -> tuple[str, ChatContext]:
    """Return ``(question_text, context_to_use)``.

    When *question* is not ``None``, returns it with *context* unchanged.
    When ``None``, extracts the text from the last turn's ``model_input``
    and rewinds *context* to before that element.

    Supports ``Message`` (via ``.content``), ``CBlock`` (via ``.value``),
    and generic ``Component`` types (via ``TemplateFormatter.print()``).
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


def _resolve_response(
    response: str | None, context: ChatContext
) -> tuple[str, ChatContext]:
    """Return ``(response_text, context_to_use)``.

    When *response* is not ``None``, returns it with *context* unchanged.
    When ``None``, extracts from the last turn's ``output.value`` and rewinds
    *context* to before that output.
    """
    if response is not None:
        return response, context
    turn = context.last_turn()
    if turn is None or turn.output is None:
        raise ValueError("response is None and context has no last turn with output")
    if turn.output.value is None:
        raise ValueError("response is None and last turn output has no value")
    rewound = context.previous_node
    if rewound is None:
        raise ValueError("Cannot rewind context past the root node")
    return turn.output.value, rewound  # type: ignore[return-value]


def call_intrinsic(
    intrinsic_name: str,
    context: ChatContext,
    backend: AdapterMixin,
    /,
    kwargs: dict | None = None,
):
    """Shared code for invoking intrinsics.

    :returns: Result of the call in JSON format.
    """
    # Adapter needs to be present in the backend before it can be invoked.
    # We must create the Adapter object in order to determine whether we need to create
    # the Adapter object.
    base_model_name = backend.base_model_name
    if base_model_name is None:
        raise ValueError("Backend has no model ID")
    adapter = IntrinsicAdapter(
        intrinsic_name, adapter_type=AdapterType.LORA, base_model_name=base_model_name
    )
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)

    # Create the AST node for the action we wish to perform.
    intrinsic = Intrinsic(intrinsic_name, intrinsic_kwargs=kwargs)

    # Execute the AST node.
    model_output_thunk, _ = mfuncs.act(
        intrinsic,
        context,
        backend,
        model_options={ModelOption.TEMPERATURE: 0.0},
        # No rejection sampling, please
        strategy=None,
    )

    # act() can return a future. Don't know how to handle one from non-async code.
    assert model_output_thunk.is_computed()

    # Output of an Intrinsic action is the string representation of the output of the
    # intrinsic. Parse the string.
    result_str = model_output_thunk.value
    if result_str is None:
        raise ValueError("Model output is None.")
    result_json = json.loads(result_str)
    return result_json
