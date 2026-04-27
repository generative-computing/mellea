"""Typed ``start_backend`` with overloaded return types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import typing_extensions

from ..backends.model_ids import IBM_GRANITE_4_MICRO_3B, ModelIdentifier
from ..core import Backend, Context
from .context import ChatContext, SimpleContext

if TYPE_CHECKING:
    from ..backends.huggingface import LocalHFBackend
    from ..backends.litellm import LiteLLMBackend
    from ..backends.ollama import OllamaModelBackend
    from ..backends.openai import OpenAIBackend
    from ..backends.watsonx import WatsonxAIBackend

CtxT = typing_extensions.TypeVar("CtxT", bound=Context, default=SimpleContext)


# ---------------------------------------------------------------------------
# Overloads: ollama
# ---------------------------------------------------------------------------
@overload
def start_backend(
    backend_name: Literal["ollama"] = ...,
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["chat"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[ChatContext, OllamaModelBackend]: ...


@overload
def start_backend(
    backend_name: Literal["ollama"] = ...,
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["simple"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[SimpleContext, OllamaModelBackend]: ...


@overload
def start_backend(
    backend_name: Literal["ollama"] = ...,
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: None = ...,
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[CtxT, OllamaModelBackend]: ...


# ---------------------------------------------------------------------------
# Overloads: hf
# ---------------------------------------------------------------------------
@overload
def start_backend(
    backend_name: Literal["hf"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["chat"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[ChatContext, LocalHFBackend]: ...


@overload
def start_backend(
    backend_name: Literal["hf"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["simple"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[SimpleContext, LocalHFBackend]: ...


@overload
def start_backend(
    backend_name: Literal["hf"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: None = ...,
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[CtxT, LocalHFBackend]: ...


# ---------------------------------------------------------------------------
# Overloads: openai
# ---------------------------------------------------------------------------
@overload
def start_backend(
    backend_name: Literal["openai"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["chat"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[ChatContext, OpenAIBackend]: ...


@overload
def start_backend(
    backend_name: Literal["openai"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["simple"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[SimpleContext, OpenAIBackend]: ...


@overload
def start_backend(
    backend_name: Literal["openai"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: None = ...,
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[CtxT, OpenAIBackend]: ...


# ---------------------------------------------------------------------------
# Overloads: watsonx
# ---------------------------------------------------------------------------
@overload
def start_backend(
    backend_name: Literal["watsonx"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["chat"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[ChatContext, WatsonxAIBackend]: ...


@overload
def start_backend(
    backend_name: Literal["watsonx"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["simple"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[SimpleContext, WatsonxAIBackend]: ...


@overload
def start_backend(
    backend_name: Literal["watsonx"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: None = ...,
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[CtxT, WatsonxAIBackend]: ...


# ---------------------------------------------------------------------------
# Overloads: litellm
# ---------------------------------------------------------------------------
@overload
def start_backend(
    backend_name: Literal["litellm"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["chat"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[ChatContext, LiteLLMBackend]: ...


@overload
def start_backend(
    backend_name: Literal["litellm"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: Literal["simple"],
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[SimpleContext, LiteLLMBackend]: ...


@overload
def start_backend(
    backend_name: Literal["litellm"],
    model_id: str | ModelIdentifier = ...,
    ctx: CtxT = ...,
    *,
    context_type: None = ...,
    model_options: dict | None = ...,
    **backend_kwargs: Any,
) -> tuple[CtxT, LiteLLMBackend]: ...


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------
def start_backend(
    backend_name: Literal["ollama", "hf", "openai", "watsonx", "litellm"] = "ollama",
    model_id: str | ModelIdentifier = IBM_GRANITE_4_MICRO_3B,
    ctx: Context | None = None,
    *,
    context_type: Literal["simple", "chat"] | None = None,
    model_options: dict | None = None,
    **backend_kwargs: Any,
) -> tuple[Context, Backend]:
    """Create a context and backend pair without a full session.

    Accepts the same backend/model/context arguments as ``start_session`` but
    returns the raw ``(Context, Backend)`` tuple for callers that manage their
    own inference loop.

    Args:
        backend_name: The backend to use (``"ollama"``, ``"hf"``, ``"openai"``,
            ``"watsonx"``, or ``"litellm"``).
        model_id: Model identifier or name.
        ctx: An explicit ``Context`` instance. Mutually exclusive with
            ``context_type``.
        context_type: Shorthand for creating a context — ``"simple"`` for
            ``SimpleContext``, ``"chat"`` for ``ChatContext``. Mutually
            exclusive with ``ctx``.
        model_options: Additional model configuration options passed to the
            backend.
        **backend_kwargs: Additional keyword arguments passed to the backend
            constructor.

    Returns:
        Tuple of ``(Context, Backend)`` with types narrowed by ``backend_name``
        and ``context_type``.

    Raises:
        ValueError: If both ``ctx`` and ``context_type`` are provided.
        Exception: If ``backend_name`` is not recognised.
    """
    from .session import _resolve_backend_and_context

    resolved_ctx, backend, _ = _resolve_backend_and_context(
        backend_name, model_id, ctx, context_type, model_options, **backend_kwargs
    )
    return resolved_ctx, backend
