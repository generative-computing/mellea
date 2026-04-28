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


def backend_name_to_class(name: str) -> Any:
    """Resolves backend names to Backend classes.

    Args:
        name: Short backend name, e.g. ``"ollama"``, ``"hf"``, ``"openai"``,
            ``"watsonx"``, or ``"litellm"``.

    Returns:
        The corresponding ``Backend`` class, or ``None`` if the name is unrecognised.

    Raises:
        ImportError: If the requested backend has optional dependencies that are
            not installed (e.g. ``mellea[hf]``, ``mellea[watsonx]``, or
            ``mellea[litellm]``).
    """
    if name == "ollama":
        from ..backends.ollama import OllamaModelBackend

        return OllamaModelBackend
    elif name == "hf" or name == "huggingface":
        try:
            from mellea.backends.huggingface import LocalHFBackend

            return LocalHFBackend
        except ImportError as e:
            raise ImportError(
                "The 'hf' backend requires extra dependencies. "
                "Please install them with: pip install 'mellea[hf]'"
            ) from e
    elif name == "openai":
        from ..backends.openai import OpenAIBackend

        return OpenAIBackend
    elif name == "watsonx":
        try:
            from ..backends.watsonx import WatsonxAIBackend

            return WatsonxAIBackend
        except ImportError as e:
            raise ImportError(
                "The 'watsonx' backend requires extra dependencies. "
                "Please install them with: pip install 'mellea[watsonx]'"
            ) from e
    elif name == "litellm":
        try:
            from ..backends.litellm import LiteLLMBackend

            return LiteLLMBackend
        except ImportError as e:
            raise ImportError(
                "The 'litellm' backend requires extra dependencies. "
                "Please install them with: pip install 'mellea[litellm]'"
            ) from e
    else:
        return None


def _resolve_context(
    ctx: Context | None, context_type: Literal["simple", "chat"] | None
) -> Context:
    """Resolve a ``Context`` from explicit instance and/or shorthand name.

    Raises:
        ValueError: If both ``ctx`` and ``context_type`` are provided.
    """
    if ctx is not None and context_type is not None:
        raise ValueError("Cannot specify both 'ctx' and 'context_type'.")
    if context_type == "chat":
        return ChatContext()
    if context_type == "simple":
        return SimpleContext()
    if ctx is not None:
        return ctx
    return SimpleContext()


def _resolve_model_id_str(model_id: str | ModelIdentifier, backend_name: str) -> str:
    """Resolve a model identifier to its string representation for a given backend."""
    if isinstance(model_id, ModelIdentifier):
        backend_to_attr = {
            "ollama": "ollama_name",
            "hf": "hf_model_name",
            "huggingface": "hf_model_name",
            "openai": "openai_name",
            "watsonx": "watsonx_name",
            "litellm": "hf_model_name",
        }
        attr = backend_to_attr.get(backend_name, "hf_model_name")
        return getattr(model_id, attr, None) or model_id.hf_model_name or str(model_id)
    return str(model_id)


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
    resolved_ctx = _resolve_context(ctx, context_type)
    backend_class = backend_name_to_class(backend_name)
    if backend_class is None:
        raise Exception(
            f"Backend name {backend_name} unknown. Please see the docstring for "
            "`mellea.stdlib.session.start_session` for a list of options."
        )
    backend = backend_class(model_id, model_options=model_options, **backend_kwargs)
    return resolved_ctx, backend
