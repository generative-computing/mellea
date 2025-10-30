"""Backends (e.g., ollama, huggingface, openai-compatible) communicate with LLMs."""

from __future__ import annotations

import abc
from typing import TypeVar

import pydantic

from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, Component, Context, GenerateLog, ModelOutputThunk

BaseModelSubclass = TypeVar(
    "BaseModelSubclass", bound=pydantic.BaseModel
)  # must be a subclass of BaseModel


class Backend(abc.ABC):
    """An abstract `Backend`."""

    def __init__(
        self, model_id: str | ModelIdentifier, *, model_options: dict | None = None
    ):
        """All backends need to be instantiated with a `model_id`.

        A backend can support multiple models, but each instance of a backend corresponds to exactly one model.

        Args:
            model_id (str | ModelIdentifier): The model_id for this model.
            model_options (Optional[dict]): If set, these model options will be used. Otherwise an empty model options dictionary will be used.
        """
        self.model_id = model_id
        self.model_options = model_options if model_options is not None else {}

    @abc.abstractmethod
    def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Context]:
        """Generates a model output from a context. May not mutate the context. This must be called from a running event loop as it creates a task to run the generation request.

        Args:
            action: The last item of the context should be passed in as an `action` instead of as part of the `ctx`. See `docs/dev/generate_signature_decisions.md`.
            ctx: The rest of the context.
            format: A response format to used for structured outputs / constrained decoding.
            model_options: Any model options to upsert into the defaults for this call.
            tool_calls: If `True`, then tool calls are extracts from the `action` `Component`. Assumption: if tool_calls is enabled, then the action `Component` has a TemplateRepresentation

        Returns:
            a tuple of (ModelOutputThunk, Context) where the Context is the new context after the generation has been completed.
        """
        ...

    @abc.abstractmethod
    def generate_from_raw(
        self,
        actions: list[Component | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Generates a model output from the provided input. Does not use context or templates.

        Args:
            actions: list of actions to generate responses for. Each action is separate.
            ctx: context passed to generation. Currently not used in generate_from_raw
            format: A response format to used for structured outputs / constrained decoding. Note: some backends do not support this parameter. They will log warnings and continue to generate.
            model_options: Any model options to upsert into the defaults for this call.
            tool_calls: Always set to false unless supported by backend.
        """
