"""``FormatterBackend``: base class for prompt-engineering backends.

``FormatterBackend`` extends the abstract ``Backend`` with a ``ChatFormatter`` and
a ``ModelIdentifier``, bridging mellea's generative programming primitives to models
that do not yet natively support spans or structured fine-tuning. Concrete backend
implementations (e.g. Ollama, HuggingFace, OpenAI) subclass ``FormatterBackend`` and
supply the model-specific ``generate_from_context`` logic.
"""

import abc
from enum import Enum
from urllib.parse import urlparse

from ..core import Backend
from ..formatters import ChatFormatter
from .model_ids import ModelIdentifier


class FormatterBackend(Backend, abc.ABC):
    """`FormatterBackend`s support legacy model types.

    The `mellea` library was designed to support generative computing with [spanned attention](https://generative.computing/what-are-spans.html) over [generative programming primitives](https://generative.computing/what-are-generative-programs.html).
    In the ideal world, context management is handled via span scope-relations and all generative programming primitives are baked into the model via fine-tuning.
    I.e., the model's instruction tuning is done in terms of generative programming primitives, and the model is then prompted with the same set of templates that were used for that tuning.

    Today, most models do not yet support spans and even those that do are not properly tuned to leverage generative programming primitives.
    The `mellea` library supports these legacy models primarily through prompt engineering surfaced via `FormatterBackends`.
    A `FormatterBackend` is a backend that uses hand-engineered prompts for rendering generative programming primitives to a model and parsing responses from the model back into `mellea`.
    By default, a `FormatterBackend` uses jinja2 templates for pretty-printing, and relies on the user's ad-hoc logic for parsing.

    Args:
        model_id (str | ModelIdentifier): The model identifier to use for generation.
        formatter (ChatFormatter): The formatter used to convert components into prompts.
        model_options (dict | None): Default model options; if ``None``, an empty dict
            is used.

    """

    def __init__(
        self,
        model_id: str | ModelIdentifier,
        formatter: ChatFormatter,
        *,
        model_options: dict | None = None,
    ):
        """Initialize a formatter-based backend for the given model ID."""
        self.model_id = model_id
        self.model_options = model_options if model_options is not None else {}
        self.formatter: ChatFormatter = formatter
