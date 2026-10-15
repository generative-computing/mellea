# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`FormatterBackend`: base class for prompt-engineering backends.

`FormatterBackend` extends the abstract `Backend` with a `ChatFormatter` and
a `ModelIdentifier`, bridging mellea's generative programming primitives to models
that do not yet natively support spans or structured fine-tuning. Concrete backend
implementations (e.g. Ollama, Hugging Face, OpenAI) subclass `FormatterBackend` and
supply the model-specific `generate_from_context` logic.
"""

import abc
from enum import Enum
from typing import ClassVar
from urllib.parse import urlparse

from ..core import Backend, ModelOutputThunk
from ..formatters import ChatFormatter, LLMONFormatter
from ..formatters.llmon_formatter import llmon_dialect_for
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
        model_options (dict | None): Default model options; if `None`, an empty dict
            is used.

    """

    # Whether the backend can send LLMON (see `LLMONFormatter`). Only `LocalHFBackend`
    # can: the LLMON chat template reads a turn's parts from message fields that other
    # backends do not pass on, so they reject an LLMON model or formatter outright.
    _supports_llmon: ClassVar[bool] = False

    def __init__(
        self,
        model_id: str | ModelIdentifier,
        formatter: ChatFormatter,
        *,
        model_options: dict | None = None,
    ):
        """Initialize a formatter-based backend for the given model ID.

        Raises:
            ValueError: If the backend does not support LLMON and `formatter` is an
                `LLMONFormatter` or `model_id` is a registered LLMON model.
        """
        if not self._supports_llmon:
            backend = type(self).__name__
            if isinstance(formatter, LLMONFormatter):
                raise ValueError(
                    f"LLMONFormatter can only be used with LocalHFBackend, not "
                    f"{backend}: the LLMON chat template reads a turn's parts from "
                    "message fields that other backends do not pass on."
                )
            if llmon_dialect_for(model_id) is not None:
                raise ValueError(
                    f"{model_id!r} is an LLMON model, which only LocalHFBackend "
                    f"supports, not {backend}: the LLMON chat template reads a turn's "
                    "parts from message fields that other backends do not pass on."
                )
        self.model_id = model_id
        self.model_options = model_options if model_options is not None else {}
        self.formatter: ChatFormatter = formatter
        self._warned_about: set[str] = set()

    def _finalize_output(self, mot: ModelOutputThunk) -> None:
        """Replace `mot.value` with the response the formatter extracts from it.

        When extraction changes the text, the generated text is kept on
        `mot.raw.text`.

        Args:
            mot: The just-completed model output.
        """
        generated = mot.value
        if generated is None:
            return
        response = self.formatter.parse_output(generated)
        if response != generated:
            mot.raw.text = generated
            mot.value = response
