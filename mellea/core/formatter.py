# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract `Formatter` interface for rendering components to strings.

A `Formatter` converts `Component` and `CBlock` objects into the text strings
fed to language model prompts. The single abstract method `print` encapsulates this
rendering contract; concrete subclasses such as `ChatFormatter` and
`TemplateFormatter` extend it with chat-message and Jinja2-template rendering
respectively. `parse_output` is the reverse hook: it extracts the model's response
from the text it generated, for formatters whose prompt format shapes the reply.
"""

import abc

from .base import Span


class Formatter(abc.ABC):
    """A Formatter converts `Component`s into strings and parses `ModelOutputThunk`s into `Component`s (or `CBlock`s)."""

    @abc.abstractmethod
    def print(self, c: Span) -> str:
        """Renders a `Component`, `CBlock`, or `ModelOutputThunk` into a string suitable for use as model input.

        Args:
            c (Span): The component, content block, or model output thunk to render.

        Returns:
            str: The rendered string representation of `c`.
        """
        ...

    def parse_output(self, text: str) -> str:
        """Extracts the model's response from the text it generated.

        Backends call this once per generation, after the model finishes and before
        the originating action parses the output. The default returns `text`
        unchanged; a formatter whose prompt format also shapes the reply (such as
        `LLMONFormatter`) overrides it.

        Args:
            text (str): The generated text, exactly as the model returned it.

        Returns:
            str: The response text to store as the output's `value`.
        """
        return text
