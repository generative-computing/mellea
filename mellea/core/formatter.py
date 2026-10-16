# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract `Formatter` interface for rendering components to strings.

A `Formatter` converts `Component` and `CBlock` objects into the text strings
fed to language model prompts. The single abstract method `print` encapsulates this
rendering contract; concrete subclasses such as `ChatFormatter` and
`TemplateFormatter` extend it with chat-message and Jinja2-template rendering
respectively.
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
