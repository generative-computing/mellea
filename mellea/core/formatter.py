"""Abstract ``Formatter`` interface for rendering components to strings.

A ``Formatter`` converts ``Component`` and ``CBlock`` objects into the text strings
fed to language model prompts. The single abstract method ``print`` encapsulates this
rendering contract; concrete subclasses such as ``ChatFormatter`` and
``TemplateFormatter`` extend it with chat-message and Jinja2-template rendering
respectively.
"""

import abc

from .base import CBlock, Component


class Formatter(abc.ABC):
    """A Formatter converts `Component`s into strings and parses `ModelOutputThunk`s into `Component`s (or `CBlock`s)."""

    @abc.abstractmethod
    def print(self, c: Component | CBlock) -> str:
        """Renders a component for input to a model."""
        ...
