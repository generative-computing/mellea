"""Interfaces for Formatters."""

import abc

from .base import CBlock, Component


class Formatter(abc.ABC):
    """A Formatter converts `Component`s into strings and parses `ModelOutputThunk`s into `Component`s (or `CBlock`s)."""

    @abc.abstractmethod
    def print(self, c: Component | CBlock) -> str:
        """Renders a component for input to a model."""
        ...
