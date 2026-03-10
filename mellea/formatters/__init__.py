"""Formatters for converting components into model-ready prompts.

Formatters translate ``Component`` objects into the prompt strings or chat message
lists that inference backends consume. This package exports the abstract ``Formatter``
interface and two concrete implementations: ``ChatFormatter``, which converts
components into role-labelled chat messages, and ``TemplateFormatter``, which renders
them through Jinja2 templates. Pass a formatter when constructing a
``FormatterBackend`` for your chosen model.
"""

# Import from core for ergonomics.
from ..core import Formatter
from .chat_formatter import ChatFormatter
from .template_formatter import TemplateFormatter

__all__ = ["ChatFormatter", "Formatter", "TemplateFormatter"]
