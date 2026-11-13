# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Formatters for converting components into model-ready prompts.

Formatters translate `Component` objects into the prompt strings or chat message
lists that inference backends consume. This package exports the abstract `Formatter`
interface and three concrete implementations: `ChatFormatter`, which converts
components into role-labelled chat messages; `TemplateFormatter`, which renders
them through Jinja2 templates; and `LLMONFormatter`, which sends them to
LLMON-trained models in the message fields their chat template turns into LLMON. Pass a formatter when constructing a `FormatterBackend`
for your chosen model, or let the backend choose one with `default_formatter`.
"""

# Import from core for ergonomics.
from ..core import Formatter
from .chat_formatter import ChatFormatter
from .llmon_formatter import LLMONFormatter, default_formatter
from .template_formatter import TemplateFormatter

__all__ = [
    "ChatFormatter",
    "Formatter",
    "LLMONFormatter",
    "TemplateFormatter",
    "default_formatter",
]
