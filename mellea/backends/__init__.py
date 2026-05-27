"""Backend implementations for the mellea inference layer.

This package exposes the concrete machinery for connecting mellea to language model
servers. It bundles `FormatterBackend` (a prompt-engineering base class for legacy
models), `ModelIdentifier` (portable cross-platform model names), `ModelOption`
(generation parameters such as token limits), `SimpleLRUCache` (KV-cache
management), and `MelleaTool` / `tool` (LLM tool definitions). Reach for this
package when configuring a backend, declaring tools, or tuning inference options.
"""

# Import from core for ergonomics.
from ..core import Backend, BaseModelSubclass
from .backend import FormatterBackend
from .cache import SimpleLRUCache
from .model_ids import ModelIdentifier
from .model_options import ModelOption
from .tools import MelleaTool, tool

__all__ = [
    "Backend",
    "BaseModelSubclass",
    "FormatterBackend",
    "MelleaTool",
    "ModelIdentifier",
    "ModelOption",
    "SimpleLRUCache",
    "tool",
]
