"""Abstract interfaces for Backends that implement Activated LoRAs."""

import abc

from mellea.backends.adapters.adapter import Adapter
from mellea.stdlib.base import CBlock, ModelOutputThunk


# TODO: JAL. Remove this.
class Alora(abc.ABC):
    """Activated LoRAs (Aloras)](https://arxiv.org/pdf/2504.12397) are are [low-rank adapters](https://arxiv.org/abs/2106.09685) that can reuse KV cache from their underlying model.

    This class should not be directly subclassed by a specific ALora. Each backend that supports ALora should provide a backend-specific abstract class that subclasses `ALora`. Individual ALoras should then be defined by subclassing the model-specific backend.

    ALoras are always attached to an underlying model and use the following calling convention:
    1. The underlying model is prompted (without the Alora active). We call this the `input`.
    2. The underlying model generates some tokens from the `input` context (again, without the ALora active). We call this the `response`.
    3. Then the adapter is activated and generates some tokens. We call then the `alora_response`.

    Args:
        name (str): An arbitrary name/label in the model serving engine (e.g. vllm, or local huggingface) to assign to an ALora. This is irrelevant from the alora's (huggingface) model id.
    """

    def __init__(self, name: str):
        """Each aLoRA is identified by a name."""
        self.name: str = name

    @abc.abstractmethod
    def generate_using_strings(self, *args, **kwargs) -> ModelOutputThunk:
        """Generates from the ALora using raw strings as the interface for inputs. In most cases, must be run from a running event loop.

        This has a generic signature because each aLoRA has different parameters depending on its functionality and how it gets called.
        """

    def generate_using_stdlib(self, *args, **kwargs) -> CBlock:
        """Generates from the Alora using Span-based backends."""
        # This is NOT marked as an `abc.abstractmethod` for now because we are not releasing span-based backends. When we release a span-based backend, we should mark this method as `abc.abstractmethod`"""
        raise NotImplementedError(
            "There are not currently ant ALoras trained to use spans."
        )


class AloraBackendMixin(abc.ABC):
    """Mixin class for backends capable of aLoRA functionality."""

    @abc.abstractmethod
    def add_alora(self, *args, **kwargs):
        """Loads an ALora."""
        ...

    @abc.abstractmethod
    def get_alora(self, alora_name: str) -> Alora | None:
        """Returns the ALora by name, or None of that ALora isn't loaded."""
        ...

    @abc.abstractmethod
    def get_aloras(self) -> list[Alora]:
        """Returns a list of all loaded aLoRA adapters."""
        ...


class AdapterMixin(abc.ABC):
    """Mixin class for backends capable of utilizing adapters."""

    def add_adapter(self, *args, **kwargs):
        """Adds the given adapter to the backend. Must not have been added to a different backend."""

    def load_adapter(self, adapter_qualified_name: str):
        """Loads the given adapter for the backend. Must have previously been added."""

    def unload_adapter(self, adapter_qualified_name: str):
        """Unloads the given adapter from the backend."""
