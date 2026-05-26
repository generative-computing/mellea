"""Core adapter scaffolding types (Epic #929 Phase 0).

Introduces the composable ``Adapter`` dataclass and its three parts:

- :class:`Identity` — name, adapter_type, optional role
- :class:`IOContract` — ABC for prompt building and output parsing
- :class:`WeightsBinding` — pluggable ABC for weights lifecycle management

Also provides three stub :class:`WeightsBinding` subclasses
(:class:`LocalFileBinding`, :class:`EmbeddedBinding`,
:class:`ServerMediatedBinding`) and :class:`AdapterSchemaMismatchError`.

.. note::
    The existing :class:`~mellea.backends.adapters.adapter.Adapter` ABC in
    ``adapter.py`` is not modified here.  This module introduces a new
    ``Adapter`` *dataclass* that is re-exported from
    ``mellea.backends.adapters``.  Both coexist until shim removal in 4.1.
    The old ABC is not part of the public ``__init__.py`` surface, so there is
    no namespace collision on the public API.
"""

import abc
import warnings
from dataclasses import dataclass
from typing import Literal

from ...core import Component
from .roles import KNOWN_ROLES


class AdapterSchemaMismatchError(Exception):
    """Raised by :meth:`IOContract.parse` when output cannot satisfy the declared contract.

    Attributes:
        name (str): Name of the adapter whose contract was violated.
        observed_keys (frozenset[str]): Keys present in the observed output.
        expected_keys (frozenset[str]): Keys required by the contract.
    """

    def __init__(
        self, name: str, observed_keys: frozenset[str], expected_keys: frozenset[str]
    ) -> None:
        self.name = name
        self.observed_keys = observed_keys
        self.expected_keys = expected_keys
        super().__init__(
            f"Adapter '{name}' output cannot satisfy declared contract. "
            f"Observed keys: {observed_keys}; expected: {expected_keys}."
        )


@dataclass
class Identity:
    """Identifies an adapter by name, type, and optional role.

    Attributes:
        name (str): Human-readable adapter name.
        adapter_type (Literal["lora", "alora"]): The LoRA variant.
        role (str | None): Advisory role string; emits :class:`UserWarning`
            when not in :data:`~mellea.backends.adapters.roles.KNOWN_ROLES`.
    """

    name: str
    adapter_type: Literal["lora", "alora"]
    role: str | None = None

    def __post_init__(self) -> None:
        if self.adapter_type not in ("lora", "alora"):
            raise ValueError(
                f"adapter_type must be 'lora' or 'alora', got {self.adapter_type!r}"
            )
        if self.role is not None and self.role not in KNOWN_ROLES:
            warnings.warn(
                f"Role {self.role!r} is not in the KNOWN_ROLES registry. "
                "This may indicate a typo or an unregistered role.",
                UserWarning,
                stacklevel=2,
            )


class IOContract(abc.ABC):
    """Abstract contract for adapter input/output transformations.

    Subclasses implement prompt construction and structured output parsing for
    a specific adapter capability.
    """

    @abc.abstractmethod
    def build_prompt(self, **kwargs) -> Component:
        """Build the prompt component for this adapter.

        Args:
            **kwargs: Adapter-specific keyword arguments.

        Returns:
            Component: The constructed prompt component.
        """
        ...

    @abc.abstractmethod
    def parse(self, raw: str) -> dict:
        """Parse raw model output into a structured dict.

        Args:
            raw (str): Raw string output from the model.

        Returns:
            dict: Parsed structured output.

        Raises:
            AdapterSchemaMismatchError: Only on contract-breaking failures (not
                benign additions to the output schema).
        """
        ...


class WeightsBinding(abc.ABC):
    """Abstract lifecycle interface for adapter weights.

    Subclasses manage how adapter weights are obtained, activated on a backend,
    and released when no longer needed.
    """

    @abc.abstractmethod
    def prepare(self) -> None:
        """Prepare the weights for activation (e.g. download or stage them)."""
        ...

    @abc.abstractmethod
    def activate(self) -> None:
        """Load the weights into the active backend."""
        ...

    @abc.abstractmethod
    def deactivate(self) -> None:
        """Unload the weights from the active backend."""
        ...

    @abc.abstractmethod
    def release(self) -> None:
        """Release all resources held by this binding."""
        ...


class LocalFileBinding(WeightsBinding):
    """Stub binding for locally stored adapter weights."""

    def prepare(self) -> None:
        raise NotImplementedError

    def activate(self) -> None:
        raise NotImplementedError

    def deactivate(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class EmbeddedBinding(WeightsBinding):
    """Stub binding for weights embedded in a model artifact."""

    def prepare(self) -> None:
        raise NotImplementedError

    def activate(self) -> None:
        raise NotImplementedError

    def deactivate(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class ServerMediatedBinding(WeightsBinding):
    """Stub binding for server-managed adapter weights."""

    def prepare(self) -> None:
        raise NotImplementedError

    def activate(self) -> None:
        raise NotImplementedError

    def deactivate(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


@dataclass
class Adapter:
    """Composable adapter dataclass (Epic #929 Phase 0).

    Composes an :class:`Identity`, an :class:`IOContract`, and a
    :class:`WeightsBinding` into a single, inspectable object.

    Attributes:
        identity (Identity): Name, type, and role for this adapter.
        io_contract (IOContract): Prompt builder and output parser.
        weights (WeightsBinding): Pluggable weights lifecycle handler.
    """

    identity: Identity
    io_contract: IOContract
    weights: WeightsBinding
