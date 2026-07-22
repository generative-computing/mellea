# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core adapter scaffolding types (Epic #929 Phase 0).

Introduces the composable ``Adapter`` dataclass and its three parts:

- :class:`Identity` — name, adapter_type, optional capability
- :class:`IOContract` — ABC for prompt building and output parsing
- :class:`WeightsBinding` — pluggable ABC for weights lifecycle management

Also provides three stub :class:`WeightsBinding` subclasses
(:class:`LocalFileBinding`, :class:`EmbeddedBinding`,
:class:`ServerMediatedBinding`) and :class:`AdapterSchemaMismatchError`.

Note:
    The existing :class:`~mellea.backends.adapters.adapter.Adapter` ABC in
    ``adapter.py`` is not modified here.  This module introduces a new
    ``Adapter`` *dataclass* that is re-exported from
    ``mellea.backends.adapters``.  Both coexist until shim removal in 4.1.
    The old ABC is not part of the public ``__init__.py`` surface, so there is
    no namespace collision on the public API.
"""

import abc
import json
import warnings
from dataclasses import dataclass
from typing import Literal

from ...core import Component
from .capabilities import KNOWN_CAPABILITIES

_PHASE_2_NOT_IMPLEMENTED = (
    "{cls} is a Phase 0 stub; implementation lands in Epic #929 Phase 2."
)


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
        # Pass the structured fields (not the formatted message) to Exception so
        # that ``self.args`` round-trips through ``pickle`` / ``copy`` — the default
        # ``Exception.__reduce__`` reconstructs by calling ``cls(*self.args)``.
        super().__init__(name, observed_keys, expected_keys)

    def __str__(self) -> str:
        return (
            f"Adapter '{self.name}' output cannot satisfy declared contract. "
            f"Observed keys: {self.observed_keys}; expected: {self.expected_keys}."
        )


@dataclass(frozen=True)
class Identity:
    """Identifies an adapter by name, type, and optional capability.

    Attributes:
        name (str): Human-readable adapter name.
        adapter_type (Literal["lora", "alora"]): The LoRA variant.
        capability (str | None): Advisory capability string; emits
            :class:`UserWarning` when not in
            :data:`~mellea.backends.adapters.capabilities.KNOWN_CAPABILITIES`.
    """

    name: str
    adapter_type: Literal["lora", "alora"]
    capability: str | None = None

    def __post_init__(self) -> None:
        # Literal[...] is a static-only constraint; mypy enforces it but Python
        # does not, so validate at runtime too.
        if self.adapter_type not in ("lora", "alora"):
            raise ValueError(
                f"adapter_type must be 'lora' or 'alora', got {self.adapter_type!r}"
            )
        if self.capability is not None and self.capability not in KNOWN_CAPABILITIES:
            warnings.warn(
                f"Capability {self.capability!r} is not in the KNOWN_CAPABILITIES "
                "registry. This may indicate a typo or an unregistered capability.",
                UserWarning,
                stacklevel=2,
            )


class IOContract(abc.ABC):
    """Abstract contract for adapter input/output transformations.

    Subclasses implement prompt construction and structured output parsing for
    a specific adapter capability.
    """

    @abc.abstractmethod
    def build_prompt(self, **kwargs: object) -> Component:
        """Build the prompt component for this adapter.

        Args:
            **kwargs: Adapter-specific keyword arguments (e.g. ``documents=...``,
                ``requirement=...``). Concrete subclasses define the keys they
                accept.

        Returns:
            Component: The constructed prompt component.
        """
        ...

    @abc.abstractmethod
    def parse(self, raw: str) -> dict[str, object]:
        """Parse raw model output into a structured dict.

        Args:
            raw (str): Raw string output from the model.

        Returns:
            dict[str, object]: Parsed structured output.

        Raises:
            AdapterSchemaMismatchError: Only on contract-breaking failures (not
                benign additions to the output schema).
        """
        ...


class _DictContract(IOContract):
    """Validate dict-shaped adapter output against a fixed set of required keys.

    Args:
        name: Adapter capability name; included in
            :class:`~mellea.backends.adapters.AdapterSchemaMismatchError` messages.
        required_keys: Keys that must be present in the parsed output dict.
    """

    def __init__(self, name: str, required_keys: frozenset[str]) -> None:
        self._name = name
        self._required_keys = required_keys

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(
            "build_prompt is not used in Phase 1; implemented in Phase 2."
        )

    def parse(self, raw: str) -> dict[str, object]:
        """Parse and validate dict-shaped adapter output.

        Args:
            raw (str): Raw JSON string from the model.

        Returns:
            dict[str, object]: Parsed output dict, unchanged.

        Raises:
            ValueError: When *raw* is not valid JSON or is not a JSON object.
            AdapterSchemaMismatchError: When a required key is absent.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Adapter '{self._name}' output must be a JSON object, "
                f"got {type(data).__name__}."
            )
        observed = frozenset(data.keys())
        missing = self._required_keys - observed
        if missing:
            raise AdapterSchemaMismatchError(self._name, observed, self._required_keys)
        return data


class WeightsBinding(abc.ABC):
    """Abstract lifecycle interface for adapter weights.

    Subclasses manage how adapter weights are obtained, activated on a backend,
    and released when no longer needed.

    Lifecycle (informal state machine):

    - ``prepare()`` — stage the weights (e.g. download); idempotent.
    - ``activate()`` — load into the backend; requires ``prepare()`` first.
    - ``deactivate()`` — unload from the backend; reversible by ``activate()``.
    - ``release()`` — terminal; releases all resources. The binding is not
      reusable after ``release()``.

    Concrete implementations are expected to document any deviations from this
    contract (e.g. servers that prepare-and-activate atomically).
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
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="LocalFileBinding")
        )

    def activate(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="LocalFileBinding")
        )

    def deactivate(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="LocalFileBinding")
        )

    def release(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="LocalFileBinding")
        )


class EmbeddedBinding(WeightsBinding):
    """Stub binding for weights embedded in a model artifact."""

    def prepare(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="EmbeddedBinding")
        )

    def activate(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="EmbeddedBinding")
        )

    def deactivate(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="EmbeddedBinding")
        )

    def release(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="EmbeddedBinding")
        )


class ServerMediatedBinding(WeightsBinding):
    """Stub binding for server-managed adapter weights."""

    def prepare(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="ServerMediatedBinding")
        )

    def activate(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="ServerMediatedBinding")
        )

    def deactivate(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="ServerMediatedBinding")
        )

    def release(self) -> None:
        raise NotImplementedError(
            _PHASE_2_NOT_IMPLEMENTED.format(cls="ServerMediatedBinding")
        )


@dataclass(frozen=True)
class Adapter:
    """Composable adapter dataclass (Epic #929 Phase 0).

    Composes an :class:`Identity`, an :class:`IOContract`, and a
    :class:`WeightsBinding` into a single, inspectable object.

    Attributes:
        identity (Identity): Name, type, and capability for this adapter.
        io_contract (IOContract): Prompt builder and output parser.
        weights (WeightsBinding): Pluggable weights lifecycle handler.
    """

    identity: Identity
    io_contract: IOContract
    weights: WeightsBinding
