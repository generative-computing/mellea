# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core adapter scaffolding types (Epic #929 Phase 0).

Introduces the composable `Adapter` dataclass and its three parts:

- :class:`Identity` — name, adapter_type, optional capability
- :class:`IOContract` — ABC for prompt building and output parsing
- :class:`WeightsBinding` — pluggable ABC for weights lifecycle management

Also provides :class:`LocalFileBinding`, two stub :class:`WeightsBinding`
subclasses (:class:`EmbeddedBinding`, :class:`ServerMediatedBinding`), and
:class:`AdapterSchemaMismatchError`.

Note:
    The existing :class:`~mellea.backends.adapters.adapter.Adapter` ABC in
    `adapter.py` is not modified here.  This module introduces a new
    `Adapter` *dataclass* that is re-exported from
    `mellea.backends.adapters`.  Both coexist until shim removal in 4.1.
    The old ABC is not part of the public `__init__.py` surface, so there is
    no namespace collision on the public API.
"""

import abc
import json
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from ...core import Component, MelleaLogger
from ...helpers.event_loop_helper import _run_async_in_thread
from ...plugins.manager import has_plugins, invoke_hook
from ...plugins.types import HookType
from .capabilities import KNOWN_CAPABILITIES
from .catalog import AdapterType, fetch_intrinsic_metadata

if TYPE_CHECKING:
    from .adapter import AdapterMixin

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
            **kwargs: Adapter-specific keyword arguments (e.g. `documents=...`,
                `requirement=...`). Concrete subclasses define the keys they
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

    - `prepare()` — stage the weights (e.g. download); idempotent.
    - `activate()` — load into the backend; requires `prepare()` first.
    - `deactivate()` — unload from the backend; reversible by `activate()`.
    - `release()` — terminal; releases all resources. The binding is not
      reusable after `release()`.

    Concrete implementations are expected to document any deviations from this
    contract (e.g. servers that prepare-and-activate atomically).

    Attributes:
        binding_type (ClassVar[str]): Weight-binding reality identifier used in
            adapter-function telemetry (e.g. `"local_file"`).
    """

    binding_type: ClassVar[str] = "unknown"

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
    """Weights binding for the LocalFile/PEFT reality (Epic #929 Phase 2).

    Downloads LoRA/aLoRA adapter weights from a Hugging Face Hub repository and
    loads them into a PEFT-capable backend (e.g. `LocalHFBackend`) via the
    `AdapterMixin` verb contract.

    `prepare()` is session-scoped: call `bind_backend()` once, then `prepare()`.
    `activate()`/`deactivate()` are call-scoped, typically driven by
    `AdapterMixin.adapter_scope`.
    `release()` is terminal.

    Attributes:
        name (str): Adapter function name (e.g. `"answerability"`).
        adapter_type (AdapterType): The LoRA variant.
        repo_id (str): Hugging Face Hub repository containing the adapter weights.
        revision (str | None): Git revision (branch, tag, or commit SHA) to
            download, or `None` to use the catalogue's pinned revision for
            `name` — resolved lazily, so it stays correct if the catalogue is
            re-pinned. Pass `"main"` explicitly to opt into tracking latest.
        backend (AdapterMixin | None): Backend this binding is registered
            with, set by `prepare()` and cleared by `release()`.
        path (str | None): Local filesystem path to the downloaded adapter
            weights, set by `prepare()` and cleared by `release()`.
    """

    binding_type: ClassVar[str] = "local_file"

    def __init__(
        self,
        name: str = "",
        adapter_type: AdapterType = AdapterType.LORA,
        repo_id: str = "",
        revision: str | None = None,
    ) -> None:
        """Constructs a LocalFileBinding.

        Args:
            name: Adapter function name (e.g. `"answerability"`).
            adapter_type: The LoRA variant.
            repo_id: Hugging Face Hub repository containing the adapter weights.
            revision: Git revision (branch, tag, or commit SHA) to download, or
                `None` to use the catalogue's pinned revision for `name`. Pass
                `"main"` explicitly to opt into tracking latest.
        """
        self.name = name
        self.adapter_type = adapter_type
        self.repo_id = repo_id
        self.revision = revision
        self.backend: AdapterMixin | None = None
        self.path: str | None = None
        self._staged_backend: AdapterMixin | None = None
        self._loaded = False
        self._active = False
        self._released = False
        # Keeps one binding from being released while it registers or loads.
        self._lifecycle_lock = threading.Lock()

    @property
    def qualified_name(self) -> str:
        """Backend-facing adapter identifier, e.g. `"answerability_lora"`."""
        return f"{self.name}_{self.adapter_type.value}"

    def resolved_revision(self) -> str:
        """Returns the revision to download, resolving `None` via the catalogue.

        A `revision` of `None` means "whatever the catalogue has pinned for this
        adapter function". Resolving here rather than in `__init__` keeps a
        long-lived binding correct across a catalogue re-pin, and keeps
        construction free of catalogue lookups.

        Returns:
            The git revision (branch, tag, or commit SHA) to download.

        Raises:
            ValueError: `revision` is `None` and `name` is not a registered
                adapter function, so there is no pinned revision to fall back on.
        """
        if self.revision is not None:
            return self.revision
        return fetch_intrinsic_metadata(self.name).revision

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Downloads (or reuses a cached copy of) the adapter weights.

        Args:
            base_model_name: Base model the adapter is being loaded against.

        Returns:
            Filesystem path to the local copy of the adapter weights.

        Raises:
            ValueError: `revision` is `None` and `name` is not a registered
                adapter function.
        """
        from ...formatters.granite import intrinsics

        return str(
            intrinsics.obtain_lora(
                self.name,
                base_model_name,
                self.repo_id,
                revision=self.resolved_revision(),
                alora=self.adapter_type is AdapterType.ALORA,
            )
        )

    @classmethod
    def from_catalog(cls, name: str) -> "LocalFileBinding":
        """Builds a `LocalFileBinding` from the adapter function catalog.

        Args:
            name: Adapter function name registered in the catalog.

        Returns:
            A `LocalFileBinding` configured with the catalog's pinned
            `repo_id`, `revision`, and first-listed adapter type.

        Raises:
            ValueError: `name` is not a registered adapter function.
        """
        metadata = fetch_intrinsic_metadata(name)
        return cls(
            name=name,
            adapter_type=metadata.adapter_types[0],
            repo_id=metadata.repo_id,
            revision=metadata.revision,
        )

    def bind_backend(self, backend: "AdapterMixin") -> None:
        """Stages the backend that `prepare()` will register this binding with.

        Args:
            backend: The backend to register with on the next `prepare()` call.

        Raises:
            RuntimeError: This binding has already been `release()`d, or is
                registered with a different backend.
        """
        with self._lifecycle_lock:
            if self._released:
                raise RuntimeError(
                    "LocalFileBinding.bind_backend() called after release(): "
                    "release() is terminal per the WeightsBinding contract and does "
                    "not revive the binding. Construct a new LocalFileBinding instead."
                )
            if self.backend is not None and backend is not self.backend:
                raise RuntimeError(
                    "LocalFileBinding.bind_backend() cannot change the backend after "
                    "registration. Release this binding and construct a new one instead."
                )
            self._staged_backend = backend

    def prepare(self) -> None:
        """Downloads the adapter weights and loads them into the staged backend.

        Idempotent: a no-op once already prepared. Retryable: if a previous
        call registered with the backend but failed during the weights load
        (e.g. a transient download/load failure), the next call retries only
        the load rather than re-registering — registration already succeeded
        and re-attempting it would hit the backend's own duplicate-registration
        guard.

        The `prepare` phase duration reported to
        `ADAPTER_FUNCTION_PHASE_COMPLETE` spans the whole operation, **including
        the Hugging Face download** — `add_adapter` calls `get_local_hf_path`,
        which can take seconds on a cache miss. That is deliberate (it is the
        wall-clock cost of preparing), but worth stating, since a phase added
        later may not want the same boundary.

        Unlike `activate`/`deactivate` (owned by `AdapterMixin.adapter_scope`),
        `prepare()` runs outside any wrapping invocation, so it opens its own
        single-phase `adapter_function_invocation_start`/`_complete` pair around
        the `adapter_function_phase_start`/`_complete` pair — this is what lets a
        `AdapterFunctionTracingPlugin` (`mellea/telemetry/tracing_plugins.py`)
        emit `adapter_function.prepare` as a child of its own `adapter_function`
        parent span, and guarantees the in-flight span registry drains even if
        `prepare()` raises (the invocation-complete hook always fires, in a
        `finally`, unlike the phase-complete hook, which only fires on success).

        Raises:
            RuntimeError: `bind_backend()` was not called first, `name` is empty,
                the binding was already `release()`d, or the backend refused
                the registration.
        """
        started_at = time.monotonic()
        with self._lifecycle_lock:
            if self._released:
                raise RuntimeError(
                    "LocalFileBinding.prepare() called after release(): release() is "
                    "terminal per the WeightsBinding contract and does not revive the "
                    "binding. Construct a new LocalFileBinding instead."
                )
            if self.backend is not None and self._loaded:
                return

            invocation_id = str(uuid.uuid4())
            revision: str | None
            try:
                revision = self.resolved_revision()
            except Exception:
                revision = self.revision
            self._fire_invocation_start(invocation_id, revision)
            self._fire_phase_start(invocation_id, "prepare", revision)

            error: BaseException | None = None
            try:
                if self.backend is None:
                    if self._staged_backend is None:
                        raise RuntimeError(
                            "LocalFileBinding.prepare() requires bind_backend() to be called first."
                        )
                    if not self.name:
                        raise RuntimeError(
                            "LocalFileBinding.prepare() requires a non-empty name. A "
                            "default-constructed LocalFileBinding() is an unconfigured "
                            "placeholder — build one with LocalFileBinding.from_catalog(name) "
                            "instead."
                        )

                    self._staged_backend.add_adapter(self)
                    # `add_adapter` signals success by setting `.backend`; it has early-return
                    # paths (notably: a different object already registered under this
                    # `qualified_name`) that log a warning and leave it unset. Without this
                    # check `prepare()` would go on to load the *other* adapter's weights and
                    # leave `.backend` None, so a later `activate()` would raise "requires
                    # prepare() to be called first" despite `prepare()` having run. Fail here
                    # instead, where the cause is still visible.
                    if self.backend is None:
                        raise RuntimeError(
                            f"Backend refused to register adapter {self.qualified_name!r}; see the "
                            "backend's warning log. Either another adapter is already registered "
                            "under this qualified name, or this binding was previously released — "
                            "`release()` is terminal and does not free the name for re-use "
                            "(see #1528)."
                        )
                # `load_peft_adapter` mutates the backend's underlying PEFT model, the
                # same shared state `activate_peft_adapter`/`deactivate_peft_adapter`
                # document "must be called while holding `_generation_lock`" for.
                # `prepare()`/`release()` aren't driven through `adapter_scope`, so
                # nothing else takes this lock on their behalf.
                with self.backend._adapter_activation_lock():
                    self.backend.load_peft_adapter(self.qualified_name)
                self._loaded = True
            except BaseException as exc:
                error = exc
                raise
            finally:
                self._fire_invocation_complete(invocation_id, revision, error)
        self._fire_phase_complete(
            "prepare", time.monotonic() - started_at, invocation_id
        )

    def activate(self) -> None:
        """Selects already-loaded adapter weights for generation.

        Raises:
            RuntimeError: `prepare()` was not called first, or called but did
                not complete (registered with the backend but the weights
                load itself failed or hasn't been retried yet).
        """
        backend = self.backend
        if backend is None or not self._loaded:
            raise RuntimeError(
                "LocalFileBinding.activate() requires prepare() to be called first."
            )
        with backend._adapter_activation_lock():
            if self.backend is not backend or not self._loaded:
                raise RuntimeError(
                    "LocalFileBinding.activate() requires prepare() to be called first."
                )
            backend.activate_peft_adapter(self.qualified_name)
            self._active = True

    def deactivate(self) -> None:
        """Deselects the adapter so generation uses the base model.

        Raises:
            RuntimeError: `prepare()` was not called first, or called but did
                not complete (registered with the backend but the weights
                load itself failed or hasn't been retried yet).
        """
        backend = self.backend
        if backend is None or not self._loaded:
            raise RuntimeError(
                "LocalFileBinding.deactivate() requires prepare() to be called first."
            )
        with backend._adapter_activation_lock():
            if self.backend is not backend or not self._loaded:
                raise RuntimeError(
                    "LocalFileBinding.deactivate() requires prepare() to be called first."
                )
            backend.deactivate_peft_adapter(self.qualified_name)
            self._active = False

    def release(self) -> None:
        """Unloads the adapter's weights from the backend and clears local state.

        Idempotent: a no-op if never prepared, or already released. Terminal, per
        the `WeightsBinding` contract — enforced: `bind_backend()` and
        `prepare()` both raise `RuntimeError` if called after `release()`,
        rather than silently reviving the binding on a new backend.

        Does **not** fully deregister. `unload_peft_adapter` removes the adapter
        from the backend's *loaded* set, but the backend's *registered* set
        (`_added_adapters` on `LocalHFBackend`) keeps its entry, because
        `add_adapter` has no inverse verb. So the `qualified_name` stays claimed
        for the backend's lifetime and no later binding can register under it.
        Tracked in #1528, which also asks whether re-registration should be
        supported at all given the terminal contract.

        Raises:
            RuntimeError: The binding is active; call `deactivate()` before
                releasing its weights.
        """
        with self._lifecycle_lock:
            if self._released:
                return
            backend = self.backend
            if backend is None:
                self._staged_backend = None
                self._released = True
                return

            # See the matching comment in `prepare()`: this mutates the same
            # shared PEFT model state `activate_peft_adapter`/
            # `deactivate_peft_adapter` require the lock for.
            with backend._adapter_activation_lock():
                if self.backend is not backend:
                    return
                if self._active:
                    raise RuntimeError(
                        "LocalFileBinding.release() requires deactivate() to be called first."
                    )
                backend.unload_peft_adapter(self.qualified_name)
                self.backend = None
                self.path = None
                self._staged_backend = None
                self._loaded = False
                self._active = False
                self._released = True

    def _fire_invocation_start(self, invocation_id: str, revision: str | None) -> None:
        """Fires `adapter_function_invocation_start` for the prepare-only invocation this call owns.

        `prepare()` does not run inside `AdapterMixin.adapter_scope`, so it opens
        its own single-phase invocation (start here, complete in
        `_fire_invocation_complete`) rather than relying on one supplied by a
        caller — this is what lets a tracing plugin open a real `adapter_function`
        parent span for `adapter_function.prepare` to nest under, and guarantees
        the invocation-complete hook always fires (even if `prepare()` raises),
        which `_fire_phase_complete`'s success-only firing cannot.

        Args:
            invocation_id: Correlation id shared with the matching
                `_fire_invocation_complete` call.
            revision: Catalog revision of the adapter, or `None` if unpinned.
        """
        if not has_plugins(HookType.ADAPTER_FUNCTION_INVOCATION_START):
            return

        from ...plugins.hooks.adapter_function import (
            AdapterFunctionInvocationStartPayload,
        )

        try:
            payload = AdapterFunctionInvocationStartPayload(
                invocation_id=invocation_id,
                name=self.name,
                revision=revision,
                binding_type=self.binding_type,
                adapter_type=self.adapter_type.value,
            )
            hook_coro = invoke_hook(HookType.ADAPTER_FUNCTION_INVOCATION_START, payload)
            _run_async_in_thread(hook_coro)
        except Exception:
            MelleaLogger.get_logger().warning(
                f"adapter_function_invocation_start hook dispatch failed for "
                f"{self.name!r}; ignoring so it does not block prepare().",
                exc_info=True,
            )

    def _fire_invocation_complete(
        self, invocation_id: str, revision: str | None, error: BaseException | None
    ) -> None:
        """Fires `adapter_function_invocation_complete` for the prepare-only invocation this call owns.

        Always fires, in `prepare()`'s `finally`, regardless of success or
        failure — unlike `_fire_phase_complete`, whose contract fires only on
        success. This is what lets a tracing plugin close the `adapter_function`
        parent span (and defensively close any dangling `adapter_function.prepare`
        child span) even when `prepare()` itself raised, so the in-flight span
        registry still drains to zero.

        `prepare()` never parses adapter output, so its outcome is always
        `"success"` or `"error"` — `"schema_error"` is reserved for
        `AdapterMixin.adapter_scope`.

        Args:
            invocation_id: Correlation id shared with the matching
                `_fire_invocation_start` call.
            revision: Catalog revision of the adapter, or `None` if unpinned.
            error: The exception raised during `prepare()`, or `None` on success.
        """
        if not has_plugins(HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE):
            return

        from ...plugins.hooks.adapter_function import (
            AdapterFunctionInvocationCompletePayload,
        )

        try:
            payload = AdapterFunctionInvocationCompletePayload(
                invocation_id=invocation_id,
                name=self.name,
                revision=revision,
                binding_type=self.binding_type,
                adapter_type=self.adapter_type.value,
                outcome="error" if error is not None else "success",
                error=error,
            )
            hook_coro = invoke_hook(
                HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE, payload
            )
            _run_async_in_thread(hook_coro)
        except Exception:
            MelleaLogger.get_logger().warning(
                f"adapter_function_invocation_complete hook dispatch failed for "
                f"{self.name!r}; ignoring so it doesn't mask the real outcome.",
                exc_info=True,
            )

    def _fire_phase_start(
        self, invocation_id: str, phase: str, revision: str | None
    ) -> None:
        """Fires `adapter_function_phase_start` for a phase this binding is about to run.

        Only `"prepare"` is fired from here — see `_fire_phase_complete`.

        Args:
            invocation_id: Correlation id of the enclosing (prepare-only) invocation.
            phase: Lifecycle phase name; must be a valid
                `AdapterFunctionPhaseStartPayload.phase` value.
            revision: Catalog revision of the adapter, or `None` if unpinned.
        """
        if not has_plugins(HookType.ADAPTER_FUNCTION_PHASE_START):
            return

        from ...plugins.hooks.adapter_function import AdapterFunctionPhaseStartPayload

        try:
            payload = AdapterFunctionPhaseStartPayload(
                invocation_id=invocation_id,
                name=self.name,
                phase=phase,
                revision=revision,
            )
            hook_coro = invoke_hook(HookType.ADAPTER_FUNCTION_PHASE_START, payload)
            _run_async_in_thread(hook_coro)
        except Exception:
            MelleaLogger.get_logger().warning(
                f"adapter_function_phase_start hook dispatch failed for {self.name!r} "
                f"during {phase!r}; ignoring so it does not block the phase from "
                "running.",
                exc_info=True,
            )

    def _fire_phase_complete(
        self, phase: str, duration_s: float, invocation_id: str
    ) -> None:
        """Fires `adapter_function_phase_complete` for a phase this binding owns.

        Only `"prepare"` is fired from here: `"activate"`/`"deactivate"` are
        owned by `AdapterMixin.adapter_scope`, and `"release"` has no firing site
        at all — it has a `Literal` value (Epic #929 Phase 1, issue #1140 first
        gave it a phase-metric contract with no metric; issue #1466 keeps that
        deliberately unfired, since `release()` runs outside any invocation) but
        is otherwise unobserved.

        Args:
            phase: Lifecycle phase name; must be a valid
                `AdapterFunctionPhaseCompletePayload.phase` value.
            duration_s: Wall-clock duration of the phase, in seconds.
            invocation_id: Correlation id of the enclosing (prepare-only) invocation.
        """
        if not has_plugins(HookType.ADAPTER_FUNCTION_PHASE_COMPLETE):
            return

        from ...plugins.hooks.adapter_function import (
            AdapterFunctionPhaseCompletePayload,
        )

        try:
            payload = AdapterFunctionPhaseCompletePayload(
                invocation_id=invocation_id,
                name=self.name,
                phase=phase,
                duration_ms=duration_s * 1000.0,
            )
            hook_coro = invoke_hook(HookType.ADAPTER_FUNCTION_PHASE_COMPLETE, payload)
            _run_async_in_thread(hook_coro)
        except Exception:
            MelleaLogger.get_logger().warning(
                f"adapter_function_phase_complete hook dispatch failed for {self.name!r} "
                f"during {phase!r}; ignoring so it does not turn a completed phase "
                "into an operation failure.",
                exc_info=True,
            )


class EmbeddedBinding(WeightsBinding):
    """Stub binding for weights embedded in a model artifact."""

    binding_type: ClassVar[str] = "embedded"

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

    binding_type: ClassVar[str] = "server_mediated"

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

    # NOTE(#1516): a construction-time cross-check that `weights.adapter_type`
    # agrees with `identity.adapter_type` was tried here and backed out. It is the
    # right invariant — the two feed different lookup paths (registration and the
    # verbs key on the binding's `qualified_name`; `_find_adapter` scans on the
    # identity) and both return `None` on a miss, so a disagreement surfaces as
    # "adapter not found" far from its cause. But it cannot be enforced yet: the
    # ten module-level `Adapter` constants in `stdlib/components/intrinsic/rag.py`
    # and `guardian.py` pair an `alora` identity with a bare, deliberately
    # unconfigured `LocalFileBinding()` that defaults to LoRA. Every catalogue
    # entry supports both types, so those are placeholders rather than genuine
    # conflicts, and the check fired on "not configured yet". Enforce it once
    # #1516 gives those constants real bindings.
