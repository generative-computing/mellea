# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter classes for adding fine-tuned modules to inference backends.

The primary public surface is :func:`AdapterMixin.resolve_adapter` (find or lazily
register an adapter by capability name) and :meth:`AdapterMixin._find_adapter`
(look up a registered adapter).  :class:`AdapterMixin` is mixed into backends that
support runtime adapter loading and unloading.

`LocalHFAdapter`, `IntrinsicAdapter`, and `EmbeddedIntrinsicAdapter` are
**deprecation shims** retained for backwards compatibility.  They satisfy
`isinstance(x, _core.Adapter)` but delegate all behaviour to the new dataclass.
`get_adapter_for_intrinsic` is similarly deprecated; prefer `resolve_adapter`.
"""

import abc
import contextlib
import hashlib
import pathlib
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable
from typing import Literal, TypeAlias, TypeVar, cast

import yaml

from ...core import Backend, MelleaLogger
from ...formatters.granite import intrinsics as intrinsics
from ...helpers.event_loop_helper import _run_async_in_thread
from ...plugins.manager import has_plugins, invoke_hook
from ...plugins.types import HookType
from ._core import (
    Adapter as _AdapterCore,
    AdapterSchemaMismatchError,
    EmbeddedBinding,
    Identity,
    LocalFileBinding,
    WeightsBinding,
)
from .catalog import AdapterType, fetch_intrinsic_metadata, known_intrinsic_names
from .io_contracts import get_io_contract


class Adapter(abc.ABC):
    """An adapter that can be added to a single backend.

    An adapter can only be registered with one backend at a time. Use
    `adapter.qualified_name` when referencing the adapter after adding it.

    Args:
        name (str): Human-readable name of the adapter.
        adapter_type (AdapterType): Enum describing the adapter type (e.g.
            `AdapterType.LORA` or `AdapterType.ALORA`).

    Attributes:
        qualified_name (str): Unique name used for loading and lookup; formed
            as `"<name>_<adapter_type.value>"`.
        backend (Backend | None): The backend this adapter has been added to,
            or `None` if not yet added.
        path (str | None): Filesystem path to the adapter weights; set when
            the adapter is added to a backend.
    """

    def __init__(self, name: str, adapter_type: AdapterType):
        """Initialize Adapter with a name and adapter type."""
        self.name = name
        self.adapter_type = adapter_type
        self.qualified_name = name + "_" + adapter_type.value
        """the name of the adapter to use when loading / looking it up"""

        self.backend: Backend | None = None
        """set when the adapter is added to a backend"""

        self.path: str | None = None
        """set when the adapter is added to a backend"""


class LocalHFAdapter(Adapter):
    """Abstract adapter subclass for locally loaded Hugging Face model backends.

    Subclasses must implement `get_local_hf_path` to return the filesystem path
    from which adapter weights should be loaded given a base model name.
    """

    @abc.abstractmethod
    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the Hugging Face model ID (e.g. `"granite-4.0-micro"`).

        Returns:
            str: Filesystem path to the adapter weights directory.
        """
        ...


class _ShimWeightsBinding(WeightsBinding):
    """Placeholder weights binding for the deprecated IntrinsicAdapter shims.

    All lifecycle verbs raise NotImplementedError; it exists only so the
    shims can satisfy the Adapter protocol.
    """

    def prepare(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")

    def activate(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")

    def deactivate(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")

    def release(self) -> None:
        raise NotImplementedError("WeightsBinding not yet implemented")


class IntrinsicAdapter(LocalHFAdapter, _AdapterCore):
    """Deprecated shim for adapters that implement adapter functions.

    Deprecated:
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        `IntrinsicAdapter` will be removed in a future release (Epic #929,
        issue #1144).

    Subtype of :class:`Adapter` for models that:

    * implement adapter functions
    * are packaged as LoRA or aLoRA adapters on top of a base model
    * use the shared model loading code in `mellea.formatters.granite.intrinsics`
    * use the shared input and output processing code in
      `mellea.formatters.granite.intrinsics`

    Args:
        intrinsic_name (str): Name of the adapter function (e.g. `"answerability"`);
            the adapter's `qualified_name` will be derived from this.
        adapter_type (AdapterType): Enum describing the adapter type; defaults to
            `AdapterType.ALORA`.
        config_file (str | pathlib.Path | None): Path to a YAML config file defining
            the adapter function's I/O transformations; mutually exclusive with
            `config_dict`.
        config_dict (dict | None): Dict defining the adapter function's I/O
            transformations; mutually exclusive with `config_file`.
        base_model_name (str | None): Base model name used to look up the I/O
            processing config when neither `config_file` nor `config_dict` are
            provided.

    Attributes:
        intrinsic_name (str): Name of the adapter function this adapter implements.
        intrinsic_metadata (IntrinsicsCatalogEntry): Catalog metadata for the adapter function.
        base_model_name (str | None): Base model name provided at construction, if any.
        adapter_type (AdapterType): The adapter type (`LORA` or `ALORA`).
        config (dict): Parsed I/O transformation configuration for the adapter function.

    Note:
        `identity`, `io_contract`, and `weights` are internal scaffolding populated
        in `__init__` to satisfy the :class:`~mellea.backends.adapters.Adapter`
        protocol; they are not meaningful consumer-facing attributes. `io_contract`
        is the real, declared contract for `intrinsic_name` (issue #1516); `weights`
        remains the Phase 1 `_ShimWeightsBinding` placeholder and raises
        `NotImplementedError` until Phase 2 (issue #1141) replaces it.
    """

    def __setattr__(self, name: str, value: object) -> None:
        """Allow mutation; bypasses the frozen restriction on _AdapterCore."""
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Allow deletion; bypasses the frozen restriction on _AdapterCore."""
        object.__delattr__(self, name)

    def __init__(
        self,
        intrinsic_name: str,
        adapter_type: AdapterType = AdapterType.ALORA,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        base_model_name: str | None = None,
    ):
        """Initialize IntrinsicAdapter for the named adapter function, loading its I/O configuration."""
        warnings.warn(
            "IntrinsicAdapter is deprecated; use Adapter directly (Epic #929, issue #1144).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(intrinsic_name, adapter_type)

        self.intrinsic_name = intrinsic_name
        self.intrinsic_metadata = fetch_intrinsic_metadata(intrinsic_name)
        self.base_model_name = base_model_name

        if adapter_type not in self.intrinsic_metadata.adapter_types:
            raise ValueError(
                f"Adapter function '{intrinsic_name}' not available as an adapter of type "
                f"'{adapter_type}. Available types are "
                f"{self.intrinsic_metadata.adapter_types}."
            )
        self.adapter_type = adapter_type

        # If any of the optional params are specified, attempt to set up the
        # config for the adapter function here.
        if config_file and config_dict:
            raise ValueError(
                f"Conflicting values for config_file and config_dict "
                f"parameters provided. Values were {config_file=} "
                f"and {config_dict=}"
            )
        if config_file is None and config_dict is None and self.base_model_name is None:
            raise ValueError(
                "At least one of [config_file, config_dict, base_model_name] "
                "must be provided."
            )
        if config_file is None and config_dict is None:
            assert self.base_model_name is not None, (
                "must provide `base_model_name` if not providing a `config_file` or `config_dict`"
            )
            # We're converting the adapter type to a boolean flag here.
            assert adapter_type in (AdapterType.ALORA, AdapterType.LORA), (
                f"{adapter_type} not supported"
            )
            is_alora = self.adapter_type == AdapterType.ALORA
            config_file = intrinsics.obtain_io_yaml(
                self.intrinsic_name,
                self.base_model_name,
                self.intrinsic_metadata.repo_id,
                revision=self.intrinsic_metadata.revision,
                alora=is_alora,
            )
        if config_file:
            with open(config_file, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
                if not isinstance(config_dict, dict):
                    raise ValueError(
                        f"YAML file {config_file} does not evaluate to a "
                        f"dictionary when parsed."
                    )
        assert config_dict is not None  # Code above should initialize this variable
        self.config: dict = config_dict

        # Populate the new Adapter triple so isinstance(self, _AdapterCore) holds.
        # io_contract comes from the same registry resolve_adapter() consults
        # (see issue #1516), not a placeholder. weights stays the Phase 2
        # _ShimWeightsBinding placeholder; that axis is #1141/#1142.
        _AdapterCore.__init__(
            self,
            identity=Identity(
                name=intrinsic_name,
                adapter_type="alora"
                if self.adapter_type == AdapterType.ALORA
                else "lora",
                capability=self.intrinsic_metadata.effective_capability,
            ),
            io_contract=get_io_contract(intrinsic_name),
            weights=_ShimWeightsBinding(),
        )

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Downloads the adapter weights if they are not already cached locally.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the Hugging Face model ID (e.g. `"granite-3.3-8b-instruct"`).

        Returns:
            str: Filesystem path to the downloaded adapter weights directory.
        """
        return self.download_and_get_path(base_model_name)

    def download_and_get_path(self, base_model_name: str) -> str:
        """Download the required adapter function files if necessary and return the path to them.

        Args:
            base_model_name: the base model; typically the last part of the Hugging Face
                model id like "granite-3.3-8b-instruct"

        Returns:
            a path to the files
        """
        is_alora = self.adapter_type == AdapterType.ALORA
        return str(
            intrinsics.obtain_lora(
                self.intrinsic_name,
                base_model_name,
                self.intrinsic_metadata.repo_id,
                revision=self.intrinsic_metadata.revision,
                alora=is_alora,
            )
        )


T = TypeVar("T")


def get_adapter_for_intrinsic(
    intrinsic_name: str,
    intrinsic_adapter_types: list[AdapterType] | tuple[AdapterType, ...],
    available_adapters: dict[str, T],
) -> T | None:
    """Find an adapter from a dict of available adapters based on the adapter function name and its allowed adapter types.

    Args:
        intrinsic_name (str): The name of the adapter function, e.g. `"answerability"`.
        intrinsic_adapter_types (list[AdapterType] | tuple[AdapterType, ...]): The
            adapter types allowed for this adapter function, e.g.
            `[AdapterType.ALORA, AdapterType.LORA]`.
        available_adapters (dict[str, T]): The available adapters to choose from;
            maps `adapter.qualified_name` to the adapter object.

    Returns:
        T | None: The first matching adapter found, or `None` if no match exists.
    """
    adapter = None
    for adapter_type in intrinsic_adapter_types:
        qualified_name = f"{intrinsic_name}_{adapter_type.value}"
        adapter = available_adapters.get(qualified_name)
        if adapter is not None:
            break

    return adapter


def _fire_phase_complete_hook(name: str, phase: str, duration_ms: float) -> None:
    """Fire the `adapter_function_phase_complete` metric hook for a phase that already ran.

    Split out of `_run_adapter_phase` so a caller that must guarantee cleanup
    after a phase's side effect — e.g. `adapter_scope` guaranteeing
    `deactivate()` runs once `activate()` has succeeded — can run the side
    effect and this hook fire under separate exception handling. A hook-dispatch
    failure is logged and ignored: observability must not turn a completed
    lifecycle phase into an operation failure.

    Args:
        name: Adapter function name, used as the metric's `name` field.
        phase: Lifecycle phase name; must be a valid
            `AdapterFunctionPhaseCompletePayload.phase` value.
        duration_ms: Wall-clock duration of the phase, in milliseconds.
    """
    if not has_plugins(HookType.ADAPTER_FUNCTION_PHASE_COMPLETE):
        return
    from ...plugins.hooks.adapter_function import AdapterFunctionPhaseCompletePayload

    payload = AdapterFunctionPhaseCompletePayload(
        name=name, phase=phase, duration_ms=duration_ms
    )
    try:
        hook_coro = invoke_hook(HookType.ADAPTER_FUNCTION_PHASE_COMPLETE, payload)
        _run_async_in_thread(hook_coro)
    except Exception:
        MelleaLogger.get_logger().warning(
            f"adapter_function_phase_complete hook dispatch failed for {name!r} "
            f"during {phase!r}; ignoring so it does not turn a completed phase "
            "into an operation failure.",
            exc_info=True,
        )


def _run_adapter_phase(name: str, phase: str, phase_fn: Callable[[], None]) -> None:
    """Run one lifecycle phase and fire its phase-complete metric hook.

    Fires the hook only; it does not open a span. Span production belongs to a
    plugin (#1464, #1466), not to code under `mellea/backends/`.

    The hook fires **only when the phase succeeds**, matching the name of
    `ADAPTER_FUNCTION_PHASE_COMPLETE`: a phase that raised did not complete. If
    `phase_fn` raises, the exception propagates and no phase event is emitted, so
    a consumer reconciling phase counts against invocation counts will see the
    failure only at invocation level, where `outcome` and `error` carry it.

    Args:
        name: Adapter function name, used as the metric's `name` field.
        phase: Lifecycle phase name; must be a valid
            `AdapterFunctionPhaseCompletePayload.phase` value.
        phase_fn: The zero-argument callable implementing the phase (e.g.
            `adapter.weights.activate`).
    """
    started_at = time.monotonic()
    phase_fn()
    _fire_phase_complete_hook(name, phase, (time.monotonic() - started_at) * 1000.0)


def _fire_invocation_complete(
    *,
    name: str,
    revision: str | None,
    binding_type: str,
    adapter_type: str,
    outcome: Literal["success", "schema_error", "error"],
    error: BaseException | None,
) -> None:
    """Fire the `adapter_function_invocation_complete` metric hook.

    Args:
        name: Adapter function name.
        revision: Catalog revision of the adapter, or `None` if unpinned.
        binding_type: Weight-binding reality the adapter ran under.
        adapter_type: Adapter mechanism (e.g. `"lora"`, `"alora"`).
        outcome: Invocation outcome.
        error: The exception raised during invocation, or `None` on success.
    """
    if not has_plugins(HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE):
        return
    from ...plugins.hooks.adapter_function import (
        AdapterFunctionInvocationCompletePayload,
    )

    payload = AdapterFunctionInvocationCompletePayload(
        name=name,
        revision=revision,
        binding_type=binding_type,
        adapter_type=adapter_type,
        outcome=outcome,
        error=error,
    )
    hook_coro = invoke_hook(HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE, payload)
    _run_async_in_thread(hook_coro)


# The full adapter-input surface `add_adapter` advertises. The legacy abc
# `Adapter` (LocalFile/PEFT) and the core dataclass adapter (`_AdapterCore`,
# Embedded/ServerMediated) are disjoint hierarchies, so the accepted type is
# their union. Concrete backends accept this union and raise `TypeError` for the
# adapter realities they do not implement — the same "reject unsupported reality"
# contract the reality-specific verbs use. See the module note on the mixin-vs-
# generic trade-off for why this is a runtime, not a type-parameter, guarantee.
AdapterInput: TypeAlias = Adapter | _AdapterCore | LocalFileBinding


class AdapterMixin(Backend, abc.ABC):
    """Mixin class for backends capable of utilizing adapters.

    Three verbs are universal across every adapter reality (LocalFile/PEFT,
    Embedded/Granite Switch, ServerMediated): `base_model_name`,
    `add_adapter`, and `list_adapters`. The remaining five verbs are
    reality-specific — a concrete backend overrides only the verb(s) matching
    its own reality; the others keep raising `NotImplementedError`.

    Attributes:
        base_model_name (str): The short model name used to identify adapter
            variants (e.g. `"granite-3.3-8b-instruct"` for
            `"ibm-granite/granite-3.3-8b-instruct"`).
    """

    # ---- Universal verbs (every adapter reality) ----

    @property
    @abc.abstractmethod
    def base_model_name(self) -> str:
        """Return the short model name used for adapter variant lookup.

        Returns:
            str: The base model name (e.g. `"granite-3.3-8b-instruct"`).
        """

    @abc.abstractmethod
    def add_adapter(self, adapter: AdapterInput) -> None:
        """Register an adapter with this backend so it can be loaded later.

        The adapter must not already have been added to a different backend.
        Concrete backends accept the full `AdapterInput` union but raise
        `TypeError` for adapter realities they do not implement (e.g. a PEFT
        backend rejects an embedded adapter), so a statically valid call may
        still be rejected at runtime.

        Args:
            adapter (AdapterInput): The adapter to register with this backend.

        Raises:
            TypeError: If `adapter` belongs to a reality this backend does not
                support.
        """

    @abc.abstractmethod
    def list_adapters(self) -> list[str]:
        """Return the qualified names of all adapters registered with this backend.

        Returns:
            list[str]: Qualified adapter names for all adapters that have been
                registered via `add_adapter`.
        """
        ...

    # ---- Reality-specific verbs ----

    def load_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Load a previously registered PEFT adapter into the underlying model.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model). The adapter must have been registered via `add_adapter`
        before calling this method.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to load.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support load_peft_adapter()."
        )

    def unload_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Unload a previously loaded PEFT adapter from the underlying model.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model).

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to unload.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support unload_peft_adapter()."
        )

    def remove_adapter(self, adapter_qualified_name: str) -> None:
        """Deregister a previously added adapter, freeing its qualified name for reuse.

        The inverse of `add_adapter()`. LocalFile/PEFT reality only today
        (#1528) — `LocalFileBinding.release()` calls this after
        `unload_peft_adapter()` so a released `qualified_name` becomes
        claimable by a fresh binding rather than staying claimed for the
        backend's lifetime.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to deregister.

        Raises:
            NotImplementedError: If this backend's adapter reality does not
                support deregistration.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support remove_adapter()."
        )

    def activate_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Switch a previously loaded PEFT adapter on for subsequent generation.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model). The adapter must have been loaded via `load_peft_adapter`
        before calling this method.

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to activate.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support activate_peft_adapter()."
        )

    def deactivate_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Switch off any active PEFT adapter so generation uses the base model.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model).

        Args:
            adapter_qualified_name (str): The `adapter.qualified_name` of the
                adapter to deactivate. Accepted for symmetry with
                `activate_peft_adapter`; the underlying primitive clears all
                active PEFT adapters regardless of name.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support deactivate_peft_adapter()."
        )

    def _adapter_activation_lock(
        self,
    ) -> contextlib.AbstractContextManager[bool | None]:
        """Exclusivity lock to hold while calling activate/deactivate verbs.

        Default is a no-op (`contextlib.nullcontext()`). Backends whose
        activation verbs mutate shared, non-thread-safe state (e.g.
        `LocalHFBackend`'s underlying PEFT model) override this to return
        their own lock, so callers like `LocalFileBinding.activate()` get
        the same exclusivity `_generate_with_adapter_lock` relies on.

        Callers, including `resolve_adapter()`, may re-enter this lock from a
        code path that already holds it on the same thread (e.g. an
        `adapter_scope()` that calls back into a verb guarded by this lock).
        An override must therefore return a reentrant lock (a
        `threading.RLock`, as `LocalHFBackend` does) — a plain
        `threading.Lock` here is a real deadlock hazard, not just a missed
        optimisation.
        """
        return contextlib.nullcontext()

    def resolve_adapter(self, name: str) -> _AdapterCore:
        """Find or lazily register an adapter by capability name.

        Default implementation preserves Phase 0 behaviour, using the internal
        `_added_adapters` dict that concrete backends maintain.  Override in
        Phase 2 (see epic #929) to implement proper lifecycle management.

        Args:
            name (str): Capability name (e.g. `"answerability"`).

        Returns:
            _AdapterCore: The registered adapter with the given capability.

        Raises:
            ValueError: If the backend has no model ID.
            KeyError: If the adapter cannot be found after registration.

        Note:
            On `LocalHFBackend`, this holds `_generation_lock` for the whole
            registration critical section below, including any Hugging Face
            Hub download it triggers (`obtain_io_yaml`/`get_local_hf_path`
            for the LORA path; `EmbeddedIntrinsicAdapter.from_source` for the
            embedded path). A cold first resolve therefore serializes with —
            and can stall behind, or block — every other caller of that same
            lock (activation, generation, other resolves), including a
            resolve made from the asyncio event-loop thread inside a
            gathered set of coroutines. This mirrors the issue's own
            suggested design (reuse the existing lock rather than add a
            second one); a warm cache after the first resolve avoids the
            download and keeps the critical section fast.
        """
        found = self._find_adapter(name)
        if found is not None:
            return found

        base = self.base_model_name
        if base is None:
            raise ValueError(
                f"Backend has no model ID; cannot resolve adapter {name!r}"
            )

        # Registration (add_adapter, possibly several times for the embedded-adapter
        # loop below) and the warnings.catch_warnings() filter mutation around it both
        # race under concurrent first-time resolves for the same name: add_adapter's
        # own duplicate-registration check is an unguarded read-then-write on
        # _added_adapters, and catch_warnings() mutates process-global filter state
        # that is not async/thread-safe. `_adapter_activation_lock()` serializes both
        # races for backends that override it to return a real lock (LocalHFBackend's
        # reentrant `_generation_lock`, shared with load_peft_adapter and
        # LocalFileBinding.activate/deactivate; see #1465) — but not universally: it
        # is a no-op nullcontext() by default (e.g. OpenAIBackend never serializes),
        # and LocalFileBinding.prepare() and register_embedded_adapter_model() still
        # call add_adapter() under their own, narrower locks rather than this one.
        # Suppress DeprecationWarning: the shim constructors warn user-facing code,
        # not internal registration paths.
        with self._adapter_activation_lock(), warnings.catch_warnings():
            # Re-check now that the lock (if any) is held: a concurrent resolve for
            # this same name may have registered it while we were waiting. Without
            # this, the loser redundantly re-fetches (LORA: re-downloads io.yaml and
            # weights; embedded: re-reads the checkpoint) and then hits the backend's
            # own duplicate-registration guard, which logs a "client code attempted
            # to add ... not idempotent" warning that misleadingly blames the caller
            # for what is actually two legitimate concurrent resolve_adapter() calls.
            found = self._find_adapter(name)
            if found is not None:
                return found

            warnings.simplefilter("ignore", DeprecationWarning)
            if getattr(self, "_uses_embedded_adapters", False):
                repo_id = (
                    getattr(self, "_adapter_source", None)
                    or getattr(self, "_model_id", None)
                    or base
                )
                for a in EmbeddedIntrinsicAdapter.from_source(
                    repo_id, intrinsic_name=name
                ):
                    # EmbeddedIntrinsicAdapter is valid only for backends whose
                    # add_adapter supports the Embedded/Granite Switch reality
                    # (currently OpenAIBackend and LocalHFBackend when configured
                    # with load_embedded_adapters=True).
                    self.add_adapter(a)
            else:
                # AdapterType.LORA is the pre-Phase-1 default (mirrors old _util.py).
                # Every current catalog entry supports LORA.  Phase 2 (see epic #929)
                # will select the type from catalog availability instead of hardcoding.
                self.add_adapter(
                    IntrinsicAdapter(
                        name, adapter_type=AdapterType.LORA, base_model_name=base
                    )
                )

        found = self._find_adapter(name)
        if found is not None:
            return found

        # `_find_adapter` only matches `_AdapterCore` entries. If registration
        # above silently failed because a `LocalFileBinding` already claims a
        # colliding qualified name (they share `f"{name}_{type}"` with
        # `IntrinsicAdapter`), say so — the alternative is an opaque KeyError
        # that gives no hint the two registration paths collided.
        # list(...): same concurrent-mutation hazard as `_find_adapter` — snapshot
        # before iterating rather than holding a live view over `_added_adapters`.
        added = list(getattr(self, "_added_adapters", {}).items())
        blocking = next(
            (
                v
                for k, v in added
                if k.startswith(f"{name}_") and not isinstance(v, _AdapterCore)
            ),
            None,
        )
        if blocking is not None:
            blocking_name = getattr(blocking, "qualified_name", None)
            raise KeyError(
                f"Adapter {name!r} not found after registration: a "
                f"{type(blocking).__name__} is already registered under "
                f"{blocking_name!r}, which collides with {name!r}'s auto-registration "
                "path. LocalFileBinding and resolve_adapter()/intrinsic-helper "
                "registrations share the same qualified-name key space on this "
                "backend and cannot both claim it."
            )

        raise KeyError(f"Adapter {name!r} not found after registration")

    @contextlib.contextmanager
    def adapter_scope(self, adapter: "_AdapterCore | None"):  # type: ignore[type-arg]
        """Context manager wrapping adapter activation and deactivation.

        A no-op when `adapter` is `None`. Otherwise: activates
        `adapter.weights`, yields, then always deactivates — even if the `with`
        body raises. Each phase fires `ADAPTER_FUNCTION_PHASE_COMPLETE`, and
        `ADAPTER_FUNCTION_INVOCATION_COMPLETE` fires on the way out, carrying the
        overall outcome.

        This method fires hooks only; it does not open spans. Span production is a
        plugin's job (see #1464 for the rule and #1466 for the adapter-function
        spans), and the `ADAPTER_FUNCTION_*` family currently has no start hook for
        a plugin to open a span on. Hook dispatch goes through
        `_run_async_in_thread` (no timeout): the dispatching call blocks the
        calling thread, but the hook coroutine itself runs on the shared
        `_EventLoopHandler` event-loop thread. A subscriber that blocks on
        something the dispatching thread is holding deadlocks rather than
        merely stalls — e.g. on `LocalHFBackend`, an intrinsic caller holds
        `_generation_lock` across the whole scope, so a subscriber that
        re-enters any `_generation_lock` path blocks the event-loop thread
        while its owner waits on that same event loop, and reentrance cannot
        bridge the gap. Even without such re-entry, a slow or blocking-mode
        `ADAPTER_FUNCTION_*` subscriber delays whatever holds this scope open.

        `deactivate()` is guarded on `activate()`'s own side effect having
        completed, not on the activate phase's hook dispatch also succeeding.
        If a plugin subscribed to `ADAPTER_FUNCTION_PHASE_COMPLETE` raises after
        `activate()` already flipped the adapter on, `deactivate()` still runs —
        telemetry must not be able to strand the adapter active.

        Not atomic across the whole scope **by itself**: `_adapter_activation_lock()`
        is held only inside each of `activate()`/`deactivate()`'s own verb calls
        (see `LocalFileBinding.activate`), not for the `with` body in between.
        Two concurrent `adapter_scope()` calls on one backend can therefore
        interleave — one thread's body can run while a different adapter is
        active, activated by another thread's call — unless the caller closes
        that gap itself. Widening *this method's own* lock to span the whole
        scope was tried and reverted: it deadlocks the moment the body does
        real async generation from the thread that opened the scope, because
        that work runs on the shared event-loop thread while this thread holds
        the lock — a same-thread `RLock` doesn't help across threads.

        `LocalHFBackend._generate_intrinsic_with_adapter_scope` is the reference
        example of a caller that *does* close the gap for its own call site: it
        holds `_generation_lock` around the entire scope, which is safe there
        only because the scope *body* is fully synchronous end to end and
        does no async generation work on the event loop (its only loop
        traffic during the scope is the hook dispatches described above —
        one-way submissions, not re-entry into this backend) — concurrent
        invocations simply land on different threads and serialise on the
        lock, rather than one thread holding it while another does async work
        on the loop. A caller whose
        body awaits work that re-enters generation on another thread must not
        widen a lock this way — that reproduces the deadlock above.

        A caller composing `adapter_scope()` with `LocalHFBackend`'s *standard*
        (non-intrinsic) generation path still silently ignores it: that path
        (`_generate_with_adapter_lock`) always deactivates any adapter before
        generating, so wrapping `generate_from_context()` in `adapter_scope()`
        activates the adapter, generates against the base model anyway, then
        deactivates. Pre-existing, not specific to the intrinsic path this
        method now supports.

        `AdapterFunctionMetricsPlugin` in
        `mellea/telemetry/metrics_plugins.py` emits the adapter-function
        metrics; their instruments and attributes are defined in
        `mellea/telemetry/metrics.py`.

        Args:
            adapter: The adapter to activate, or `None` (no-op).

        Raises:
            TypeError: `adapter.weights` is not a `WeightsBinding` (e.g. an
                `EmbeddedBinding`, which has no activate()/deactivate() to
                scope — call its `apply_activation()` directly instead).
            BaseException: An error raised by activation, the `with` body, or
                deactivation. If both the body and deactivation fail, the body
                error remains primary and the deactivation error is chained.
        """
        if adapter is None:
            yield
            return

        name = adapter.identity.name
        # Prefer `resolved_revision()` over the raw `.revision` attribute: a
        # lazily-resolved binding (`revision=None`) still downloads and runs
        # against the catalogue's pinned SHA, so reporting the unresolved
        # `None` would mislabel an effectively-pinned invocation as unpinned.
        # `resolved_revision()` only exists on `LocalFileBinding`, not the
        # `WeightsBinding` base, so both the lookup and the call are guarded.
        revision: str | None
        if isinstance(adapter.weights, LocalFileBinding):
            try:
                revision = adapter.weights.resolved_revision()
            except Exception:
                revision = adapter.weights.revision
        else:
            revision = cast(str | None, getattr(adapter.weights, "revision", None))
        binding_type = adapter.weights.binding_type
        adapter_type = adapter.identity.adapter_type

        # adapter_scope drives the WeightsBinding lifecycle (activate/deactivate);
        # a binding with no lifecycle (e.g. EmbeddedBinding) activates through its
        # own apply_activation() instead (issue #1142) and never reaches this scope.
        if not isinstance(adapter.weights, WeightsBinding):
            raise TypeError(
                f"adapter_scope() requires a WeightsBinding-backed adapter; "
                f"{binding_type!r} bindings have no activate()/deactivate() to "
                "scope. Call apply_activation() directly instead."
            )

        outcome: Literal["success", "schema_error", "error"] = "success"
        exception: BaseException | None = None
        activated = False
        body_exception: BaseException | None = None
        try:
            started_at = time.monotonic()
            try:
                adapter.weights.activate()
                activated = True
                _fire_phase_complete_hook(
                    name, "activate", (time.monotonic() - started_at) * 1000.0
                )
                try:
                    yield
                except BaseException as exc:
                    body_exception = exc
                    raise
            finally:
                if activated:
                    try:
                        _run_adapter_phase(
                            name, "deactivate", adapter.weights.deactivate
                        )
                    except BaseException as deactivate_exc:
                        if body_exception is None:
                            raise
                        body_exception.add_note(
                            "Adapter deactivation also failed: "
                            f"{type(deactivate_exc).__name__}: {deactivate_exc}"
                        )
        except AdapterSchemaMismatchError as exc:
            # Distinct from a generic error: this is the schema-drift signal the
            # `parse_failures` counter exists to detect, so collapsing it into
            # "error" would leave that counter permanently at zero. Reachable
            # today — `adapter_scope` is public, so a caller can parse inside the
            # scope — and it becomes the common case once #1465 moves generation
            # and parsing in here.
            outcome = "schema_error"
            exception = exc
            raise
        except BaseException as exc:
            outcome = "error"
            exception = exc
            raise
        finally:
            # A hook-dispatch failure here must not replace or mask the real
            # outcome computed above — that would turn a clean `with` block
            # into a thrown error, or swap a genuine body exception for a
            # telemetry-plumbing one. Log and swallow instead.
            try:
                _fire_invocation_complete(
                    name=name,
                    revision=revision,
                    binding_type=binding_type,
                    adapter_type=adapter_type,
                    outcome=outcome,
                    error=exception,
                )
            except Exception:
                MelleaLogger.get_logger().warning(
                    f"adapter_function_invocation_complete hook dispatch failed for "
                    f"{name!r}; ignoring so it doesn't mask the real outcome "
                    f"({outcome!r}).",
                    exc_info=True,
                )

    def _find_adapter(
        self, capability: str, adapter_types: tuple[str, ...] | None = None
    ) -> "_AdapterCore | None":
        """Return the first registered adapter matching capability and (optionally) type.

        Args:
            capability (str): Capability name (e.g. `"answerability"`).
            adapter_types (tuple[str, ...] | None): Adapter type strings in
                preference order (e.g. `("alora", "lora")`).  When provided,
                aLoRA is returned before LoRA if both are registered for the same
                capability.  `None` matches any type (insertion order wins).

        Returns:
            _AdapterCore | None: Matching adapter, or `None` if not found.
        """
        # Snapshot into a list: `_added_adapters` is no longer insert-only since
        # `remove_adapter()` (#1528) can delete from it. A concurrent `release()`
        # mutating the dict while this loop holds a live `.values()` view would
        # raise "dictionary changed size during iteration"; iterating a list
        # copy instead is immune to a mutation of the underlying dict.
        #
        # The snapshot also means this lookup can still see an entry that
        # `remove_adapter()` just popped — harmless today because a qualified
        # name is held by either a `LocalFileBinding` or an `IntrinsicAdapter`
        # shim (never both) and the generation path consumes only shims.
        # `remove_adapter()` is public, though, so any registered entry — shim
        # or binding — can be popped: re-check that invariant when #1465 moves
        # generation inside `adapter_scope`.
        adapters = list(getattr(self, "_added_adapters", {}).values())
        if adapter_types is None:
            for a in adapters:
                if isinstance(a, _AdapterCore) and (
                    a.identity.name == capability or a.identity.capability == capability
                ):
                    return a
            return None
        for preferred_type in adapter_types:
            for a in adapters:
                if (
                    isinstance(a, _AdapterCore)
                    and (
                        a.identity.name == capability
                        or a.identity.capability == capability
                    )
                    and a.identity.adapter_type == preferred_type
                ):
                    return a
        return None


class EmbeddedIntrinsicAdapter(_AdapterCore):
    """Deprecated shim for adapter functions embedded in a Granite Switch model.

    Deprecated:
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        `EmbeddedIntrinsicAdapter` will be removed in a future release
        (Epic #929, issue #1144).

    Unlike PEFT-based adapters that are loaded into the model at runtime,
    embedded adapters are already baked into the model weights and activated
    via control tokens injected by the model's chat template.  Only the I/O
    transformation config (`io.yaml`) is needed; no adapter weights are
    downloaded or loaded.

    Args:
        intrinsic_name (str): Name of the adapter function (e.g. `"answerability"`).
        config (dict): Parsed I/O transformation configuration (from `io.yaml`).
        technology (str): Adapter technology in the switch model — `"lora"` or
            `"alora"`.  Determines where the control token is placed in the
            chat template (beginning of sequence for LoRA, before generation
            prompt for aLoRA).

    Attributes:
        intrinsic_name (str): Name of the adapter function this adapter implements.
        config (dict): Parsed I/O transformation configuration.
        technology (str): `"lora"` or `"alora"`.

    Note:
        `identity`, `io_contract`, and `weights` are internal scaffolding
        populated in `__init__` to satisfy the `Adapter` protocol; they are
        not meaningful consumer-facing attributes.

        - `identity`: always a real value.

        - `io_contract`: the real, declared contract for `intrinsic_name`
          (issue #1516); no longer a placeholder.

        - `weights`: a real `EmbeddedBinding`; activation runs through it.
    """

    def __setattr__(self, name: str, value: object) -> None:
        """Allow mutation; bypasses the frozen restriction on _AdapterCore."""
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Allow deletion; bypasses the frozen restriction on _AdapterCore."""
        object.__delattr__(self, name)

    def __init__(self, intrinsic_name: str, config: dict, technology: str = "lora"):
        """Initialize an embedded adapter function with its I/O config."""
        if technology not in ("lora", "alora"):
            raise ValueError(
                f"technology must be 'lora' or 'alora', got '{technology}'"
            )
        warnings.warn(
            "EmbeddedIntrinsicAdapter is deprecated; use Adapter directly (Epic #929, issue #1144).",
            DeprecationWarning,
            stacklevel=2,
        )
        adapter_type = AdapterType.ALORA if technology == "alora" else AdapterType.LORA

        # Old-style Adapter fields — set manually since we no longer inherit from the
        # legacy Adapter ABC.  Preserved for backward compatibility until Phase 4.
        self.name = intrinsic_name
        self.adapter_type = adapter_type
        self.qualified_name = intrinsic_name + "_" + adapter_type.value
        self.backend: Backend | None = None
        self.path: str | None = None

        self.intrinsic_name = intrinsic_name
        self.config = config
        self.technology = technology
        capability = intrinsic_name
        if intrinsic_name in known_intrinsic_names():
            capability = fetch_intrinsic_metadata(intrinsic_name).effective_capability

        # Populate the new Adapter triple so isinstance(self, _AdapterCore) holds.
        # technology is validated above; cast to the Literal type mypy expects.
        identity = Identity(
            name=intrinsic_name,
            adapter_type=cast(Literal["lora", "alora"], technology),
            capability=capability,
        )

        io_contract = get_io_contract(intrinsic_name)

        weights = EmbeddedBinding()

        _AdapterCore.__init__(
            self, identity=identity, io_contract=io_contract, weights=weights
        )

    @staticmethod
    def from_model_directory(
        model_path: str | pathlib.Path, intrinsic_name: str | None = None
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a Granite Switch model directory.

        Reads `adapter_index.json` and the corresponding `io_configs/*/io.yaml`
        files from the model directory.

        Args:
            model_path (str | pathlib.Path): Path to a Granite Switch model
                directory that contains `adapter_index.json` and `io_configs/`.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. `None` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.

        Raises:
            FileNotFoundError: If `adapter_index.json` is missing.
            ValueError: If an `io.yaml` file listed in the index cannot be found
                or if no adapters are found.
        """
        import json as _json

        model_path = pathlib.Path(model_path)
        index_path = model_path / "adapter_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No adapter_index.json found at {index_path}")

        with open(index_path, encoding="utf-8") as f:
            index = _json.load(f)

        adapters: list[EmbeddedIntrinsicAdapter] = []
        for entry in index.get("adapters", []):
            entry_name = entry.get("adapter_name")
            if entry_name is None:
                continue
            if intrinsic_name is not None and entry_name != intrinsic_name:
                continue
            io_config_rel = entry.get("io_config")
            if io_config_rel is None:
                continue

            io_config_path = model_path / io_config_rel
            try:
                io_config_path = io_config_path.resolve(strict=True)
            except (FileNotFoundError, OSError):
                raise ValueError(
                    f"io.yaml for adapter function '{entry_name}' "
                    f"not found at {model_path / io_config_rel}"
                )
            if not io_config_path.is_relative_to(model_path.resolve()):
                raise ValueError(
                    f"io_config path for adapter function '{entry_name}' "
                    f"escapes the model directory: {io_config_path}"
                )

            with open(io_config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            adapters.append(
                EmbeddedIntrinsicAdapter(
                    intrinsic_name=entry_name,
                    config=config_dict,
                    technology=entry.get("technology", "lora"),
                )
            )

        if not adapters:
            if intrinsic_name is not None:
                raise ValueError(
                    f"No adapter found for adapter function '{intrinsic_name}' in {model_path}"
                )
            raise ValueError(f"No adapters found in {model_path}")

        return adapters

    @staticmethod
    def from_hub(
        repo_id: str,
        revision: str = "main",
        cache_dir: str | None = None,
        intrinsic_name: str | None = None,
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a Granite Switch model on Hugging Face Hub.

        Downloads `adapter_index.json` and the `io_configs/` directory into a
        persistent self-contained local directory, then delegates to
        `from_model_directory`.

        `huggingface_hub.snapshot_download`'s default cache-backed snapshot
        directory populates `io_configs/` with symlinks that resolve into a
        sibling `blobs/` directory *outside* the snapshot root. That breaks the
        contract `from_model_directory` expects (a self-contained model
        directory) and trips its path-escape check. To satisfy that contract,
        the downloaded snapshot is materialised under the Hugging Face cache
        into a self-contained directory keyed by its immutable revision, so
        `io_configs/` contains real files rather than symlinks escaping the
        directory. This preserves standard Hugging Face Hub cache reuse and
        offline loading while preventing stale files from a mutable revision.

        Args:
            repo_id (str): Hugging Face Hub repository ID
                (e.g. `"ibm-granite/granite-switch-micro"`).
            revision (str): Git revision to download from.
            cache_dir (str | None): Local cache directory; `None` for the default.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. `None` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.

        Raises:
            ImportError: If `huggingface_hub` is not installed.
            PermissionError: If the repository is private or gated and the
                current Hugging Face credentials do not grant access.
            FileNotFoundError: If the downloaded snapshot has no
                `adapter_index.json` (wrong repo/revision, not a Granite Switch
                model, or a stale cache).
            ValueError: If no adapters are found (delegated from
                `from_model_directory`).
        """
        try:
            import huggingface_hub
            from huggingface_hub.constants import HF_HUB_CACHE
            from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download embedded adapter configs from "
                'Hugging Face Hub. Please install it with: pip install "mellea[switch]"'
            ) from e

        try:
            snapshot_root = pathlib.Path(
                huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["adapter_index.json", "io_configs/**"],
                    cache_dir=cache_dir,
                    revision=revision,
                )
            )
        except (GatedRepoError, RepositoryNotFoundError) as e:
            auth_hint = (
                f"Could not access '{repo_id}' on Hugging Face Hub. If this is a "
                "private or gated repository, authenticate first (run "
                "`huggingface-cli login` or set the HF_TOKEN environment variable) "
                "and confirm your account has been granted access to the repository. "
                "Otherwise, the repository ID may be misspelled."
            )
            raise PermissionError(auth_hint) from e

        cache_root = pathlib.Path(cache_dir or HF_HUB_CACHE)
        cache_key = hashlib.sha256(
            f"{repo_id}\0{snapshot_root.name}".encode()
        ).hexdigest()
        local_root = cache_root / "mellea" / "embedded-adapter-configs" / cache_key

        try:
            if not local_root.is_dir():
                local_root.parent.mkdir(parents=True, exist_ok=True)
                temporary_dir = pathlib.Path(
                    tempfile.mkdtemp(dir=local_root.parent, prefix=f"{cache_key}-")
                )
                try:
                    import json as _json

                    index_path = snapshot_root / "adapter_index.json"
                    with open(index_path, encoding="utf-8") as f:
                        index = _json.load(f)
                    shutil.copyfile(index_path, temporary_dir / "adapter_index.json")

                    snapshot_cache_root = snapshot_root.parent.parent.resolve()
                    for entry in index.get("adapters", []):
                        io_config_rel = entry.get("io_config")
                        if io_config_rel is None:
                            continue
                        io_config_path = (snapshot_root / io_config_rel).resolve(
                            strict=True
                        )
                        if not io_config_path.is_relative_to(snapshot_cache_root):
                            raise ValueError(
                                f"io_config path '{io_config_rel}' escapes "
                                "the downloaded Hugging Face snapshot"
                            )
                        destination = temporary_dir / io_config_rel
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(io_config_path, destination)

                    adapters = EmbeddedIntrinsicAdapter.from_model_directory(
                        temporary_dir, intrinsic_name=intrinsic_name
                    )
                    try:
                        temporary_dir.replace(local_root)
                    except OSError:
                        if not local_root.is_dir():
                            raise
                    else:
                        return adapters
                finally:
                    if temporary_dir.exists():
                        shutil.rmtree(temporary_dir)

            return EmbeddedIntrinsicAdapter.from_model_directory(
                local_root, intrinsic_name=intrinsic_name
            )
        except FileNotFoundError as e:
            # snapshot_download succeeded but the index is absent: wrong
            # repo/revision, a repo that isn't a Granite Switch model, or a
            # stale cache. Replace the cryptic snapshot-cache path with a
            # repo-scoped message that names authentication as one possible
            # cause without asserting it.
            raise FileNotFoundError(
                f"adapter_index.json was not found in the downloaded snapshot of "
                f"'{repo_id}'. Verify it is a Granite Switch model and that the "
                f"revision '{revision}' is correct; if the repository is private "
                "or gated, confirm you are authenticated (run "
                "`huggingface-cli login` or set the HF_TOKEN environment variable)."
            ) from e
        except ValueError as e:
            if intrinsic_name is not None:
                raise ValueError(
                    f"No adapter found for adapter function '{intrinsic_name}' in {repo_id}"
                ) from e
            raise ValueError(f"No adapters found in {repo_id}") from e

    @staticmethod
    def from_source(
        source: str,
        revision: str = "main",
        cache_dir: str | None = None,
        intrinsic_name: str | None = None,
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a local directory or Hugging Face Hub.

        Automatically detects whether `source` is a local filesystem path
        or a Hugging Face Hub repo ID, and delegates accordingly.

        Args:
            source (str): Local path to a model directory, or a Hugging Face
                Hub repo ID (e.g. `"ibm-granite/granite-switch-micro"`).
            revision (str): Git revision (only used for Hub downloads).
            cache_dir (str | None): Cache directory (only used for Hub downloads).
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. `None` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.
        """
        if pathlib.Path(source).is_dir():
            return EmbeddedIntrinsicAdapter.from_model_directory(
                source, intrinsic_name=intrinsic_name
            )
        return EmbeddedIntrinsicAdapter.from_hub(
            source,
            revision=revision,
            cache_dir=cache_dir,
            intrinsic_name=intrinsic_name,
        )


class CustomIntrinsicAdapter(IntrinsicAdapter):
    """Deprecated shim for user-defined custom adapter functions.

    .. deprecated::
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        `CustomIntrinsicAdapter` will be removed in a future release
        (Epic #929, issue #1144).

    This class has the same functionality as `IntrinsicAdapter`, except that
    its constructor monkey-patches Mellea global variables to enable the backend
    to load the user's adapter.

    Args:
        model_id (str): The Hugging Face model ID used for downloading model weights;
            expected format is `"<user-id>/<repo-name>"`.
        intrinsic_name (str | None): Catalog name for the adapter function; defaults to the
            repository name portion of `model_id` if not provided.
        base_model_name (str): The short name of the base model (NOT its repo ID).
    """

    def __init__(
        self, *, model_id: str, intrinsic_name: str | None = None, base_model_name: str
    ):
        """Initialize CustomIntrinsicAdapter and patch the global adapter function catalog if needed."""
        warnings.warn(
            "CustomIntrinsicAdapter is deprecated; use Adapter directly (Epic #929, issue #1144).",
            DeprecationWarning,
            stacklevel=2,
        )
        assert re.match(".*/.*", model_id), (
            "expected a Hugging Face model id with format <user-id>/<repo-name>"
        )
        intrinsic_name = (
            intrinsic_name if intrinsic_name is not None else model_id.split("/")[1]
        )

        # patch the catalog. TODO this is a temporary hack until we re-org adapters.
        from mellea.backends.adapters import catalog

        if intrinsic_name not in catalog._INTRINSICS_CATALOG:
            catalog._INTRINSICS_CATALOG_ENTRIES.append(
                catalog.IntrinsicsCatalogEntry(
                    name=intrinsic_name, repo_id=model_id, revision="main"
                )
            )
            catalog._INTRINSICS_CATALOG = {
                e.name: e for e in catalog._INTRINSICS_CATALOG_ENTRIES
            }

        # Suppress DeprecationWarning from the IntrinsicAdapter shim: the warning we
        # emitted above is already correctly attributed to the caller's frame.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            IntrinsicAdapter.__init__(
                self, intrinsic_name=intrinsic_name, base_model_name=base_model_name
            )
