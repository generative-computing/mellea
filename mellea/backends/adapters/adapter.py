# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter classes for adding fine-tuned modules to inference backends.

The primary public surface is :func:`AdapterMixin.resolve_adapter` (find or lazily
register an adapter by capability name) and :meth:`AdapterMixin._find_adapter`
(look up a registered adapter).  :class:`AdapterMixin` is mixed into backends that
support runtime adapter loading and unloading.

`Adapter` and `LocalHFAdapter` are the legacy adapter ABCs retained for
third-party backends; the composed `Adapter` dataclass in `._core` is the
current adapter surface. `get_adapter_for_intrinsic` is deprecated; prefer
`resolve_adapter`.
"""

import abc
import contextlib
import hashlib
import pathlib
import shutil
import tempfile
import time
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


T = TypeVar("T")


def _composed_adapter_key(adapter: "_AdapterCore") -> str:
    """Return the registry key for a composed `Adapter`, mirroring `qualified_name`.

    A composed `Adapter` has no `qualified_name` of its own; backends key
    their registries on this instead. For a
    LocalFile/PEFT composed adapter, this must produce the same string as
    `adapter.weights.qualified_name` (`LocalFileBinding`'s own key) so a
    backend's `_added_adapters` (keyed on the binding) and `_composed_adapters`
    (keyed on this function) agree — which holds because `Identity.adapter_type`
    is the `Literal["lora", "alora"]` string and `AdapterType.LORA.value`/
    `AdapterType.ALORA.value` are those same two strings. `LocalHFBackend.add_adapter`
    enforces this at registration time for a `LocalFileBinding` (see the
    `NOTE(#1516)` on the `Adapter` dataclass); a composed `Adapter` whose
    `identity.adapter_type` disagrees with its own `weights.adapter_type`
    otherwise produces two different keys instead of one.

    Args:
        adapter: The composed adapter to key.

    Returns:
        str: `"<identity.name>_<identity.adapter_type>"`.
    """
    return f"{adapter.identity.name}_{adapter.identity.adapter_type}"


def _embedded_adapter_from_entry(
    intrinsic_name: str, config: dict, technology: str
) -> "_AdapterCore":
    """Build a composed embedded `Adapter` from one `adapter_index.json` entry.

    Args:
        intrinsic_name (str): Adapter function name from the index entry
            (e.g. `"answerability"`).
        config (dict): Parsed `io.yaml` transformation configuration.
        technology (str): Adapter technology in the switch model — `"lora"` or
            `"alora"`. Determines the `Identity.adapter_type`.

    Returns:
        _AdapterCore: A composed `Adapter` whose `weights` is an
            `EmbeddedBinding` (activation runs through the served model's chat
            template), carrying the declared I/O contract for `intrinsic_name`.

    Raises:
        ValueError: If `technology` is neither `"lora"` nor `"alora"`.
    """
    if technology not in ("lora", "alora"):
        raise ValueError(f"technology must be 'lora' or 'alora', got '{technology}'")
    capability = intrinsic_name
    if intrinsic_name in known_intrinsic_names():
        capability = fetch_intrinsic_metadata(intrinsic_name).effective_capability
    return _AdapterCore(
        identity=Identity(
            name=intrinsic_name,
            adapter_type=cast(Literal["lora", "alora"], technology),
            capability=capability,
        ),
        io_contract=get_io_contract(intrinsic_name),
        weights=EmbeddedBinding(),
    )


def _embedded_adapters_from_model_directory(
    model_path: str | pathlib.Path, intrinsic_name: str | None = None
) -> list[tuple["_AdapterCore", dict]]:
    """Load composed embedded adapters from a Granite Switch model directory.

    Reads `adapter_index.json` and the corresponding `io_configs/*/io.yaml`
    files from the model directory, returning one composed `Adapter` (paired
    with its raw `io.yaml` config) per index entry.

    Args:
        model_path (str | pathlib.Path): Path to a Granite Switch model
            directory that contains `adapter_index.json` and `io_configs/`.
        intrinsic_name (str | None): If provided, only load the adapter
            matching this adapter function name. `None` loads all adapters.

    Returns:
        list[tuple[_AdapterCore, dict]]: One `(adapter, io_yaml_config)` pair
            per entry in the index.

    Raises:
        FileNotFoundError: If `adapter_index.json` is missing.
        ValueError: If an `io.yaml` file listed in the index cannot be found,
            if an `io_config` path escapes the model directory, if an entry's
            `technology` is not `"lora"`/`"alora"`, or if no adapters are found.
    """
    import json as _json

    model_path = pathlib.Path(model_path)
    index_path = model_path / "adapter_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No adapter_index.json found at {index_path}")

    with open(index_path, encoding="utf-8") as f:
        index = _json.load(f)

    adapters: list[tuple[_AdapterCore, dict]] = []
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

        adapter = _embedded_adapter_from_entry(
            entry_name, config_dict, entry.get("technology", "lora")
        )
        adapters.append((adapter, config_dict))

    if not adapters:
        if intrinsic_name is not None:
            raise ValueError(
                f"No adapter found for adapter function '{intrinsic_name}' in {model_path}"
            )
        raise ValueError(f"No adapters found in {model_path}")

    return adapters


def _embedded_adapters_from_hub(
    repo_id: str,
    revision: str = "main",
    cache_dir: str | None = None,
    intrinsic_name: str | None = None,
) -> list[tuple["_AdapterCore", dict]]:
    """Load composed embedded adapters from a Granite Switch model on the Hub.

    Downloads `adapter_index.json` and the `io_configs/` directory into a
    persistent self-contained local directory, then delegates to
    `_embedded_adapters_from_model_directory`.

    `huggingface_hub.snapshot_download`'s default cache-backed snapshot
    directory populates `io_configs/` with symlinks that resolve into a
    sibling `blobs/` directory *outside* the snapshot root. That breaks the
    contract `_embedded_adapters_from_model_directory` expects (a
    self-contained model directory) and trips its path-escape check. To satisfy
    that contract, the downloaded snapshot is materialised under the Hugging
    Face cache into a self-contained directory keyed by its immutable revision,
    so `io_configs/` contains real files rather than symlinks escaping the
    directory. This preserves standard Hugging Face Hub cache reuse and offline
    loading while preventing stale files from a mutable revision.

    Args:
        repo_id (str): Hugging Face Hub repository ID
            (e.g. `"ibm-granite/granite-switch-micro"`).
        revision (str): Git revision to download from.
        cache_dir (str | None): Local cache directory; `None` for the default.
        intrinsic_name (str | None): If provided, only load the adapter
            matching this adapter function name. `None` loads all adapters.

    Returns:
        list[tuple[_AdapterCore, dict]]: One `(adapter, io_yaml_config)` pair
            per entry in the index.

    Raises:
        ImportError: If `huggingface_hub` is not installed.
        PermissionError: If the repository is private or gated and the current
            Hugging Face credentials do not grant access.
        FileNotFoundError: If the downloaded snapshot has no
            `adapter_index.json` (wrong repo/revision, not a Granite Switch
            model, or a stale cache).
        ValueError: If no adapters are found (delegated from
            `_embedded_adapters_from_model_directory`).
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
    cache_key = hashlib.sha256(f"{repo_id}\0{snapshot_root.name}".encode()).hexdigest()
    local_root = cache_root / "mellea" / "embedded-adapter-configs" / cache_key

    try:
        # Validate the cache by content, not just directory existence: a
        # prior run that crashed or lost a `temporary_dir.replace()` race
        # (see below) can leave `local_root` as a directory missing
        # `adapter_index.json`, which `is_dir()` alone would trust forever.
        if not (local_root / "adapter_index.json").is_file():
            local_root.parent.mkdir(parents=True, exist_ok=True)
            if local_root.is_dir():
                # Invalid leftover from a prior partial run -- clear it so
                # `temporary_dir.replace(local_root)` below doesn't fail
                # trying to rename onto a non-empty stale directory.
                shutil.rmtree(local_root)
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

                adapters = _embedded_adapters_from_model_directory(
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

        return _embedded_adapters_from_model_directory(
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


def _embedded_adapters_from_source(
    source: str,
    revision: str = "main",
    cache_dir: str | None = None,
    intrinsic_name: str | None = None,
) -> list[tuple["_AdapterCore", dict]]:
    """Load composed embedded adapters from a local directory or Hugging Face Hub.

    Automatically detects whether `source` is a local filesystem path or a
    Hugging Face Hub repo ID, and delegates accordingly.

    Args:
        source (str): Local path to a model directory, or a Hugging Face Hub
            repo ID (e.g. `"ibm-granite/granite-switch-micro"`).
        revision (str): Git revision (only used for Hub downloads).
        cache_dir (str | None): Cache directory (only used for Hub downloads).
        intrinsic_name (str | None): If provided, only load the adapter
            matching this adapter function name. `None` loads all adapters.

    Returns:
        list[tuple[_AdapterCore, dict]]: One `(adapter, io_yaml_config)` pair
            per entry in the index.
    """
    if pathlib.Path(source).is_dir():
        return _embedded_adapters_from_model_directory(
            source, intrinsic_name=intrinsic_name
        )
    return _embedded_adapters_from_hub(
        source, revision=revision, cache_dir=cache_dir, intrinsic_name=intrinsic_name
    )


def _discover_embedded_adapters(
    source: str,
    *,
    revision: str = "main",
    cache_dir: str | None = None,
    intrinsic_name: str | None = None,
) -> list[tuple["_AdapterCore", dict]]:
    """Discover embedded adapter functions from a Granite Switch source.

    Thin keyword-only wrapper over `_embedded_adapters_from_source`, kept as
    the stable name both backends import. Returns composed `Adapter` instances
    paired with their raw `io.yaml` config.

    The raw parsed `io.yaml` config is returned alongside each adapter because
    a composed `Adapter` has no field for it and it cannot be cheaply
    re-derived later — callers that need it at generation time (see
    `LocalHFBackend`/`OpenAIBackend`'s `_generate_from_intrinsic`) must cache
    it themselves, keyed by `_composed_adapter_key`.

    Args:
        source (str): Local path to a model directory, or a Hugging Face Hub
            repo ID (e.g. `"ibm-granite/granite-switch-micro"`).
        revision (str): Git revision (only used for Hub downloads).
        cache_dir (str | None): Cache directory (only used for Hub downloads).
        intrinsic_name (str | None): If provided, only load the adapter
            matching this adapter function name. `None` loads all adapters.

    Returns:
        list[tuple[_AdapterCore, dict]]: One `(adapter, io_yaml_config)` pair
            per entry in the index.
    """
    return _embedded_adapters_from_source(
        source, revision=revision, cache_dir=cache_dir, intrinsic_name=intrinsic_name
    )


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
    def add_adapter(self, adapter: AdapterInput, *, config: dict | None = None) -> None:
        """Register an adapter with this backend so it can be loaded later.

        The adapter must not already have been added to a different backend.
        Concrete backends accept the full `AdapterInput` union but raise
        `TypeError` for adapter realities they do not implement (e.g. a PEFT
        backend rejects an embedded adapter), so a statically valid call may
        still be rejected at runtime.

        `config` is the raw io.yaml mapping for a composed `Adapter`/
        `_AdapterCore` whose `weights` is an `EmbeddedBinding` or
        `ServerMediatedBinding`. Those realities do not retain the raw
        configuration required by the legacy rewriter, so it must be supplied
        at registration rather than being fetched lazily like a
        `LocalFileBinding`'s io.yaml.

        Args:
            adapter (AdapterInput): The adapter to register with this backend.
            config (dict | None): Raw io.yaml config for a composed
                `EmbeddedBinding` or `ServerMediatedBinding` adapter. Ignored
                (and rejected) for every other adapter reality.

        Raises:
            TypeError: If `adapter` belongs to a reality this backend does not
                support, or `config` is given for a reality other than a
                composed `EmbeddedBinding` or `ServerMediatedBinding` adapter.
            ValueError: If `adapter.weights` is an `EmbeddedBinding` or
                `ServerMediatedBinding` and `config` is not given —
                registering it without a config would make it discoverable
                but permanently unable to generate.
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

        The inverse of `add_adapter()`. LocalFile/PEFT bindings call this from
        `LocalFileBinding.release()` after unloading their weights. Embedded
        backends use it to clear their registration and cached configuration;
        their weights are already part of the served model, so there is no
        unload step.

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
        """Exclusivity lock to hold while calling activate/deactivate and dict writes.

        Default is a no-op (`contextlib.nullcontext()`). Backends whose
        activation verbs mutate shared, non-thread-safe state (e.g.
        `LocalHFBackend`'s underlying PEFT model) override this to return
        their own lock, so callers like `LocalFileBinding.activate()` get
        the same exclusivity `_generate_with_adapter_lock` relies on.
        `add_adapter()`'s registration-dict writes also hold it, briefly.

        Innermost in the adapter lock order
        (`_adapter_resolve_lock` -> `binding._lifecycle_lock` ->
        `_adapter_activation_lock`). Never hold this lock across a
        `WeightsBinding` lifecycle verb (`prepare`/`activate`/`deactivate`/
        `release`) or across any I/O — those take `_lifecycle_lock`, and
        holding activation across them inverts the order and can deadlock
        against a concurrent lifecycle call on the same binding.

        A code path already holding this lock can re-enter it: on
        `LocalHFBackend`, `_generate_composed_local_file_with_adapter_scope` holds
        `_generation_lock` for the whole generation, and the
        `LocalFileBinding` verbs it drives through `adapter_scope()`
        take `_adapter_activation_lock()` again on the same thread. An
        override must therefore return a reentrant lock (`threading.RLock`,
        as `LocalHFBackend` does) — a plain `threading.Lock` here is a real
        deadlock hazard.
        """
        return contextlib.nullcontext()

    def _adapter_resolve_lock(self) -> contextlib.AbstractContextManager[bool | None]:
        """Exclusivity lock for discovery-plus-registration orchestration.

        Default is a no-op (`contextlib.nullcontext()`). Held by
        `resolve_adapter()` and `register_embedded_adapter_model()` across
        their catalogue/Hugging Face Hub discovery I/O and their
        `add_adapter()` calls.

        Outermost in the adapter lock order
        (`_adapter_resolve_lock` -> `binding._lifecycle_lock` ->
        `_adapter_activation_lock`). Deliberately a separate lock from
        `_adapter_activation_lock`, not a reuse of it: the activation lock is
        taken *inside* every `WeightsBinding` lifecycle verb, so holding it
        across `add_adapter()` — which can call those verbs — would invert
        the order and deadlock against a concurrent `prepare()`/`release()`
        on the same binding.

        Must not be acquired while already holding
        `_adapter_activation_lock()` — that reverses the order. No
        production caller does: `resolve_adapter()` runs before the backend
        takes its generation lock, not after (see
        `mellea/stdlib/components/intrinsic/_util.py`).

        An override must return a reentrant lock (`threading.RLock`) for the
        same same-thread-reentry reason `_adapter_activation_lock()`
        documents.

        Known limitation: a `HookType.ADAPTER_FUNCTION_PHASE_COMPLETE`
        subscriber must not synchronously call `resolve_adapter()`,
        `register_embedded_adapter_model()`, or `add_adapter()` with a
        composed `Adapter` on this same backend. `LocalFileBinding.prepare()`
        fires that hook via `_run_async_in_thread`, which blocks the calling
        thread on a dedicated event-loop thread's result — if this lock is
        held at that point (as it is for the whole body of the three methods
        above) and the hook's handler re-enters one of them on the same
        backend, the event-loop thread blocks on this lock while the original
        thread blocks waiting for the event-loop thread: deadlock. No shipped
        caller does this today.
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
            A cold first resolve holds `_adapter_resolve_lock()` across any
            Hugging Face Hub download it triggers, so it can stall every
            other caller of *that* lock (other resolves,
            `register_embedded_adapter_model()`) — including a resolve on
            the asyncio event-loop thread inside a gathered set of
            coroutines. It does **not** stall generation or activation:
            `_adapter_resolve_lock` is a separate lock from
            `_adapter_activation_lock`, deliberately (see the lock-order note
            on `_adapter_activation_lock()`). A warm cache after the first
            resolve avoids the download either way.
        """
        found = self._find_adapter(name)
        if found is not None:
            return found

        base = self.base_model_name
        if base is None:
            raise ValueError(
                f"Backend has no model ID; cannot resolve adapter {name!r}"
            )

        # add_adapter()'s own duplicate check is an unguarded read-then-write on
        # _added_adapters, which races under concurrent first-time resolves for
        # the same name. `_adapter_resolve_lock()` closes it: a no-op by default,
        # and each concrete backend's reentrant lock otherwise. Deliberately not
        # `_adapter_activation_lock()`: that lock is taken *inside*
        # `add_adapter()`'s composed-LocalFileBinding branch around the dict
        # writes only, and inside every `WeightsBinding` lifecycle verb — holding
        # it here, across the whole discover-and-register loop below (which calls
        # `add_adapter()`, which can call `binding.prepare()`), would invert the
        # adapter lock order and deadlock against a concurrent
        # `prepare()`/`release()` on the same binding.
        with self._adapter_resolve_lock():
            # Re-check now the lock is held: a concurrent resolve may have already
            # registered this name. Without this, the loser redundantly re-fetches
            # and then hits the backend's own duplicate guard, which logs a
            # misleading "client code ... not idempotent" warning.
            found = self._find_adapter(name)
            if found is not None:
                return found

            if getattr(self, "_uses_embedded_adapters", False):
                repo_id = (
                    getattr(self, "_adapter_source", None)
                    or getattr(self, "_model_id", None)
                    or base
                )
                # Register composed Adapters discovered from the Granite Switch
                # source. Valid only for backends whose add_adapter supports the
                # Embedded/Granite Switch reality (currently OpenAIBackend and
                # LocalHFBackend when configured with load_embedded_adapters=True).
                #
                # Passing config= lets add_adapter() cache it atomically with
                # registration, gated behind its own duplicate-key guard — a
                # refused duplicate (a different object already holds the
                # key) therefore never reaches the config write, so this
                # can't clobber a live adapter's cached config the way a
                # register-then-separately-cache sequence could.
                for a, config in _discover_embedded_adapters(
                    repo_id, intrinsic_name=name
                ):
                    self.add_adapter(a, config=config)
            else:
                # AdapterType.LORA is the pre-Phase-1 default (mirrors old _util.py).
                # Every current catalog entry supports LORA.  Phase 2 (see epic #929)
                # will select the type from catalog availability instead of hardcoding.
                metadata = fetch_intrinsic_metadata(name)
                self.add_adapter(
                    _AdapterCore(
                        identity=Identity(
                            name=name,
                            adapter_type="lora",
                            capability=metadata.effective_capability,
                        ),
                        io_contract=get_io_contract(name),
                        weights=LocalFileBinding(
                            name=name,
                            adapter_type=AdapterType.LORA,
                            repo_id=metadata.repo_id,
                            revision=metadata.revision,
                        ),
                    )
                )

        found = self._find_adapter(name)
        if found is not None:
            return found

        # `_find_adapter` only matches `_AdapterCore` entries. If registration
        # above silently failed because a `LocalFileBinding` already claims a
        # colliding qualified name (both registration paths share the
        # `f"{name}_{type}"` key space), say so — the alternative is an opaque
        # KeyError that gives no hint the two registration paths collided.
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

        `LocalHFBackend._generate_composed_local_file_with_adapter_scope` is the
        reference
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
        # name is held by a single registered entry. `remove_adapter()` is
        # public, though, so any registered entry can be popped: re-check that
        # invariant when #1465 moves generation inside `adapter_scope`.
        adapters = list(getattr(self, "_added_adapters", {}).values())
        # LocalFile/PEFT composed Adapters (Epic #929, issue #1144) don't live
        # in _added_adapters — that dict holds their LocalFileBinding, keyed
        # for the PEFT lifecycle (see LocalHFBackend.add_adapter); only
        # _composed_adapters carries the _AdapterCore object this method
        # matches on. Embedded composed adapters may live in either, per
        # backend (LocalHFBackend uses _composed_adapters; OpenAIBackend
        # stores them directly in _added_adapters), so checking both here
        # keeps this method backend-agnostic without either backend needing
        # to override it.
        adapters += list(getattr(self, "_composed_adapters", {}).values())
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
