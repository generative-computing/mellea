# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter classes for adding fine-tuned modules to inference backends.

The primary public surface is :func:`AdapterMixin.resolve_adapter` (find or lazily
register an adapter by capability name) and :meth:`AdapterMixin._find_adapter`
(look up a registered adapter).  :class:`AdapterMixin` is mixed into backends that
support runtime adapter loading and unloading.

``LocalHFAdapter``, ``IntrinsicAdapter``, and ``EmbeddedIntrinsicAdapter`` are
**deprecation shims** retained for backwards compatibility.  They satisfy
``isinstance(x, _core.Adapter)`` but delegate all behaviour to the new dataclass.
``get_adapter_for_intrinsic`` is similarly deprecated; prefer ``resolve_adapter``.
"""

import abc
import contextlib
import pathlib
import re
import warnings
from typing import Literal, TypeVar, cast

import yaml

from ...core import Backend
from ...formatters.granite import intrinsics as intrinsics
from ._core import Adapter as _AdapterCore, Identity, IOContract, WeightsBinding
from .catalog import AdapterType, fetch_intrinsic_metadata


class Adapter(abc.ABC):
    """An adapter that can be added to a single backend.

    An adapter can only be registered with one backend at a time. Use
    ``adapter.qualified_name`` when referencing the adapter after adding it.

    Args:
        name (str): Human-readable name of the adapter.
        adapter_type (AdapterType): Enum describing the adapter type (e.g.
            ``AdapterType.LORA`` or ``AdapterType.ALORA``).

    Attributes:
        qualified_name (str): Unique name used for loading and lookup; formed
            as ``"<name>_<adapter_type.value>"``.
        backend (Backend | None): The backend this adapter has been added to,
            or ``None`` if not yet added.
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

    Subclasses must implement ``get_local_hf_path`` to return the filesystem path
    from which adapter weights should be loaded given a base model name.
    """

    @abc.abstractmethod
    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the Hugging Face model ID (e.g. ``"granite-4.0-micro"``).

        Returns:
            str: Filesystem path to the adapter weights directory.
        """
        ...


class _ShimIOContract(IOContract):
    """Phase 1 placeholder; Phase 2 (issue #1137) implements real I/O."""

    def build_prompt(self, **kwargs: object):  # type: ignore[override]
        raise NotImplementedError(
            "Phase 2 (issue #1137) — IOContract not yet implemented"
        )

    def parse(self, raw: str) -> dict[str, object]:
        raise NotImplementedError(
            "Phase 2 (issue #1137) — IOContract not yet implemented"
        )


class _ShimWeightsBinding(WeightsBinding):
    """Phase 1 placeholder; Phase 2 (see epic #929) wires in real lifecycle."""

    def prepare(self) -> None:
        raise NotImplementedError(
            "Phase 2 (see epic #929) — WeightsBinding not yet implemented"
        )

    def activate(self) -> None:
        raise NotImplementedError(
            "Phase 2 (see epic #929) — WeightsBinding not yet implemented"
        )

    def deactivate(self) -> None:
        raise NotImplementedError(
            "Phase 2 (see epic #929) — WeightsBinding not yet implemented"
        )

    def release(self) -> None:
        raise NotImplementedError(
            "Phase 2 (see epic #929) — WeightsBinding not yet implemented"
        )


class IntrinsicAdapter(LocalHFAdapter, _AdapterCore):
    """Deprecated shim for adapters that implement adapter functions.

    .. deprecated::
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        ``IntrinsicAdapter`` will be removed in a future release (Epic #929,
        issue #1144).

    Subtype of :class:`Adapter` for models that:

    * implement adapter functions
    * are packaged as LoRA or aLoRA adapters on top of a base model
    * use the shared model loading code in ``mellea.formatters.granite.intrinsics``
    * use the shared input and output processing code in
      ``mellea.formatters.granite.intrinsics``

    Args:
        intrinsic_name (str): Name of the adapter function (e.g. ``"answerability"``);
            the adapter's ``qualified_name`` will be derived from this.
        adapter_type (AdapterType): Enum describing the adapter type; defaults to
            ``AdapterType.ALORA``.
        config_file (str | pathlib.Path | None): Path to a YAML config file defining
            the adapter function's I/O transformations; mutually exclusive with
            ``config_dict``.
        config_dict (dict | None): Dict defining the adapter function's I/O
            transformations; mutually exclusive with ``config_file``.
        base_model_name (str | None): Base model name used to look up the I/O
            processing config when neither ``config_file`` nor ``config_dict`` are
            provided.

    Attributes:
        intrinsic_name (str): Name of the adapter function this adapter implements.
        intrinsic_metadata (IntrinsicsCatalogEntry): Catalog metadata for the adapter function.
        base_model_name (str | None): Base model name provided at construction, if any.
        adapter_type (AdapterType): The adapter type (``LORA`` or ``ALORA``).
        config (dict): Parsed I/O transformation configuration for the adapter function.

    .. note::
        ``identity``, ``io_contract``, and ``weights`` are Phase 1 internal scaffolding
        populated in ``__init__`` to satisfy the new :class:`~mellea.backends.adapters.Adapter`
        protocol.  They are not meaningful consumer-facing attributes; ``io_contract`` and
        ``weights`` raise :exc:`NotImplementedError` and will be replaced in Phase 2
        (issues #1137, #1141).
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
            # TODO(phase-2.2): pass revision=self.intrinsic_metadata.revision
            # once revision-aware prepare() is merged (issue #1141 / epic #929).
            config_file = intrinsics.obtain_io_yaml(
                self.intrinsic_name,
                self.base_model_name,
                self.intrinsic_metadata.repo_id,
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
        _AdapterCore.__init__(
            self,
            identity=Identity(
                name=intrinsic_name,
                adapter_type="alora"
                if self.adapter_type == AdapterType.ALORA
                else "lora",
                capability=intrinsic_name,
            ),
            io_contract=_ShimIOContract(),
            weights=_ShimWeightsBinding(),
        )

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Downloads the adapter weights if they are not already cached locally.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the Hugging Face model ID (e.g. ``"granite-3.3-8b-instruct"``).

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
        # TODO(phase-2.2): pass revision=self.intrinsic_metadata.revision once
        # revision-aware prepare() is merged (issue #1141 / epic #929).
        return str(
            intrinsics.obtain_lora(
                self.intrinsic_name,
                base_model_name,
                self.intrinsic_metadata.repo_id,
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
        intrinsic_name (str): The name of the adapter function, e.g. ``"answerability"``.
        intrinsic_adapter_types (list[AdapterType] | tuple[AdapterType, ...]): The
            adapter types allowed for this adapter function, e.g.
            ``[AdapterType.ALORA, AdapterType.LORA]``.
        available_adapters (dict[str, T]): The available adapters to choose from;
            maps ``adapter.qualified_name`` to the adapter object.

    Returns:
        T | None: The first matching adapter found, or ``None`` if no match exists.
    """
    adapter = None
    for adapter_type in intrinsic_adapter_types:
        qualified_name = f"{intrinsic_name}_{adapter_type.value}"
        adapter = available_adapters.get(qualified_name)
        if adapter is not None:
            break

    return adapter


class AdapterMixin(Backend, abc.ABC):
    """Mixin class for backends capable of utilizing adapters.

    Three verbs are universal across every adapter reality (LocalFile/PEFT,
    Embedded/Granite Switch, ServerMediated): ``base_model_name``,
    ``add_adapter``, and ``list_adapters``. The remaining four verbs are
    reality-specific — a concrete backend overrides only the verb(s) matching
    its own reality; the others keep raising ``NotImplementedError``.

    Attributes:
        base_model_name (str): The short model name used to identify adapter
            variants (e.g. ``"granite-3.3-8b-instruct"`` for
            ``"ibm-granite/granite-3.3-8b-instruct"``).
    """

    # ---- Universal verbs (every adapter reality) ----

    @property
    @abc.abstractmethod
    def base_model_name(self) -> str:
        """Return the short model name used for adapter variant lookup.

        Returns:
            str: The base model name (e.g. ``"granite-3.3-8b-instruct"``).
        """

    @abc.abstractmethod
    def add_adapter(self, adapter: "Adapter | _AdapterCore") -> None:
        """Register an adapter with this backend so it can be loaded later.

        The adapter must not already have been added to a different backend.

        Args:
            adapter (Adapter | _AdapterCore): The adapter to register with
                this backend.
        """

    @abc.abstractmethod
    def list_adapters(self) -> list[str]:
        """Return the qualified names of all adapters registered with this backend.

        Returns:
            list[str]: Qualified adapter names for all adapters that have been
                registered via ``add_adapter``.

        Raises:
            NotImplementedError: If the concrete backend subclass has not
                implemented this method.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not implement list_adapters() API call."
        )

    # ---- Reality-specific verbs ----

    def load_peft_adapter(self, adapter_qualified_name: str) -> None:
        """Load a previously registered PEFT adapter into the underlying model.

        LocalFile/PEFT reality only (e.g. a locally hosted Hugging Face
        model). The adapter must have been registered via ``add_adapter``
        before calling this method.

        Args:
            adapter_qualified_name (str): The ``adapter.qualified_name`` of the
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
            adapter_qualified_name (str): The ``adapter.qualified_name`` of the
                adapter to unload.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                LocalFile/PEFT.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support unload_peft_adapter()."
        )

    def render_controls(self, adapter_qualified_name: str, active: bool) -> None:
        """Render or clear the control tokens for a baked-in embedded adapter.

        Embedded/Granite Switch reality only. Weights are already baked into
        the model; this only toggles the control-token rendering that
        activates or deactivates the adapter's behaviour for subsequent
        requests.

        Args:
            adapter_qualified_name (str): The ``adapter.qualified_name`` of the
                adapter to activate or deactivate.
            active (bool): ``True`` to render the adapter's control tokens,
                ``False`` to clear them.

        Raises:
            NotImplementedError: If this backend's adapter reality is not
                Embedded/Granite Switch.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support render_controls()."
        )

    def set_request_adapter(self, adapter_qualified_name: str) -> None:
        """Select the adapter to use for the next request.

        ServerMediated reality only — for servers that accept an adapter
        selection per request rather than loading/unloading weights or
        toggling control tokens locally. No backend implements this reality
        yet.

        Args:
            adapter_qualified_name (str): The ``adapter.qualified_name`` of the
                adapter to select.

        Raises:
            NotImplementedError: Always — the ServerMediated adapter reality
                has no implementation yet.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not support set_request_adapter(); "
            "the ServerMediated adapter reality is not implemented yet."
        )

    def resolve_adapter(self, name: str) -> _AdapterCore:
        """Find or lazily register an adapter by capability name.

        Default implementation preserves Phase 0 behaviour, using the internal
        ``_added_adapters`` dict that concrete backends maintain.  Override in
        Phase 2 (see epic #929) to implement proper lifecycle management.

        Args:
            name (str): Capability name (e.g. ``"answerability"``).

        Returns:
            _AdapterCore: The registered adapter with the given capability.

        Raises:
            ValueError: If the backend has no model ID.
            KeyError: If the adapter cannot be found after registration.
        """
        found = self._find_adapter(name)
        if found is not None:
            return found

        base = self.base_model_name
        if base is None:
            raise ValueError(
                f"Backend has no model ID; cannot resolve adapter {name!r}"
            )

        # warnings.catch_warnings() modifies the process-global filter state and is not
        # async/thread-safe.  Concurrent first-time resolves race on filter restoration;
        # add_adapter is idempotent so the double-registration hazard is benign, but the
        # filter race is a known Phase-1 gap: two concurrent first-time call_intrinsic
        # calls can interleave their catch_warnings contexts, causing a DeprecationWarning
        # to surface in user code during lazy-registration.  Phase 2 (see epic #929) adds a lock.
        # Suppress DeprecationWarning: the shim constructors warn user-facing code,
        # not internal registration paths.
        with warnings.catch_warnings():
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
                    # EmbeddedIntrinsicAdapter is only valid for backends whose
                    # add_adapter accepts the full Adapter type (e.g. OpenAIBackend).
                    # LocalHFBackend.add_adapter expects LocalHFAdapter; HF backends
                    # never set _uses_embedded_adapters=True.
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

        raise KeyError(f"Adapter {name!r} not found after registration")

    @contextlib.contextmanager
    def adapter_scope(self, adapter: "_AdapterCore | None"):  # type: ignore[type-arg]
        """Context manager wrapping adapter activation and deactivation.

        Phase 1 stub — yields immediately (no-op). Phase 2 (see epic #929) wires
        in ``adapter.weights.activate()`` and ``adapter.weights.deactivate()``.

        Args:
            adapter: The adapter to activate, or ``None`` (no-op in Phase 1).
        """
        yield

    def _find_adapter(
        self, capability: str, adapter_types: tuple[str, ...] | None = None
    ) -> "_AdapterCore | None":
        """Return the first registered adapter matching capability and (optionally) type.

        Args:
            capability (str): Capability name (e.g. ``"answerability"``).
            adapter_types (tuple[str, ...] | None): Adapter type strings in
                preference order (e.g. ``("alora", "lora")``).  When provided,
                aLoRA is returned before LoRA if both are registered for the same
                capability.  ``None`` matches any type (insertion order wins).

        Returns:
            _AdapterCore | None: Matching adapter, or ``None`` if not found.
        """
        adapters = getattr(self, "_added_adapters", {})
        if adapter_types is None:
            for a in adapters.values():
                if isinstance(a, _AdapterCore) and a.identity.capability == capability:
                    return a
            return None
        for preferred_type in adapter_types:
            for a in adapters.values():
                if (
                    isinstance(a, _AdapterCore)
                    and a.identity.capability == capability
                    and a.identity.adapter_type == preferred_type
                ):
                    return a
        return None


class EmbeddedIntrinsicAdapter(_AdapterCore):
    """Deprecated shim for adapter functions embedded in a Granite Switch model.

    .. deprecated::
        Use :class:`~mellea.backends.adapters.Adapter` directly.
        ``EmbeddedIntrinsicAdapter`` will be removed in a future release
        (Epic #929, issue #1144).

    Unlike PEFT-based adapters that are loaded into the model at runtime,
    embedded adapters are already baked into the model weights and activated
    via control tokens injected by the model's chat template.  Only the I/O
    transformation config (``io.yaml``) is needed; no adapter weights are
    downloaded or loaded.

    Args:
        intrinsic_name (str): Name of the adapter function (e.g. ``"answerability"``).
        config (dict): Parsed I/O transformation configuration (from ``io.yaml``).
        technology (str): Adapter technology in the switch model — ``"lora"`` or
            ``"alora"``.  Determines where the control token is placed in the
            chat template (beginning of sequence for LoRA, before generation
            prompt for aLoRA).

    Attributes:
        intrinsic_name (str): Name of the adapter function this adapter implements.
        config (dict): Parsed I/O transformation configuration.
        technology (str): ``"lora"`` or ``"alora"``.

    .. note::
        ``identity``, ``io_contract``, and ``weights`` are Phase 1 internal scaffolding
        populated in ``__init__`` to satisfy the new :class:`~mellea.backends.adapters.Adapter`
        protocol.  They are not meaningful consumer-facing attributes; ``io_contract`` and
        ``weights`` raise :exc:`NotImplementedError` and will be replaced in Phase 2
        (issues #1137, #1142).
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

        # Populate the new Adapter triple so isinstance(self, _AdapterCore) holds.
        # technology is validated above; cast to the Literal type mypy expects.
        _AdapterCore.__init__(
            self,
            identity=Identity(
                name=intrinsic_name,
                adapter_type=cast(Literal["lora", "alora"], technology),
                capability=intrinsic_name,
            ),
            io_contract=_ShimIOContract(),
            weights=_ShimWeightsBinding(),
        )

    @staticmethod
    def from_model_directory(
        model_path: str | pathlib.Path, intrinsic_name: str | None = None
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a Granite Switch model directory.

        Reads ``adapter_index.json`` and the corresponding ``io_configs/*/io.yaml``
        files from the model directory.

        Args:
            model_path (str | pathlib.Path): Path to a Granite Switch model
                directory that contains ``adapter_index.json`` and ``io_configs/``.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. ``None`` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.

        Raises:
            FileNotFoundError: If ``adapter_index.json`` is missing.
            ValueError: If an ``io.yaml`` file listed in the index cannot be found
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

        Downloads ``adapter_index.json`` and the ``io_configs/`` directory, then
        delegates to :meth:`from_model_directory`.

        Args:
            repo_id (str): Hugging Face Hub repository ID
                (e.g. ``"ibm-granite/granite-switch-micro"``).
            revision (str): Git revision to download from.
            cache_dir (str | None): Local cache directory; ``None`` for the default.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. ``None`` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
            FileNotFoundError: If ``adapter_index.json`` is missing (delegated
                from :meth:`from_model_directory`).
            ValueError: If no adapters are found (delegated from
                :meth:`from_model_directory`).
        """
        try:
            import huggingface_hub
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download embedded adapter configs from "
                'Hugging Face Hub. Please install it with: pip install "mellea[switch]"'
            ) from e

        local_root = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            allow_patterns=["adapter_index.json", "io_configs/**"],
            cache_dir=cache_dir,
            revision=revision,
        )
        try:
            return EmbeddedIntrinsicAdapter.from_model_directory(
                local_root, intrinsic_name=intrinsic_name
            )
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

        Automatically detects whether ``source`` is a local filesystem path
        or a Hugging Face Hub repo ID, and delegates accordingly.

        Args:
            source (str): Local path to a model directory, or a Hugging Face
                Hub repo ID (e.g. ``"ibm-granite/granite-switch-micro"``).
            revision (str): Git revision (only used for Hub downloads).
            cache_dir (str | None): Cache directory (only used for Hub downloads).
            intrinsic_name (str | None): If provided, only load the adapter
                matching this adapter function name. ``None`` loads all adapters.

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
        ``CustomIntrinsicAdapter`` will be removed in a future release
        (Epic #929, issue #1144).

    This class has the same functionality as ``IntrinsicAdapter``, except that
    its constructor monkey-patches Mellea global variables to enable the backend
    to load the user's adapter.

    Args:
        model_id (str): The Hugging Face model ID used for downloading model weights;
            expected format is ``"<user-id>/<repo-name>"``.
        intrinsic_name (str | None): Catalog name for the adapter function; defaults to the
            repository name portion of ``model_id`` if not provided.
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
