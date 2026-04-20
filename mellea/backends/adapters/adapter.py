"""Adapter classes for adding fine-tuned modules to inference backends.

Defines the abstract ``Adapter`` base class and its concrete subclasses
``LocalHFAdapter`` (for locally loaded HuggingFace models) and ``IntrinsicAdapter``
(for adapters whose metadata is stored in Mellea's intrinsic catalog). Also provides
``get_adapter_for_intrinsic`` for resolving the right adapter class given an
intrinsic name, and ``AdapterMixin`` for backends that support runtime adapter
loading and unloading.
"""

import abc
import pathlib
import re
from typing import TypeVar

import yaml

from ...core import Backend
from ...formatters.granite import intrinsics as intrinsics
from ...helpers import _ServerType
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
    """Abstract adapter subclass for locally loaded HuggingFace model backends.

    Subclasses must implement ``get_local_hf_path`` to return the filesystem path
    from which adapter weights should be loaded given a base model name.
    """

    @abc.abstractmethod
    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the HuggingFace model ID (e.g. ``"granite-4.0-micro"``).

        Returns:
            str: Filesystem path to the adapter weights directory.
        """
        ...


class IntrinsicAdapter(LocalHFAdapter):
    """Base class for adapters that implement intrinsics.

    Subtype of :class:`Adapter` for models that:

    * implement intrinsic functions
    * are packaged as LoRA or aLoRA adapters on top of a base model
    * use the shared model loading code in ``mellea.formatters.granite.intrinsics``
    * use the shared input and output processing code in
      ``mellea.formatters.granite.intrinsics``

    Args:
        intrinsic_name (str): Name of the intrinsic (e.g. ``"answerability"``); the
            adapter's ``qualified_name`` will be derived from this.
        adapter_type (AdapterType): Enum describing the adapter type; defaults to
            ``AdapterType.ALORA``.
        config_file (str | pathlib.Path | None): Path to a YAML config file defining
            the intrinsic's I/O transformations; mutually exclusive with
            ``config_dict``.
        config_dict (dict | None): Dict defining the intrinsic's I/O transformations;
            mutually exclusive with ``config_file``.
        base_model_name (str | None): Base model name used to look up the I/O
            processing config when neither ``config_file`` nor ``config_dict`` are
            provided.

    Attributes:
        intrinsic_name (str): Name of the intrinsic this adapter implements.
        intrinsic_metadata (IntriniscsCatalogEntry): Catalog metadata for the intrinsic.
        base_model_name (str | None): Base model name provided at construction, if any.
        adapter_type (AdapterType): The adapter type (``LORA`` or ``ALORA``).
        config (dict): Parsed I/O transformation configuration for the intrinsic.
    """

    def __init__(
        self,
        intrinsic_name: str,
        adapter_type: AdapterType = AdapterType.ALORA,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        base_model_name: str | None = None,
    ):
        """Initialize IntrinsicAdapter for the named intrinsic, loading its I/O configuration."""
        super().__init__(intrinsic_name, adapter_type)

        self.intrinsic_name = intrinsic_name
        self.intrinsic_metadata = fetch_intrinsic_metadata(intrinsic_name)
        self.base_model_name = base_model_name

        if adapter_type not in self.intrinsic_metadata.adapter_types:
            raise ValueError(
                f"Intrinsic '{intrinsic_name}' not available as an adapter of type "
                f"'{adapter_type}. Available types are "
                f"{self.intrinsic_metadata.adapter_types}."
            )
        self.adapter_type = adapter_type

        # If any of the optional params are specified, attempt to set up the
        # config for the intrinsic here.
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

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local filesystem path from which adapter weights should be loaded.

        Downloads the adapter weights if they are not already cached locally.

        Args:
            base_model_name (str): The base model name; typically the last component
                of the HuggingFace model ID (e.g. ``"granite-3.3-8b-instruct"``).

        Returns:
            str: Filesystem path to the downloaded adapter weights directory.
        """
        return self.download_and_get_path(base_model_name)

    def download_and_get_path(self, base_model_name: str) -> str:
        """Downloads the required rag intrinsics files if necessary and returns the path to them.

        Args:
            base_model_name: the base model; typically the last part of the huggingface
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
                alora=is_alora,
            )
        )


T = TypeVar("T")


def get_adapter_for_intrinsic(
    intrinsic_name: str,
    intrinsic_adapter_types: list[AdapterType] | tuple[AdapterType, ...],
    available_adapters: dict[str, T],
) -> T | None:
    """Find an adapter from a dict of available adapters based on the intrinsic name and its allowed adapter types.

    Args:
        intrinsic_name (str): The name of the intrinsic, e.g. ``"answerability"``.
        intrinsic_adapter_types (list[AdapterType] | tuple[AdapterType, ...]): The
            adapter types allowed for this intrinsic, e.g.
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

    Attributes:
        base_model_name (str): The short model name used to identify adapter
            variants (e.g. ``"granite-3.3-8b-instruct"`` for
            ``"ibm-granite/granite-3.3-8b-instruct"``).
    """

    @property
    @abc.abstractmethod
    def base_model_name(self) -> str:
        """Return the short model name used for adapter variant lookup.

        Returns:
            str: The base model name (e.g. ``"granite-3.3-8b-instruct"``).
        """

    @abc.abstractmethod
    def add_adapter(self, *args, **kwargs):
        """Register an adapter with this backend so it can be loaded later.

        The adapter must not already have been added to a different backend.

        Args:
            args: Positional arguments forwarded to the concrete implementation.
            kwargs: Keyword arguments forwarded to the concrete implementation.
        """

    @abc.abstractmethod
    def load_adapter(self, adapter_qualified_name: str):
        """Load a previously registered adapter into the underlying model.

        The adapter must have been registered via ``add_adapter`` before calling
        this method.

        Args:
            adapter_qualified_name (str): The ``adapter.qualified_name`` of the
                adapter to load.
        """

    @abc.abstractmethod
    def unload_adapter(self, adapter_qualified_name: str):
        """Unload a previously loaded adapter from the underlying model.

        Args:
            adapter_qualified_name (str): The ``adapter.qualified_name`` of the
                adapter to unload.
        """

    @abc.abstractmethod
    def list_adapters(self) -> list[str]:
        """Return the qualified names of all adapters currently loaded in this backend.

        Returns:
            list[str]: Qualified adapter names for all adapters that have been
                loaded via ``load_adapter``.

        Raises:
            NotImplementedError: If the concrete backend subclass has not
                implemented this method.
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not implement list_adapters() API call."
        )


class EmbeddedIntrinsicAdapter(Adapter):
    """Adapter for intrinsics embedded in a Granite Switch model.

    Unlike PEFT-based adapters that are loaded into the model at runtime,
    embedded adapters are already baked into the model weights and activated
    via control tokens injected by the model's chat template.  Only the I/O
    transformation config (``io.yaml``) is needed; no adapter weights are
    downloaded or loaded.

    Args:
        intrinsic_name (str): Name of the intrinsic (e.g. ``"answerability"``).
        config (dict): Parsed I/O transformation configuration (from ``io.yaml``).
        technology (str): Adapter technology in the switch model — ``"lora"`` or
            ``"alora"``.  Determines where the control token is placed in the
            chat template (beginning of sequence for LoRA, before generation
            prompt for aLoRA).

    Attributes:
        intrinsic_name (str): Name of the intrinsic this adapter implements.
        config (dict): Parsed I/O transformation configuration.
        technology (str): ``"lora"`` or ``"alora"``.
    """

    def __init__(self, intrinsic_name: str, config: dict, technology: str = "lora"):
        """Initialize an embedded intrinsic adapter with its I/O config."""
        if technology not in ("lora", "alora"):
            raise ValueError(
                f"technology must be 'lora' or 'alora', got '{technology}'"
            )
        adapter_type = AdapterType.ALORA if technology == "alora" else AdapterType.LORA
        super().__init__(intrinsic_name, adapter_type)
        self.intrinsic_name = intrinsic_name
        self.config = config
        self.technology = technology

    @staticmethod
    def from_model_directory(
        model_path: str | pathlib.Path,
        intrinsic_name: str | None = None,
    ) -> list["EmbeddedIntrinsicAdapter"]:
        """Load embedded adapters from a Granite Switch model directory.

        Reads ``adapter_index.json`` and the corresponding ``io_configs/*/io.yaml``
        files from the model directory.

        Args:
            model_path (str | pathlib.Path): Path to a Granite Switch model
                directory that contains ``adapter_index.json`` and ``io_configs/``.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this intrinsic name. ``None`` loads all adapters.

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
            if intrinsic_name is not None and entry.get("intrinsic_name") != intrinsic_name:
                continue
            io_config_rel = entry.get("io_config")
            if io_config_rel is None:
                continue

            io_config_path = model_path / io_config_rel
            if not io_config_path.exists():
                raise ValueError(
                    f"io.yaml for intrinsic '{entry['intrinsic_name']}' "
                    f"not found at {io_config_path}"
                )

            with open(io_config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            adapters.append(
                EmbeddedIntrinsicAdapter(
                    intrinsic_name=entry["intrinsic_name"],
                    config=config_dict,
                    technology=entry.get("technology", "lora"),
                )
            )

        if not adapters:
            if intrinsic_name is not None:
                raise ValueError(
                    f"No adapter found for intrinsic '{intrinsic_name}' "
                    f"in {model_path}"
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
        """Load embedded adapters from a Granite Switch model on HuggingFace Hub.

        Downloads ``adapter_index.json`` and the ``io_configs/`` directory, then
        delegates to :meth:`from_model_directory`.

        Args:
            repo_id (str): HuggingFace Hub repository ID
                (e.g. ``"ibm-granite/granite-switch-micro"``).
            revision (str): Git revision to download from.
            cache_dir (str | None): Local cache directory; ``None`` for the default.
            intrinsic_name (str | None): If provided, only load the adapter
                matching this intrinsic name. ``None`` loads all adapters.

        Returns:
            list[EmbeddedIntrinsicAdapter]: One adapter per entry in the index.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
            ValueError: If no adapters are found (delegated from
                :meth:`from_model_directory`).
        """
        try:
            import huggingface_hub
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to load adapters from the Hub. "
                "Please install it: pip install huggingface_hub"
            ) from e

        local_root = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            allow_patterns=["adapter_index.json", "io_configs/**"],
            cache_dir=cache_dir,
            revision=revision,
        )
        try:
            return EmbeddedIntrinsicAdapter.from_model_directory(local_root, intrinsic_name=intrinsic_name)
        except ValueError:
            if intrinsic_name is not None:
                raise ValueError(
                    f"No adapter found for intrinsic '{intrinsic_name}' "
                    f"in {repo_id}"
                ) from None
            raise ValueError(f"No adapters found in {repo_id}") from None


class CustomIntrinsicAdapter(IntrinsicAdapter):
    """Special class for users to subclass when creating custom intrinsic adapters.

    The documentation says that any developer who creates an intrinsic should create
    a subclass of this class. Creating a subclass of this class appears to be a cosmetic
    boilerplate development task that isn't actually necessary for any existing use case.

    This class has the same functionality as ``IntrinsicAdapter``, except that its
    constructor monkey-patches Mellea global variables to enable the backend to load
    the user's adapter. The code that performs this monkey-patching is marked as a
    temporary hack.

    Args:
        model_id (str): The HuggingFace model ID used for downloading model weights;
            expected format is ``"<user-id>/<repo-name>"``.
        intrinsic_name (str | None): Catalog name for the intrinsic; defaults to the
            repository name portion of ``model_id`` if not provided.
        base_model_name (str): The short name of the base model (NOT its repo ID).
    """

    def __init__(
        self, *, model_id: str, intrinsic_name: str | None = None, base_model_name: str
    ):
        """Initialize CustomIntrinsicAdapter and patch the global intrinsics catalog if needed."""
        assert re.match(".*/.*", model_id), (
            "expected a huggingface model id with format <user-id>/<repo-name>"
        )
        intrinsic_name = (
            intrinsic_name if intrinsic_name is not None else model_id.split("/")[1]
        )

        # patch the catalog. TODO this is a temporary hack until we re-org adapters.
        from mellea.backends.adapters import catalog

        if intrinsic_name not in catalog._INTRINSICS_CATALOG:
            catalog._INTRINSICS_CATALOG_ENTRIES.append(
                catalog.IntriniscsCatalogEntry(name=intrinsic_name, repo_id=model_id)
            )
            catalog._INTRINSICS_CATALOG = {
                e.name: e for e in catalog._INTRINSICS_CATALOG_ENTRIES
            }

        super().__init__(intrinsic_name=intrinsic_name, base_model_name=base_model_name)
