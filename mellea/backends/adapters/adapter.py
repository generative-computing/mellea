"""Module for adapters to backends."""

import abc
import pathlib
import re
from typing import TypeVar

import granite_common.intrinsics
import yaml

from ...core import Backend
from ...helpers import _ServerType
from .catalog import AdapterType, fetch_intrinsic_metadata


class Adapter(abc.ABC):
    """An adapter that can be added to a single backend."""

    def __init__(self, name: str, adapter_type: AdapterType):
        """An adapter that can be added to a backend.

        Note: An adapter can only be added to a single backend.

        Args:
            name: name of the adapter; when referencing this adapter, use
                adapter.qualified_name
            adapter_type: enum describing what type of adapter it is (ie LORA / ALORA)
        """
        self.name = name
        self.adapter_type = adapter_type
        self.qualified_name = name + "_" + adapter_type.value
        """the name of the adapter to use when loading / looking it up"""

        self.backend: Backend | None = None
        """set when the adapter is added to a backend"""

        self.path: str | None = None
        """set when the adapter is added to a backend"""


class OpenAIAdapter(Adapter):
    """Adapter for OpenAIBackends."""

    @abc.abstractmethod
    def get_open_ai_path(
        self,
        base_model_name: str,
        server_type: _ServerType = _ServerType.LOCALHOST,
        remote_path: str | None = None,
    ) -> str:
        """Returns the path needed to load the adapter.

        Args:
            base_model_name: the base model; typically the last part of the huggingface model id like "granite-3.3-8b-instruct"
            server_type: the server type (ie LOCALHOST / OPENAI); usually the backend has information on this
            remote_path: optional; used only if the server_type is REMOTE_VLLM; base path at which to find the adapter
        """
        ...


class LocalHFAdapter(Adapter):
    """Adapter for LocalHFBackends."""

    @abc.abstractmethod
    def get_local_hf_path(self, base_model_name: str) -> str:
        """Returns the path needed to load the adapter.

        Args:
            base_model_name: the base model; typically the last part of the huggingface model id like "granite-3.3-8b-instruct"
        """
        ...


class GraniteCommonAdapter(OpenAIAdapter, LocalHFAdapter):
    """Adapter for intrinsics that utilize the ``granite-common`` library."""

    def __init__(
        self,
        intrinsic_name: str,
        adapter_type: AdapterType = AdapterType.ALORA,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        base_model_name: str | None = None,
    ):
        """Entry point for creating GraniteCommonAdapter objects.

        An adapter that can be added to either an `OpenAIBackend` or a `LocalHFBackend`.
        Most intrinsics support LoRA or aLoRA adapter types.

        Args:
            intrinsic_name: name of the intrinsic; the local name of the loaded adapter
                that implements this intrinsic will be adapter.qualified_name
            adapter_type: enum describing what type of adapter it is (ie LORA / ALORA)
            config_file: optional; file for defining the intrinsic / transformations
            config_dict: optional; dict for defining the intrinsic / transformations
            base_model_name: optional; if provided with no config_file/config_dict,
                will be used to look up the granite_common config for this adapter
        """
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
            config_file = granite_common.intrinsics.obtain_io_yaml(
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

    def get_open_ai_path(
        self,
        base_model_name: str,
        server_type: _ServerType = _ServerType.LOCALHOST,
        remote_path: str | None = None,
    ) -> str:
        """Returns the path needed to load the adapter.

        Args:
            base_model_name: the base model; typically the last part of the huggingface
                model id like "granite-3.3-8b-instruct"
            server_type: the server type (ie LOCALHOST / OPENAI); usually the backend
                has information on this
            remote_path: optional; used only if the server_type is REMOTE_VLLM; base
                path at which to find the adapter
        """
        if server_type == _ServerType.LOCALHOST:
            path = self.download_and_get_path(base_model_name)
        elif server_type == _ServerType.REMOTE_VLLM:
            if remote_path is None:
                remote_path = "rag-intrinsics-lib"
            path = self.get_path_on_remote(base_model_name, remote_path)
        else:
            raise ValueError(
                f"{self} not supported for OpenAIBackend with server_type: {server_type}"
            )

        return path

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Returns the path needed to load the adapter.

        Args:
            base_model_name: the base model; typically the last part of the huggingface
                model id like "granite-3.3-8b-instruct"
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
            granite_common.intrinsics.obtain_lora(
                self.intrinsic_name,
                base_model_name,
                self.intrinsic_metadata.repo_id,
                alora=is_alora,
            )
        )

    def get_path_on_remote(self, base_model_name: str, base_path: str) -> str:
        """Assumes the files have already been downloaded on the remote server."""
        # TODO: This will break when we switch to the new repo!!!
        return f"./{base_path}/{self.name}/{self.adapter_type.value}/{base_model_name}"


T = TypeVar("T")


def get_adapter_for_intrinsic(
    intrinsic_name: str,
    intrinsic_adapter_types: list[AdapterType] | tuple[AdapterType, ...],
    available_adapters: dict[str, T],
) -> T | None:
    """Finds an adapter from a dict of available adapters based on the intrinsic name and its allowed adapter types.

    Args:
        repo_id: Name of Hugging Face Hub repository containing the adapters that
                implement the intrinsic
        intrinsic_name: the name of the intrinsic, like "answerability"
        intrinsic_adapter_types: the adapter types allowed for this intrinsic, like ALORA / LORA
        available_adapters: the available adapters to choose from; maps adapter.qualified_name to the Adapter

    Returns:
        an Adapter if found; else None
    """
    adapter = None
    for adapter_type in intrinsic_adapter_types:
        qualified_name = f"{intrinsic_name}_{adapter_type.value}"
        adapter = available_adapters.get(qualified_name)
        if adapter is not None:
            break

    return adapter


class AdapterMixin(Backend, abc.ABC):
    """Mixin class for backends capable of utilizing adapters."""

    @property
    @abc.abstractmethod
    def base_model_name(self) -> str:
        """Returns the base_model_id of the model used by the backend. For example, `granite-3.3-8b-instruct` for `ibm-granite/granite-3.3-8b-instruct`."""

    @abc.abstractmethod
    def add_adapter(self, *args, **kwargs):
        """Adds the given adapter to the backend. Must not have been added to a different backend."""

    @abc.abstractmethod
    def load_adapter(self, adapter_qualified_name: str):
        """Loads the given adapter for the backend. Must have previously been added."""

    @abc.abstractmethod
    def unload_adapter(self, adapter_qualified_name: str):
        """Unloads the given adapter from the backend."""

    @abc.abstractmethod
    def list_adapters(self) -> list[str]:
        """Lists the adapters added via add_adapter().

        :returns: list of adapter names that are currently registered with this backend
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not implement list_adapters() API call."
        )


class CustomGraniteCommonAdapter(GraniteCommonAdapter):
    """A custom granite adapter."""

    def __init__(
        self, *, model_id: str, intrinsic_name: str | None = None, base_model_name: str
    ):
        """Custom granite adapters.

        Args:
            model_id: the huggingface model id. Used for downloading model weights.
            intrinsic_name (optional): catalog name. Defaults to the repo name if none is provided. For example, nfulton/stembolts model_id uses an intrinsic name of stembolts.
            base_model_name: the name (NOT repo_id) of the model.
        """
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
