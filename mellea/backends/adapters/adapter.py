"""Module for adapters to backends."""

import abc
import pathlib
from enum import Enum
from typing import Any, TypeVar

import granite_common
import yaml
from litellm import cast

from mellea.backends import Backend
from mellea.backends.types import _ServerType


class AdapterType(Enum):
    """Possible types of adapters for a backend."""

    LORA = "lora"
    ALORA = "alora"


class Adapter(abc.ABC):
    """An adapter that can be added to a single backend."""

    def __init__(self, name: str, adapter_type: AdapterType):
        """An adapter that can be added to a backend.

        Note: An adapter can only be added to a single backend.

        Args:
            name: name of the adapter; when referencing this adapter, use adapter.qualified_name
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
    """Adapter for intrinsics that utilize the GraniteCommon library."""

    def __init__(
        self,
        repo_id: str,
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
            repo_id: Name of Hugging Face Hub repository containing the adapters that
                implement the intrinsic; for example,
                "generative-computing/rag-intrinsics-lib"
            intrinsic_name: name of the intrinsic; the local name of the loaded adapter
                that implements this intrinsic will be adapter.qualified_name
            adapter_type: enum describing what type of adapter it is (ie LORA / ALORA)
            config_file: optional; file for defining the intrinsic / transformations
            config_dict: optional; dict for defining the intrinsic / transformations
            base_model_name: optional; if provided with no config_file/config_dict,
                will be used to look up the granite_common config for this adapter
        """
        assert adapter_type in (AdapterType.ALORA, AdapterType.LORA), (
            f"{adapter_type} not supported"
        )
        super().__init__(f"{repo_id}_{intrinsic_name}", adapter_type)

        self.repo_id = repo_id
        self.intrinsic_name = intrinsic_name
        self.base_model_name = base_model_name

        # If any of the optional params are specified, attempt to set up the
        # config for the intrinsic here.
        if config_file and config_dict:
            raise ValueError(
                f"Conflicting values for config_file and config_dict "
                f"parameters provided. Values were {config_file=} "
                f"and {config_dict=}"
            )
        if config_file is None and config_dict is None and base_model_name is None:
            raise ValueError(
                "At least one of [config_file, config_dict, base_model_name] "
                "must be provided."
            )
        if config_file is None and config_dict is None:
            is_alora = self.adapter_type == AdapterType.ALORA
            config_file = granite_common.intrinsics.util.obtain_io_yaml(
                self.intrinsic_name,
                self.base_model_name,
                alora=is_alora,
                repo_id=repo_id,
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
            base_model_name: the base model; typically the last part of the huggingface model id like "granite-3.3-8b-instruct"
        """
        return self.download_and_get_path(base_model_name)

    def download_and_get_path(self, base_model_name: str) -> str:
        """Downloads the required rag intrinsics files if necessary and returns the path to the them.

        Args:
            base_model_name: the base model; typically the last part of the huggingface model id like "granite-3.3-8b-instruct"

        Returns:
            a path to the files
        """
        is_alora = self.adapter_type == AdapterType.ALORA
        return str(
            granite_common.intrinsics.util.obtain_lora(
                self.intrinsic_name,
                base_model_name,
                alora=is_alora,
                repo_id=self.repo_id,
            )
        )

    def get_path_on_remote(self, base_model_name: str, base_path: str) -> str:
        """Assumes the files have already been downloaded on the remote server."""
        # TODO: This will break when we switch to the new repo!!!
        return f"./{base_path}/{self.name}/{self.adapter_type.value}/{base_model_name}"


T = TypeVar("T")


def get_adapter_for_intrinsic(
    repo_id: str,
    intrinsic_name: str,
    intrinsic_adapter_types: list[AdapterType],
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
        qualified_name = f"{repo_id}_{intrinsic_name}_{adapter_type.value}"
        adapter = available_adapters.get(qualified_name)
        if adapter is not None:
            break

    return adapter


class AdapterMixin(abc.ABC):
    """Mixin class for backends capable of utilizing adapters."""

    def add_adapter(self, *args, **kwargs):
        """Adds the given adapter to the backend. Must not have been added to a different backend."""

    def load_adapter(self, adapter_qualified_name: str):
        """Loads the given adapter for the backend. Must have previously been added."""

    def unload_adapter(self, adapter_qualified_name: str):
        """Unloads the given adapter from the backend."""

    def list_adapters(self) -> list[str]:
        """Lists the adapters added via add_adapter().

        :returns: list of adapter names that are currently registered with this backend
        """
        raise NotImplementedError(
            f"Backend type {type(self)} does not implement list_adapters() API call."
        )
