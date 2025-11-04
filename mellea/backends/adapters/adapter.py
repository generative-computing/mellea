"""Module for adapters to backends."""

import abc
import pathlib
from enum import Enum
from typing import Any, TypeVar

import granite_common
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
        name: str,
        adapter_type: AdapterType = AdapterType.ALORA,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        base_model_name: str | None = None,
    ):
        """An adapter that can be added to either an `OpenAIBackend` or a `LocalHFBackend`. Most rag-lib-intrinsics support lora or alora adapter types.

        Args:
            name: name of the adapter; when referencing this adapter, use adapter.qualified_name
            adapter_type: enum describing what type of adapter it is (ie LORA / ALORA)
            config_file: optional; file for defining the intrinsic / transformations
            config_dict: optional; dict for defining the intrinsic / transformations
            base_model_name: optional; if provided with no config_file/config_dict, will be used to lookup the granite_common config for this adapter
        """
        assert adapter_type == AdapterType.ALORA or adapter_type == AdapterType.LORA, (
            f"{adapter_type} not supported"
        )
        super().__init__(name, adapter_type)

        self.base_model_name = base_model_name

        # If any of the optional params are specified, attempt to set up the
        # config for the intrinsic here.
        config: dict | None = None
        if config_file is not None or config_dict is not None:
            config = granite_common.intrinsics.util.make_config_dict(
                config_file=config_file, config_dict=config_dict
            )
            config = cast(
                dict, config
            )  # Can remove if util function gets exported properly.

        if config is None and self.base_model_name is not None:
            is_alora = True if self.adapter_type == AdapterType.ALORA else False
            io_yaml_file = granite_common.intrinsics.util.obtain_io_yaml(
                self.name, self.base_model_name, alora=is_alora
            )
            config = granite_common.intrinsics.util.make_config_dict(
                config_file=io_yaml_file
            )
            config = cast(
                dict, config
            )  # Can remove if util function gets exported properly.

        self.config: dict | None = config

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
                self.name, base_model_name, alora=is_alora
            )
        )

    def get_path_on_remote(self, base_model_name: str, base_path: str) -> str:
        """Assumes the files have already been downloaded on the remote server."""
        return f"./{base_path}/{self.name}/{self.adapter_type.value}/{base_model_name}"


T = TypeVar("T")


def get_adapter_for_intrinsic(
    intrinsic_name: str,
    intrinsic_adapter_types: list[AdapterType],
    available_adapters: dict[str, T],
) -> T | None:
    """Finds an adapter from a dict of available adapters based on the intrinsic name and its allowed adapter types.

    Args:
        intrinsic_name: the name of the intrinsic, like "answerability"
        intrinsic_adapter_types: the adapter types allowed for this intrinsic, like ALORA / LORA
        available_adapters: the available adapters to choose from; maps adapter.qualified_name to the Adapter

    Returns:
        an Adapter if found; else None
    """
    adapter = None
    for adapter_type in intrinsic_adapter_types:
        qualified_name = intrinsic_name + "_" + adapter_type.value
        adapter = available_adapters.get(qualified_name, None)
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
