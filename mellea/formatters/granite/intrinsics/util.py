# SPDX-License-Identifier: Apache-2.0

"""Common utility functions for this package."""

# Standard
import copy
import json
import os
import pathlib

# Third Party
import yaml

# Local
from .constants import (
    BASE_MODEL_TO_CANONICAL_NAME,
    OLD_LAYOUT_REPOS,
    YAML_JSON_FIELDS,
    YAML_OPTIONAL_FIELDS,
    YAML_REQUIRED_FIELDS,
)


def make_config_dict(
    config_file: str | pathlib.Path | None = None, config_dict: dict | None = None
) -> dict | None:
    """Create a configuration dictionary from YAML file or dict.

    This function is not a public API and is not intended for use outside this library.

    Common initialization code for reading YAML config files in factory classes.
    Also parses JSON fields.

    Args:
        config_file: Path to a YAML configuration file. Exactly one of ``config_file``
            and ``config_dict`` must be provided.
        config_dict: Pre-parsed configuration dict (from ``yaml.safe_load()``). Exactly
            one of ``config_file`` and ``config_dict`` must be provided.

    Returns:
        Validated configuration dict with optional fields set to ``None`` and JSON
        string fields parsed to Python objects.

    Raises:
        ValueError: If both or neither of ``config_file`` and ``config_dict`` are
            provided, if a required field is missing, if an unexpected top-level
            field is encountered, or if a JSON field cannot be parsed.
    """
    if (config_file is None and config_dict is None) or (
        config_file is not None and config_dict is not None
    ):
        raise ValueError("Exactly one of config_file and config_dict must be set.")

    all_fields = sorted(YAML_REQUIRED_FIELDS + YAML_OPTIONAL_FIELDS)

    result_dict: dict | None = None
    if config_dict:
        # Don't modify input
        result_dict = copy.deepcopy(config_dict)
    if config_file:
        with open(config_file, encoding="utf8") as file:
            result_dict = yaml.safe_load(file)

    # Validate top-level field names. No schema checking for YAML, so we need to do this
    # manually.
    if result_dict is None:
        raise ValueError("No configuration provided")
    for field in YAML_REQUIRED_FIELDS:
        if field not in result_dict:
            raise ValueError(f"Configuration is missing required field '{field}'")
    for name in result_dict:
        if name not in all_fields:
            raise ValueError(
                f"Configuration contains unexpected top-level field "
                f"'{name}'. Known top level fields are: {all_fields}"
            )
    for name in YAML_OPTIONAL_FIELDS:
        # Optional fields should be None if not present, to simplify downstream code.
        if name not in result_dict:
            result_dict[name] = None

    # Parse fields that contain JSON data.
    for name in YAML_JSON_FIELDS:
        if result_dict[name]:
            value = result_dict[name]
            # Users seem to be intent on passing YAML data through this function
            # multiple times, so we assume that values other than a string have already
            # been parsed by a previous call of this function.
            if isinstance(value, str):
                try:
                    result_dict[name] = json.loads(value)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Error parsing JSON in '{name}' field. Raw value was '{value}'"
                    ) from e

    return result_dict


def obtain_lora(
    intrinsic_name: str,
    target_model_name: str,
    repo_id: str,
    /,
    revision: str = "main",
    alora: bool = False,
    cache_dir: str | None = None,
    file_glob: str = "*",
) -> pathlib.Path:
    """Download and cache an adapter that implements and intrinsic.

    Downloads a LoRA or aLoRA adapter from a collection of adapters that follow the
    same layout as the [Granite Intrinsics Library](
    https://huggingface.co/ibm-granite/granitelib-rag-r1.0). Caches the downloaded
    adapter files on local disk.

    Args:
        intrinsic_name: Short name of the intrinsic model, such as ``"certainty"``.
        target_model_name: Name of the base model for the LoRA or aLoRA adapter.
        repo_id: Hugging Face Hub repository containing a collection of LoRA and/or
            aLoRA adapters for intrinsics.
        revision: Git revision of the repository to download from.
        alora: If ``True``, load the aLoRA version of the intrinsic; otherwise use LoRA.
        cache_dir: Local directory to use as a cache (Hugging Face Hub format), or
            ``None`` to use the default location.
        file_glob: Only files matching this glob will be downloaded to the cache.

    Returns:
        Full path to the local copy of the specified (a)LoRA adapter, suitable for
        passing to commands that serve the adapter.

    Raises:
        ValueError: If the specified intrinsic adapter cannot be found in the
            Hugging Face Hub repository at the expected path.
    """
    # Third Party
    import huggingface_hub

    # Normalize target model name if a normalization exists.
    target_model_name = BASE_MODEL_TO_CANONICAL_NAME.get(
        target_model_name, target_model_name
    )

    lora_str = "alora" if alora else "lora"

    if repo_id in OLD_LAYOUT_REPOS:
        # Old repository layout
        lora_subdir_name = f"{intrinsic_name}/{lora_str}/{target_model_name}"
    else:
        # Assume new layout otherwise
        lora_subdir_name = f"{intrinsic_name}/{target_model_name}/{lora_str}"

    # Download just the files for this LoRA if not already present
    local_root_path = huggingface_hub.snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{lora_subdir_name}/{file_glob}",
        cache_dir=cache_dir,
        revision=revision,
    )
    lora_dir = pathlib.Path(local_root_path) / lora_subdir_name

    # Hugging Face Hub API will happily download nothing. Check whether that happened.
    if not os.path.exists(lora_dir):
        raise ValueError(
            f"Intrinsic '{intrinsic_name}' as "
            f"{'aLoRA' if alora else 'LoRA'} adapter on base model "
            f"'{target_model_name}' not found in "
            f"{repo_id} repository on Hugging Face Hub. "
            f"Searched for path {lora_subdir_name}/{file_glob}"
        )

    return lora_dir


def obtain_io_yaml(
    intrinsic_name: str,
    target_model_name: str,
    repo_id: str,
    /,
    revision: str = "main",
    alora: bool = False,
    cache_dir: str | None = None,
) -> pathlib.Path:
    """Download cached ``io.yaml`` configuration file for an intrinsic.

    Downloads an ``io.yaml`` configuration file for an intrinsic
    with a model repository that follows the format of the
    [Granite Intrinsics Library](
    https://huggingface.co/ibm-granite/granitelib-rag-r1.0) if one is not
    already in the local cache.

    Args:
        intrinsic_name: Short name of the intrinsic model, such as ``"certainty"``.
        target_model_name: Name of the base model for the LoRA or aLoRA adapter.
        repo_id: Hugging Face Hub repository containing a collection of LoRA and/or
            aLoRA adapters for intrinsics.
        revision: Git revision of the repository to download from.
        alora: If ``True``, load the aLoRA version of the intrinsic; otherwise use LoRA.
        cache_dir: Local directory to use as a cache (Hugging Face Hub format), or
            ``None`` to use the default location.

    Returns:
        Full path to the local copy of the ``io.yaml`` file, suitable for passing to
        ``IntrinsicsRewriter``.
    """
    lora_dir = obtain_lora(
        intrinsic_name,
        target_model_name,
        repo_id,
        revision=revision,
        alora=alora,
        cache_dir=cache_dir,
        file_glob="io.yaml",
    )
    return lora_dir / "io.yaml"
