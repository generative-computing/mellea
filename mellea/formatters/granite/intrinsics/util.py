# Copyright IBM Corp. All Rights Reserved.
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


class AdapterNotFoundError(ValueError):
    """Raised when no adapter or prompt `io.yaml` exists at the expected repository path.

    Subclasses `ValueError` so a broad `except ValueError` still catches it, while
    callers can catch `AdapterNotFoundError` specifically to treat "not found" as a
    fallback signal without also swallowing other `ValueError`s.
    """


def make_config_dict(
    config_file: str | pathlib.Path | None = None, config_dict: dict | None = None
) -> dict | None:
    """Create a configuration dictionary from YAML file or dict.

    This function is not a public API and is not intended for use outside this library.

    Common initialization code for reading YAML config files in factory classes.
    Also parses JSON fields.

    Args:
        config_file: Path to a YAML configuration file. Exactly one of `config_file`
            and `config_dict` must be provided.
        config_dict: Pre-parsed configuration dict (from `yaml.safe_load()`). Exactly
            one of `config_file` and `config_dict` must be provided.

    Returns:
        Validated configuration dict with optional fields set to `None` and JSON
        string fields parsed to Python objects.

    Raises:
        ValueError: If both or neither of `config_file` and `config_dict` are
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


def adapter_subpath(
    intrinsic_name: str,
    target_model_name: str,
    repo_id: str,
    /,
    adapter_type: str = "lora",
) -> str:
    """Return the Hugging Face Hub subpath where an intrinsic's adapter lives.

    Encapsulates the layout convention used by the Granite Intrinsics Library and
    related repositories so callers don't replicate the rules. Both `obtain_lora`
    and out-of-tree consumers (e.g. drift checks in tests) should call this function
    rather than building the path themselves.

    Args:
        intrinsic_name: Short name of the intrinsic model, such as `"certainty"`.
        target_model_name: Name of the base model for the adapter, or `"any"` for a
            model-agnostic prompt fallback. May be a raw HF repo ID; canonical
            normalization is applied (`"any"` is not normalized and passes through).
        repo_id: Hugging Face Hub repository containing the adapter collection.
            Used to select between old and new directory layouts.
        adapter_type: Adapter-type slot — `"lora"`, `"alora"`, or `"prompt"`.

    Returns:
        Subpath relative to the repo root, e.g. `"certainty/granite-4.1-3b/lora"` or
        `"answerability/any/prompt"`.
    """
    # Normalize target model name if a normalization exists.
    target_model_name = BASE_MODEL_TO_CANONICAL_NAME.get(
        target_model_name, target_model_name
    )

    if repo_id in OLD_LAYOUT_REPOS:
        # Old repository layout.
        return f"{intrinsic_name}/{adapter_type}/{target_model_name}"

    return f"{intrinsic_name}/{target_model_name}/{adapter_type}"


def obtain_lora(
    intrinsic_name: str,
    target_model_name: str,
    repo_id: str,
    /,
    revision: str = "main",
    cache_dir: str | None = None,
    file_glob: str = "*",
    adapter_type: str = "lora",
) -> pathlib.Path:
    """Download and cache an adapter that implements an intrinsic.

    Downloads a LoRA, aLoRA, or prompt adapter from a collection of adapters that
    follow the same layout as the [Granite Intrinsics Library](
    https://huggingface.co/ibm-granite/granitelib-rag-r1.0). Caches the downloaded
    adapter files on local disk.

    Args:
        intrinsic_name: Short name of the intrinsic model, such as `"certainty"`.
        target_model_name: Name of the base model for the adapter, or `"any"` for a
            model-agnostic prompt fallback.
        repo_id: Hugging Face Hub repository containing a collection of LoRA and/or
            aLoRA adapters for intrinsics.
        revision: Git revision of the repository to download from.
        cache_dir: Local directory to use as a cache (Hugging Face Hub format), or
            `None` to use the default location.
        file_glob: Only files matching this glob will be downloaded to the cache.
        adapter_type: Adapter-type slot — `"lora"`, `"alora"`, or `"prompt"`.

    Returns:
        Full path to the local copy of the specified adapter, suitable for
        passing to commands that serve the adapter.

    Raises:
        AdapterNotFoundError: If the specified intrinsic adapter cannot be found in
            the Hugging Face Hub repository at the expected path.
    """
    # Third Party
    import huggingface_hub

    lora_subdir_name = adapter_subpath(
        intrinsic_name, target_model_name, repo_id, adapter_type=adapter_type
    )

    # Download just the files for this adapter if not already present
    local_root_path = huggingface_hub.snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{lora_subdir_name}/{file_glob}",
        cache_dir=cache_dir,
        revision=revision,
    )
    lora_dir = pathlib.Path(local_root_path) / lora_subdir_name

    # Hugging Face Hub API will happily download nothing. Check whether that happened.
    if not os.path.exists(lora_dir):
        raise AdapterNotFoundError(
            f"Intrinsic '{intrinsic_name}' as {adapter_type!r} adapter on base "
            f"model '{target_model_name}' not found in {repo_id} repository on "
            f"Hugging Face Hub. Searched for path {lora_subdir_name}/{file_glob}"
        )

    return lora_dir


def obtain_io_yaml(
    intrinsic_name: str,
    target_model_name: str,
    repo_id: str,
    /,
    revision: str = "main",
    cache_dir: str | None = None,
    adapter_type: str = "lora",
) -> pathlib.Path:
    """Download cached `io.yaml` configuration file for an intrinsic.

    Downloads an `io.yaml` configuration file for an intrinsic
    with a model repository that follows the format of the
    [Granite Intrinsics Library](
    https://huggingface.co/ibm-granite/granitelib-rag-r1.0) if one is not
    already in the local cache.

    Args:
        intrinsic_name: Short name of the intrinsic model, such as `"certainty"`.
        target_model_name: Name of the base model for the adapter, or `"any"` for a
            model-agnostic prompt fallback.
        repo_id: Hugging Face Hub repository containing a collection of LoRA and/or
            aLoRA adapters for intrinsics.
        revision: Git revision of the repository to download from.
        cache_dir: Local directory to use as a cache (Hugging Face Hub format), or
            `None` to use the default location.
        adapter_type: Adapter-type slot — `"lora"`, `"alora"`, or `"prompt"`.

    Returns:
        Full path to the local copy of the `io.yaml` file, suitable for passing to
        `IntrinsicsRewriter`.
    """
    lora_dir = obtain_lora(
        intrinsic_name,
        target_model_name,
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        file_glob="io.yaml",
        adapter_type=adapter_type,
    )
    return lora_dir / "io.yaml"


def resolve_prompt_io_yaml(
    intrinsic_name: str,
    base_model_name: str,
    repo_id: str,
    /,
    revision: str = "main",
    cache_dir: str | None = None,
) -> tuple[pathlib.Path, bool]:
    """Resolve a prompt fallback's `io.yaml`, preferring a model-specific one.

    Tries the model-specific prompt (`{intrinsic}/{base_model_name}/prompt`) first,
    then the model-agnostic fallback (`{intrinsic}/any/prompt`). Advances from the
    model-specific to the `any` slot only on the not-found `AdapterNotFoundError`;
    any other error (network/auth/bad-revision) from the model-specific fetch
    propagates.

    Args:
        intrinsic_name: Short name of the intrinsic model, such as `"certainty"`.
        base_model_name: Name of the base model to look for a model-specific prompt
            under.
        repo_id: Hugging Face Hub repository containing the prompt collection.
        revision: Git revision of the repository to download from.
        cache_dir: Local directory to use as a cache (Hugging Face Hub format), or
            `None` to use the default location.

    Returns:
        A `(path, used_any)` tuple: the local `io.yaml` path, and `True` if the
        model-agnostic `any` fallback was used rather than a model-specific prompt
        (callers warn on `True`, since an `any` prompt is sub-par).

    Raises:
        AdapterNotFoundError: Neither a model-specific prompt nor the `any` fallback
            exists.
    """
    try:
        return (
            obtain_io_yaml(
                intrinsic_name,
                base_model_name,
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                adapter_type="prompt",
            ),
            False,
        )
    except AdapterNotFoundError:
        pass
    return (
        obtain_io_yaml(
            intrinsic_name,
            "any",
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            adapter_type="prompt",
        ),
        True,
    )
