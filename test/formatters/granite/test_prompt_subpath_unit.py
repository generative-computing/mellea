# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the prompt-fallback path helpers."""

import pathlib
from unittest.mock import patch

import pytest

from mellea.formatters.granite.intrinsics.constants import OLD_LAYOUT_REPOS
from mellea.formatters.granite.intrinsics.util import (
    AdapterNotFoundError,
    adapter_subpath,
    resolve_prompt_io_yaml,
)

_NEW_REPO = "ibm-granite/granitelib-rag-r1.0"


def test_adapter_subpath_prompt_new_layout():
    assert (
        adapter_subpath(
            "answerability", "granite-4.2-8b", _NEW_REPO, adapter_type="prompt"
        )
        == "answerability/granite-4.2-8b/prompt"
    )


def test_adapter_subpath_any_prompt_new_layout():
    # "any" is absent from BASE_MODEL_TO_CANONICAL_NAME and passes through as-is,
    # occupying the model slot.
    assert (
        adapter_subpath("answerability", "any", _NEW_REPO, adapter_type="prompt")
        == "answerability/any/prompt"
    )


def test_adapter_subpath_prompt_old_layout():
    # Old layout swaps the type slot ahead of the model slot.
    old_repo = OLD_LAYOUT_REPOS[0]
    assert (
        adapter_subpath(
            "answerability", "granite-4.2-8b", old_repo, adapter_type="prompt"
        )
        == "answerability/prompt/granite-4.2-8b"
    )


def test_adapter_subpath_lora_and_alora_slots():
    # Default slot is lora; alora is passed like any other slot.
    assert (
        adapter_subpath("answerability", "granite-4.1-3b", _NEW_REPO)
        == "answerability/granite-4.1-3b/lora"
    )
    assert (
        adapter_subpath(
            "answerability", "granite-4.1-3b", _NEW_REPO, adapter_type="alora"
        )
        == "answerability/granite-4.1-3b/alora"
    )


def test_resolve_prompt_io_yaml_prefers_model_specific():
    model_path = pathlib.Path("/cache/answerability/granite-4.2-8b/prompt/io.yaml")
    with patch(
        "mellea.formatters.granite.intrinsics.util.obtain_io_yaml",
        return_value=model_path,
    ) as mock_obtain:
        path, used_any = resolve_prompt_io_yaml(
            "answerability", "granite-4.2-8b", _NEW_REPO
        )
    assert path == model_path
    assert used_any is False
    # Only the model-specific probe runs when it succeeds.
    mock_obtain.assert_called_once()
    assert mock_obtain.call_args.args[1] == "granite-4.2-8b"
    assert mock_obtain.call_args.kwargs["adapter_type"] == "prompt"


def test_resolve_prompt_io_yaml_falls_back_to_any():
    any_path = pathlib.Path("/cache/answerability/any/prompt/io.yaml")

    def side_effect(intrinsic, model, repo, /, **kwargs):
        if model == "any":
            return any_path
        raise AdapterNotFoundError("not found")

    with patch(
        "mellea.formatters.granite.intrinsics.util.obtain_io_yaml",
        side_effect=side_effect,
    ):
        path, used_any = resolve_prompt_io_yaml(
            "answerability", "granite-4.2-3b", _NEW_REPO
        )
    assert path == any_path
    assert used_any is True


def test_resolve_prompt_io_yaml_raises_when_neither_exists():
    with patch(
        "mellea.formatters.granite.intrinsics.util.obtain_io_yaml",
        side_effect=AdapterNotFoundError("not found"),
    ):
        with pytest.raises(AdapterNotFoundError):
            resolve_prompt_io_yaml("answerability", "granite-4.2-3b", _NEW_REPO)


def test_resolve_prompt_io_yaml_propagates_non_not_found_error():
    # A non-ValueError from the model-specific probe (e.g. network/auth) must
    # abort, not fall through to the "any" slot.
    def side_effect(intrinsic, model, repo, /, **kwargs):
        if model != "any":
            raise RuntimeError("network down")
        raise AssertionError("must not reach the 'any' probe after a hard error")

    with patch(
        "mellea.formatters.granite.intrinsics.util.obtain_io_yaml",
        side_effect=side_effect,
    ):
        with pytest.raises(RuntimeError):
            resolve_prompt_io_yaml("answerability", "granite-4.2-8b", _NEW_REPO)


def test_resolve_prompt_io_yaml_propagates_valueerror_subclass():
    # A ValueError *subclass* that is not AdapterNotFoundError (e.g. a Hugging Face
    # HFValidationError) from the model-specific probe must propagate, not be
    # mistaken for "not found" and fall through to the "any" slot.
    class FakeHFValidationError(ValueError):
        pass

    def side_effect(intrinsic, model, repo, /, **kwargs):
        if model != "any":
            raise FakeHFValidationError("malformed repo id")
        raise AssertionError("must not reach the 'any' probe after a hard error")

    with patch(
        "mellea.formatters.granite.intrinsics.util.obtain_io_yaml",
        side_effect=side_effect,
    ):
        with pytest.raises(FakeHFValidationError):
            resolve_prompt_io_yaml("answerability", "granite-4.2-8b", _NEW_REPO)
