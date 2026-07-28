# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
from unittest.mock import patch

import pytest

from mellea.backends.adapters import IntrinsicAdapter
from mellea.backends.adapters.catalog import fetch_intrinsic_metadata


# The backend tests handle most of the adapter testing. Do a basic test here
# to make sure init and config loading work.
def test_adapter_init():
    dir_file = pathlib.Path(__file__).parent.joinpath("intrinsics-data")
    answerability_file = f"{dir_file}/answerability.yaml"

    adapter = IntrinsicAdapter("answerability", config_file=answerability_file)

    assert adapter.config is not None
    assert adapter.config["parameters"]["max_completion_tokens"] == 6


def test_init_forwards_pinned_revision_to_obtain_io_yaml():
    """Regression guard (issue #1141): when no config_file/config_dict is given,
    __init__ must forward the catalog's pinned revision to obtain_io_yaml, not the
    default `"main"`.
    """
    pinned_revision = fetch_intrinsic_metadata("answerability").revision

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_io_yaml"
    ) as mock_obtain_io_yaml:
        mock_obtain_io_yaml.return_value = (
            pathlib.Path(__file__).parent / "intrinsics-data" / "answerability.yaml"
        )
        IntrinsicAdapter("answerability", base_model_name="granite-3.3-8b-instruct")

    assert mock_obtain_io_yaml.call_args.kwargs["revision"] == pinned_revision
    assert pinned_revision != "main"


def test_download_and_get_path_forwards_pinned_revision_to_obtain_lora():
    """Regression guard (issue #1141): download_and_get_path must forward the
    catalog's pinned revision to obtain_lora, not the default `"main"`.
    """
    dir_file = pathlib.Path(__file__).parent.joinpath("intrinsics-data")
    answerability_file = f"{dir_file}/answerability.yaml"
    adapter = IntrinsicAdapter("answerability", config_file=answerability_file)
    pinned_revision = adapter.intrinsic_metadata.revision

    with patch("mellea.formatters.granite.intrinsics.obtain_lora") as mock_obtain_lora:
        mock_obtain_lora.return_value = pathlib.Path("/fake/adapter/path")
        adapter.download_and_get_path("granite-3.3-8b-instruct")

    assert mock_obtain_lora.call_args.kwargs["revision"] == pinned_revision
    assert pinned_revision != "main"


if __name__ == "__main__":
    pytest.main([__file__])
