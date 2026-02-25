import pathlib

import pytest

from mellea.backends.adapters import GraniteCommonAdapter


# The backend tests handle most of the adapter testing. Do a basic test here
# to make sure init and config loading work.
def test_adapter_init():
    dir_file = pathlib.Path(__file__).parent.joinpath("intrinsics-data")
    answerability_file = f"{dir_file}/answerability.yaml"

    adapter = GraniteCommonAdapter("answerability", config_file=answerability_file)

    assert adapter.config is not None
    assert adapter.config["parameters"]["max_completion_tokens"] == 6


if __name__ == "__main__":
    pytest.main([__file__])
