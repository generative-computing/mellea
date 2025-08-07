import pytest
from mellea.backends.types import ModelOption


def test_model_option_remove():
    model_opts = {
        ModelOption.CONTEXT_WINDOW: 3,
        ModelOption.MAX_NEW_TOKENS: 2,
        ModelOption.TEMPERATURE: 1,
        "random_opt": "test",
    }

    expected_opts = {ModelOption.TEMPERATURE: 1, "random_opt": "test"}

    processed_model_opts = ModelOption.remove_special_keys(model_opts)
    assert (
        expected_opts == processed_model_opts
    ), "dict with removed special keys did not match expected"


def test_model_option_replace_to_common_opts(caplog):
    model_opts = {
        ModelOption.CONTEXT_WINDOW: 3,
        ModelOption.TEMPERATURE: 1,
        "random_opt": "test",
        "context_size": 4
    }

    # Equivalent to what the replacement should be at the beginning of
    # the backend.generate call.
    replacements_to_mellea = {
        "context_size": ModelOption.CONTEXT_WINDOW,
        "random_opt": "randomize",
    }

    expected_opts_to = {ModelOption.CONTEXT_WINDOW: 3, ModelOption.TEMPERATURE: 1, "randomize": "test"}

    processed_model_opts = ModelOption.replace_keys(model_opts, replacements_to_mellea)
    assert (
        expected_opts_to == processed_model_opts
    ), "dict with replaced keys did not match expected"

    # There should also be a logged message due to context_window key clashes.
    assert "old_key (context_size) to new_key (@@@context_window@@@): lost value associated with old_key (4) and kept original value of new_key (3)" in caplog.text, f"expected log for conflicting keys not found in: {caplog.text}"


def test_model_option_replace_to_backend_specific():
    # There shouldn't be any conflicting options at this point. We leave one in here
    # for testing.
    model_opts = {
        ModelOption.CONTEXT_WINDOW: 3,
        ModelOption.TEMPERATURE: 1,
        "random_opt": "test",
        "context_size": 4
    }

    # Equivalent to what the replacement should be to get the model_options
    # that actually get passed to the model's generate call.
    replacements_from_mellea = {
        ModelOption.CONTEXT_WINDOW: "context_size",
    }

    expected_opts_from = {"context_size": 4, ModelOption.TEMPERATURE: 1, "random_opt": "test"}

    processed_model_opts = ModelOption.replace_keys(model_opts, replacements_from_mellea)
    assert (
        expected_opts_from == processed_model_opts
    ), "dict with replaced keys did not match expected"


def test_model_option_replace_with_conflicts():
    # Equivalent to instances where multiple backend specific values map to the same
    # mellea ModelOption.Option.
    model_opts = {
        "tools": "tools1",
        "functions": "tools2"
    }

    replacements_to_mellea = {
        "tools": ModelOption.TOOLS,
        "functions": ModelOption.TOOLS,
    }

    processed_model_opts = ModelOption.replace_keys(model_opts, replacements_to_mellea)
    assert ModelOption.TOOLS in processed_model_opts


def test_model_option_merge():
    default_model_opts = {
        ModelOption.CONTEXT_WINDOW: 3,
        ModelOption.TEMPERATURE: 1,
        "random_opt": "test",
    }

    overwrite_opts = {ModelOption.CONTEXT_WINDOW: 4, "new_opt": "new_value"}

    expected_opts = {
        ModelOption.CONTEXT_WINDOW: 4,
        "new_opt": "new_value",
        ModelOption.TEMPERATURE: 1,
        "random_opt": "test",
    }

    processed_model_opts = ModelOption.merge_model_options(
        default_model_opts, overwrite_opts
    )
    assert expected_opts == processed_model_opts, "merged dict did not match expected"


if __name__ == "__main__":
    pytest.main([__file__])
