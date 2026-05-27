"""Unit tests for LocalHFBackend option-filter helpers.

These tests verify that generate-only options are excluded from apply_chat_template
kwargs, and that template-only options are excluded from generate() kwargs.

No GPU or real model is needed to run these tests — the filter methods are pure dict
operations and the fixture bypasses __init__ entirely. However, torch must be
importable because importing LocalHFBackend triggers the top-level ``import torch``
in huggingface.py. Install mellea[hf] to satisfy this requirement.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend


@pytest.fixture
def backend() -> LocalHFBackend:
    """A LocalHFBackend instance with no model loaded, sufficient for testing filter helpers.

    Uses __new__ to bypass __init__ (which would load model weights). Only
    from_mellea_model_opts_map is set because that is the sole instance attribute
    accessed by _filter_generate_only_options, _filter_chat_template_only_options,
    and _make_backend_specific_and_remove. If any of those methods gains a new
    self.* dependency, update this fixture.
    """
    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}
    return b


# ---------------------------------------------------------------------------
# _filter_generate_only_options
# ---------------------------------------------------------------------------


def test_filter_generate_only_removes_temperature(backend: LocalHFBackend) -> None:
    opts = {ModelOption.TEMPERATURE: 0.7, "think": True}
    result = backend._filter_generate_only_options(opts)
    assert ModelOption.TEMPERATURE not in result
    assert "think" in result


def test_filter_generate_only_removes_max_new_tokens(backend: LocalHFBackend) -> None:
    opts = {ModelOption.MAX_NEW_TOKENS: 256, "guardian_config": {"foo": "bar"}}
    result = backend._filter_generate_only_options(opts)
    assert ModelOption.MAX_NEW_TOKENS not in result
    assert "guardian_config" in result


def test_filter_generate_only_removes_do_sample(backend: LocalHFBackend) -> None:
    opts = {"do_sample": True, "add_generation_prompt": True}
    result = backend._filter_generate_only_options(opts)
    assert "do_sample" not in result
    assert "add_generation_prompt" in result


def test_filter_generate_only_removes_all_three(backend: LocalHFBackend) -> None:
    opts = {
        ModelOption.TEMPERATURE: 0.9,
        ModelOption.MAX_NEW_TOKENS: 128,
        "do_sample": True,
        "think": True,
        "guardian_config": {"key": "val"},
    }
    result = backend._filter_generate_only_options(opts)
    assert ModelOption.TEMPERATURE not in result
    assert ModelOption.MAX_NEW_TOKENS not in result
    assert "do_sample" not in result
    # Template-only keys must survive
    assert result["think"] is True
    assert result["guardian_config"] == {"key": "val"}


def test_filter_generate_only_empty_input(backend: LocalHFBackend) -> None:
    assert backend._filter_generate_only_options({}) == {}


def test_filter_generate_only_passthrough_keys_preserved(
    backend: LocalHFBackend,
) -> None:
    opts = {"some_custom_template_var": 42, "documents": [{"text": "hi"}]}
    result = backend._filter_generate_only_options(opts)
    assert result == opts


# ---------------------------------------------------------------------------
# _filter_chat_template_only_options (existing method — regression guard)
# ---------------------------------------------------------------------------


def test_filter_chat_template_only_removes_guardian_config(
    backend: LocalHFBackend,
) -> None:
    opts = {"guardian_config": {"foo": "bar"}, ModelOption.TEMPERATURE: 0.5}
    result = backend._filter_chat_template_only_options(opts)
    assert "guardian_config" not in result
    assert ModelOption.TEMPERATURE in result


def test_filter_chat_template_only_removes_think(backend: LocalHFBackend) -> None:
    opts = {"think": True, "max_new_tokens": 64}
    result = backend._filter_chat_template_only_options(opts)
    assert "think" not in result
    assert "max_new_tokens" in result


def test_filter_chat_template_only_removes_add_generation_prompt(
    backend: LocalHFBackend,
) -> None:
    opts = {"add_generation_prompt": True, "temperature": 1.0}
    result = backend._filter_chat_template_only_options(opts)
    assert "add_generation_prompt" not in result
    assert "temperature" in result


def test_filter_chat_template_only_removes_documents(backend: LocalHFBackend) -> None:
    opts = {"documents": [{"text": "hello"}], "do_sample": False}
    result = backend._filter_chat_template_only_options(opts)
    assert "documents" not in result
    assert "do_sample" in result


# ---------------------------------------------------------------------------
# Integration: filter_generate_only → _make_backend_specific_and_remove
# Mirrors the exact chain used at both apply_chat_template call sites.
# ---------------------------------------------------------------------------


def test_apply_chat_template_kwargs_exclude_generate_only(
    backend: LocalHFBackend,
) -> None:
    """Full chain: after filtering and renaming, no generate-only key reaches apply_chat_template."""
    model_options = {
        ModelOption.TEMPERATURE: 0.8,
        ModelOption.MAX_NEW_TOKENS: 200,
        "do_sample": True,
        "think": True,
        "guardian_config": {"harm_categories": []},
        "add_generation_prompt": True,
    }

    kwargs = backend._make_backend_specific_and_remove(
        backend._filter_generate_only_options(model_options)
    )

    # Generate-only options must not appear
    assert "temperature" not in kwargs, (
        "temperature leaked into apply_chat_template kwargs"
    )
    assert "max_new_tokens" not in kwargs, (
        "max_new_tokens leaked into apply_chat_template kwargs"
    )
    assert "do_sample" not in kwargs, "do_sample leaked into apply_chat_template kwargs"

    # Template-only options must still be present
    assert kwargs.get("think") is True
    assert kwargs.get("guardian_config") == {"harm_categories": []}
    assert kwargs.get("add_generation_prompt") is True


def test_generate_kwargs_exclude_chat_template_only(backend: LocalHFBackend) -> None:
    """Inverse chain: after filtering template-only keys, none reach generate()."""
    model_options = {
        ModelOption.TEMPERATURE: 0.8,
        ModelOption.MAX_NEW_TOKENS: 200,
        "do_sample": True,
        "think": True,
        "guardian_config": {"harm_categories": []},
        "add_generation_prompt": True,
    }

    generate_options = backend._filter_chat_template_only_options(model_options)
    kwargs = backend._make_backend_specific_and_remove(generate_options)

    # Template-only options must not appear
    assert "think" not in kwargs
    assert "guardian_config" not in kwargs
    assert "add_generation_prompt" not in kwargs

    # Generate-only options must still be present
    assert kwargs.get("temperature") == 0.8
    assert kwargs.get("max_new_tokens") == 200
    assert kwargs.get("do_sample") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
