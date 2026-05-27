"""Unit tests for LocalHFBackend chat-template kwarg filtering.

These tests verify that _filter_for_chat_template passes only the variables the
tokenizer's Jinja template actually references, and that _HF_INTERNAL_TEMPLATE_VARS
are correctly excluded from the allowlist regardless of whether the template uses them.

No GPU or real model is needed — the methods under test are pure dict operations that
happen to read self._tokenizer.chat_template.  The fixture sets that attribute to a
known Jinja string, bypassing __init__ (and therefore model loading) entirely.

torch must be importable because importing LocalHFBackend triggers the top-level
``import torch`` in huggingface.py.  Install mellea[hf] to satisfy this requirement.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from mellea.backends import ModelOption
from mellea.backends.huggingface import _HF_INTERNAL_TEMPLATE_VARS, LocalHFBackend


def _make_backend(template: str) -> LocalHFBackend:
    """Return a LocalHFBackend with __init__ bypassed, wired with a known template.

    Only the attributes accessed by _chat_template_allowlist, _filter_for_chat_template,
    _filter_chat_template_only_options, and _make_backend_specific_and_remove are set.
    If any of those methods gains a new self.* dependency, update this helper.
    """

    class _FakeTokenizer:
        chat_template = template

    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b._tokenizer = _FakeTokenizer()
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}
    return b


# ---------------------------------------------------------------------------
# _HF_INTERNAL_TEMPLATE_VARS — contents are load-bearing; review on each
# transformers upgrade.  This test makes the expected set explicit so that any
# unintentional change to the constant is caught immediately.
# ---------------------------------------------------------------------------

_EXPECTED_INTERNAL_VARS = {
    # Named parameters wired by apply_chat_template / render_jinja_template
    "messages",
    "tools",
    "add_generation_prompt",
    # Jinja environment globals from _compile_jinja_template
    "raise_exception",
    "strftime_now",
}


def test_hf_internal_template_vars_contents() -> None:
    """_HF_INTERNAL_TEMPLATE_VARS must contain exactly the expected set.

    This test is intentionally strict: it catches both missing entries (a new HF
    internal variable not yet excluded) and accidental additions (a variable
    incorrectly labelled as HF-internal that users might legitimately pass).
    If transformers changes what it injects, update _HF_INTERNAL_TEMPLATE_VARS
    AND update _EXPECTED_INTERNAL_VARS here to match.
    """
    assert _HF_INTERNAL_TEMPLATE_VARS == _EXPECTED_INTERNAL_VARS


# ---------------------------------------------------------------------------
# _chat_template_allowlist — verify the Jinja AST parsing logic
# ---------------------------------------------------------------------------


def test_allowlist_includes_template_variables() -> None:
    """Variables declared in the template appear in the allowlist."""
    template = "{% for m in messages %}{{ m.role }}: {{ think }}{% endfor %}"
    b = _make_backend(template)
    assert "think" in b._chat_template_allowlist


def test_allowlist_excludes_hf_internal_vars() -> None:
    """HF-internal variables are excluded even when the template references them."""
    # A template that uses every internal variable
    template = (
        "{% if raise_exception %}{{ messages }}{% endif %}"
        "{{ strftime_now('%Y') }}{{ tools }}{{ add_generation_prompt }}"
    )
    b = _make_backend(template)
    for var in _HF_INTERNAL_TEMPLATE_VARS:
        assert var not in b._chat_template_allowlist, (
            f"HF-internal var '{var}' leaked into allowlist"
        )


def test_allowlist_excludes_generate_only_options() -> None:
    """Generate-only option names are not in the allowlist of a typical chat template.

    Real HF templates do not reference GenerationConfig parameter names as Jinja
    variables.  This test uses a realistic (but minimal) template to confirm that
    common generation options are absent from the computed allowlist.
    """
    template = (
        "{% for message in messages %}"
        "{{ message.role }}: {{ message.content }}"
        "{% endfor %}"
        "{% if think %}Think step by step.{% endif %}"
    )
    b = _make_backend(template)
    generate_only_examples = [
        "temperature",
        "max_new_tokens",
        "do_sample",
        "top_k",
        "top_p",
        "num_beams",
    ]
    for key in generate_only_examples:
        assert key not in b._chat_template_allowlist, (
            f"generate-only key '{key}' unexpectedly in allowlist"
        )


def test_allowlist_empty_for_missing_chat_template() -> None:
    """No crash and empty allowlist when the tokenizer has no chat_template."""
    b = _make_backend(None)  # type: ignore[arg-type]
    assert b._chat_template_allowlist == frozenset()


def test_allowlist_is_cached() -> None:
    """_chat_template_allowlist is computed once and reused (cached_property)."""
    b = _make_backend("{{ think }}")
    first = b._chat_template_allowlist
    second = b._chat_template_allowlist
    assert first is second


# ---------------------------------------------------------------------------
# _filter_for_chat_template — end-to-end: sentinel renaming + allowlist filter
# ---------------------------------------------------------------------------


def test_filter_for_chat_template_passes_template_vars() -> None:
    """Variables the template references survive the filter."""
    template = "{{ think }}{{ guardian_config }}"
    b = _make_backend(template)
    result = b._filter_for_chat_template(
        {"think": True, "guardian_config": {"key": "val"}}
    )
    assert result["think"] is True
    assert result["guardian_config"] == {"key": "val"}


def test_filter_for_chat_template_drops_generate_only() -> None:
    """Generate-only options not referenced by the template are filtered out."""
    template = "{% for m in messages %}{{ m.content }}{% endfor %}{{ think }}"
    b = _make_backend(template)
    result = b._filter_for_chat_template(
        {
            ModelOption.TEMPERATURE: 0.8,
            ModelOption.MAX_NEW_TOKENS: 200,
            "do_sample": True,
            "top_k": 50,
            "num_beams": 4,
            "think": True,
        }
    )
    # Template only uses 'think' (messages is HF-internal, excluded from allowlist)
    assert result == {"think": True}


def test_filter_for_chat_template_renames_sentinel() -> None:
    """Mellea sentinels are renamed before the allowlist check.

    If a (hypothetical) template used 'max_new_tokens' as a Jinja variable,
    _filter_for_chat_template would rename the sentinel and then keep the key.
    """
    template = "{{ max_new_tokens }}"  # unusual but valid for testing rename path
    b = _make_backend(template)
    result = b._filter_for_chat_template({ModelOption.MAX_NEW_TOKENS: 256})
    assert result == {"max_new_tokens": 256}


def test_filter_for_chat_template_empty_input() -> None:
    """Empty model_options produces an empty dict."""
    b = _make_backend("{{ think }}")
    assert b._filter_for_chat_template({}) == {}


def test_filter_for_chat_template_unknown_key_dropped() -> None:
    """A key that is not in the template is silently dropped."""
    template = "{{ think }}"
    b = _make_backend(template)
    result = b._filter_for_chat_template({"think": True, "not_in_template": 99})
    assert result == {"think": True}
    assert "not_in_template" not in result


# ---------------------------------------------------------------------------
# _filter_chat_template_only_options (pre-existing method — regression guard)
# ---------------------------------------------------------------------------


@pytest.fixture
def plain_backend() -> LocalHFBackend:
    """Backend fixture without a tokenizer — sufficient for the pre-existing filter."""
    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}
    return b


def test_filter_chat_template_only_removes_guardian_config(
    plain_backend: LocalHFBackend,
) -> None:
    opts = {"guardian_config": {"foo": "bar"}, ModelOption.TEMPERATURE: 0.5}
    result = plain_backend._filter_chat_template_only_options(opts)
    assert "guardian_config" not in result
    assert ModelOption.TEMPERATURE in result


def test_filter_chat_template_only_removes_think(plain_backend: LocalHFBackend) -> None:
    opts = {"think": True, "max_new_tokens": 64}
    result = plain_backend._filter_chat_template_only_options(opts)
    assert "think" not in result
    assert "max_new_tokens" in result


def test_filter_chat_template_only_removes_add_generation_prompt(
    plain_backend: LocalHFBackend,
) -> None:
    opts = {"add_generation_prompt": True, "temperature": 1.0}
    result = plain_backend._filter_chat_template_only_options(opts)
    assert "add_generation_prompt" not in result
    assert "temperature" in result


def test_filter_chat_template_only_removes_documents(
    plain_backend: LocalHFBackend,
) -> None:
    opts = {"documents": [{"text": "hello"}], "do_sample": False}
    result = plain_backend._filter_chat_template_only_options(opts)
    assert "documents" not in result
    assert "do_sample" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
