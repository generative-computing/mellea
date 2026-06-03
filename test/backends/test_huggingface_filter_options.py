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

import logging

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from mellea.backends import ModelOption
from mellea.backends.huggingface import _HF_INTERNAL_TEMPLATE_VARS, LocalHFBackend


def _make_backend(template: object) -> LocalHFBackend:
    """Return a LocalHFBackend with __init__ bypassed, wired with a known template.

    Only the attributes accessed by _chat_template_allowlist, _filter_for_chat_template,
    _filter_chat_template_only_options, and _make_backend_specific_and_remove are set.
    If any of those methods gains a new self.* dependency, update this helper.
    """

    class _FakeTokenizer:
        chat_template = template

    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    object.__setattr__(b, "_tokenizer", _FakeTokenizer())
    b._hf_model_id = "test-org/test-model"
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
    b = _make_backend(None)
    assert b._chat_template_allowlist == frozenset()


def test_allowlist_is_cached() -> None:
    """_chat_template_allowlist is computed once and reused (cached_property)."""
    b = _make_backend("{{ think }}")
    first = b._chat_template_allowlist
    second = b._chat_template_allowlist
    assert first is second


def test_allowlist_non_string_template_returns_empty() -> None:
    """Non-string chat_template (e.g. list of alternates) returns empty frozenset.

    Some tokenizers store chat_template as a list-of-dicts (multiple named templates).
    apply_chat_template handles that format internally; we must not crash trying to
    parse it as a Jinja string.
    """
    b = _make_backend([{"name": "default", "template": "{{ think }}"}])
    assert b._chat_template_allowlist == frozenset()


def test_allowlist_graceful_on_generation_tag() -> None:
    """Templates using {% generation %} / {% endgeneration %} return empty frozenset.

    The {% generation %} tag comes from transformers' AssistantTracker extension and
    is not registered in a plain jinja2.Environment. Parsing must not raise; instead
    _chat_template_allowlist returns frozenset() so _filter_for_chat_template forwards
    nothing (safe default) rather than crashing during inference.

    Used by DeepSeek-R1, Qwen3, and other models as of transformers ≥ 4.47.
    """
    template = (
        "{% for message in messages %}"
        "{% generation %}{{ message.content }}{% endgeneration %}"
        "{% endfor %}"
        "{{ think }}"
    )
    b = _make_backend(template)
    assert b._chat_template_allowlist == frozenset()


def test_allowlist_warns_on_generation_tag(caplog: pytest.LogCaptureFixture) -> None:
    """A warning is emitted when the template cannot be parsed due to unknown tags.

    Callers passing model_options (e.g. think=True) to a DeepSeek-R1/Qwen3 model
    would otherwise see their options silently dropped with no diagnostic.
    """
    template = (
        "{% for message in messages %}"
        "{% generation %}{{ message.content }}{% endgeneration %}"
        "{% endfor %}"
        "{{ think }}"
    )
    b = _make_backend(template)
    with caplog.at_level(logging.WARNING, logger="mellea"):
        _ = b._chat_template_allowlist
    assert any("Could not parse chat template" in r.message for r in caplog.records)
    assert any("test-org/test-model" in r.message for r in caplog.records)


def test_allowlist_warns_on_non_string_template(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warning is emitted when chat_template is a list or dict (not a plain string).

    Some tokenizers store multiple named templates as a list-of-dicts.  The warning
    tells the caller that their model_options will not be forwarded.
    """
    b = _make_backend([{"name": "default", "template": "{{ think }}"}])
    with caplog.at_level(logging.WARNING, logger="mellea"):
        _ = b._chat_template_allowlist
    assert any("not a plain string" in r.message for r in caplog.records)


def test_allowlist_no_warning_for_none_template(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No warning is emitted when chat_template is None (model has no chat template).

    A missing chat_template is normal and expected for base models — no diagnostic needed.
    """
    b = _make_backend(None)
    with caplog.at_level(logging.WARNING, logger="mellea"):
        _ = b._chat_template_allowlist
    assert not any("chat template" in r.message.lower() for r in caplog.records)


def test_allowlist_break_continue_tags_parsed_correctly() -> None:
    """Templates using {% break %} / {% continue %} are parsed without error.

    jinja2.ext.loopcontrols must be registered so that {% break %} inside a for
    loop does not raise TemplateSyntaxError. Variables outside the loop body must
    still appear in the allowlist.
    """
    template = (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}{% break %}{% endif %}"
        "{{ message.content }}"
        "{% endfor %}"
        "{% if think %}Think step by step.{% endif %}"
    )
    b = _make_backend(template)
    assert "think" in b._chat_template_allowlist


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


# ---------------------------------------------------------------------------
# Integration: real Granite tokenizer — verifies the allowlist against the
# actual production template rather than a hand-crafted synthetic one.
#
# Marked huggingface because it reads from the HuggingFace model cache.
# Skips automatically if the tokenizer is not cached locally.
# No GPU required — only the tokenizer files are loaded.
# ---------------------------------------------------------------------------

_GRANITE_MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"


def _try_load_granite_tokenizer():
    """Return the Granite tokenizer if cached locally, else None."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(_GRANITE_MODEL_ID, local_files_only=True)
    except Exception:
        return None


@pytest.mark.huggingface
def test_granite_allowlist_includes_known_template_vars() -> None:
    """Granite's chat template exposes 'think' and 'guardian_config' as Jinja vars.

    This test loads the real tokenizer (tokenizer files only — no GPU, no model
    weights) and asserts the computed allowlist includes the template-specific
    options that Mellea passes for Granite models.

    If the allowlist is empty or missing these keys after a transformers upgrade,
    it means either the Granite template changed or _HF_INTERNAL_TEMPLATE_VARS
    is now incorrectly excluding something it should not.
    """
    tok = _try_load_granite_tokenizer()
    if tok is None:
        pytest.skip(
            f"{_GRANITE_MODEL_ID} not in local HF cache — run qualitative tests first"
        )

    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b._tokenizer = tok
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}

    allowlist = b._chat_template_allowlist

    # These are Granite-specific Jinja variables that must survive the filter.
    # If either is absent the filter is over-aggressive and will break Granite
    # think-mode and guardian calls.
    assert "think" in allowlist, (
        f"'think' missing from allowlist; got: {sorted(allowlist)}"
    )
    assert "guardian_config" in allowlist, (
        f"'guardian_config' missing from allowlist; got: {sorted(allowlist)}"
    )


@pytest.mark.huggingface
def test_granite_allowlist_excludes_generate_only_options() -> None:
    """The Granite template does not reference GenerationConfig param names as Jinja vars.

    This is the integration-level proof that the allowlist approach correctly
    drops generate-only options for the real production model — and that the
    approach remains valid after a transformers upgrade.

    Failure here means the Granite template now references a GenerationConfig
    parameter name as a Jinja variable, which would require revisiting the design.
    """
    tok = _try_load_granite_tokenizer()
    if tok is None:
        pytest.skip(
            f"{_GRANITE_MODEL_ID} not in local HF cache — run qualitative tests first"
        )

    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b._tokenizer = tok
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}

    allowlist = b._chat_template_allowlist

    generate_only = [
        "temperature",
        "max_new_tokens",
        "do_sample",
        "top_k",
        "top_p",
        "num_beams",
        "repetition_penalty",
        "min_new_tokens",
        "pad_token_id",
    ]
    for key in generate_only:
        assert key not in allowlist, (
            f"generate-only key '{key}' appeared in Granite allowlist — "
            f"the template may now reference it as a Jinja variable"
        )


@pytest.mark.huggingface
def test_granite_allowlist_excludes_hf_internal_vars() -> None:
    """HF-internal vars are excluded from the Granite allowlist.

    Verifies that _HF_INTERNAL_TEMPLATE_VARS correctly covers all variables
    HuggingFace injects for the real production template. If any internal var
    leaks into the allowlist, forwarding it from model_options would duplicate
    a kwarg that apply_chat_template already provides, causing a TypeError.
    """
    tok = _try_load_granite_tokenizer()
    if tok is None:
        pytest.skip(
            f"{_GRANITE_MODEL_ID} not in local HF cache — run qualitative tests first"
        )

    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b._tokenizer = tok
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}

    allowlist = b._chat_template_allowlist

    for var in _HF_INTERNAL_TEMPLATE_VARS:
        assert var not in allowlist, (
            f"HF-internal var '{var}' leaked into Granite allowlist — "
            f"check _HF_INTERNAL_TEMPLATE_VARS against the installed transformers version"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
