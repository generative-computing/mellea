# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from mellea.backends import ModelOption
from mellea.backends.huggingface import (
    _GENERATE_KWARGS_ALLOWLIST,
    _HF_INTERNAL_TEMPLATE_VARS,
    LocalHFBackend,
)


def _make_backend(template: object) -> LocalHFBackend:
    """Return a LocalHFBackend with __init__ bypassed, wired with a known template.

    Only the attributes accessed by _chat_template_allowlist, _filter_for_chat_template,
    and _make_backend_specific_and_remove are set.
    If any of those methods gains a new self.* dependency, update this helper.
    """

    class _FakeTokenizer:
        chat_template = template

    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    object.__setattr__(b, "_tokenizer", _FakeTokenizer())
    b._model_id = "test-org/test-model"
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}
    return b


# ---------------------------------------------------------------------------
# _HF_INTERNAL_TEMPLATE_VARS — contents are load-bearing; review on each
# transformers upgrade.  This test makes the expected set explicit so that any
# unintentional change to the constant is caught immediately.
# ---------------------------------------------------------------------------

_EXPECTED_INTERNAL_VARS = {
    # Variables bound in the Jinja namespace by render_jinja_template (transformers v5+)
    "messages",
    "tools",
    "documents",
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
# _GENERATE_KWARGS_ALLOWLIST + _make_backend_specific_and_remove(for_generate=True):
# kwargs that generate() does not accept (chat-template variables) must be
# dropped before they reach generate.
# ---------------------------------------------------------------------------


@pytest.fixture
def plain_backend() -> LocalHFBackend:
    """Backend fixture without a tokenizer — sufficient for the generate-kwargs filter."""
    b: LocalHFBackend = LocalHFBackend.__new__(LocalHFBackend)
    b.from_mellea_model_opts_map = {ModelOption.MAX_NEW_TOKENS: "max_new_tokens"}
    return b


def test_generate_filter_drops_guardian_config(plain_backend: LocalHFBackend) -> None:
    opts = {"guardian_config": {"foo": "bar"}, ModelOption.TEMPERATURE: 0.5}
    result = plain_backend._make_backend_specific_and_remove(opts)
    assert "guardian_config" not in result
    assert "temperature" in result


def test_generate_filter_drops_thinking(plain_backend: LocalHFBackend) -> None:
    opts = {"thinking": True, ModelOption.MAX_NEW_TOKENS: 64}
    result = plain_backend._make_backend_specific_and_remove(opts)
    assert "thinking" not in result
    assert result["max_new_tokens"] == 64


def test_generate_filter_drops_add_generation_prompt(
    plain_backend: LocalHFBackend,
) -> None:
    opts = {"add_generation_prompt": True, "temperature": 1.0}
    result = plain_backend._make_backend_specific_and_remove(opts)
    assert "add_generation_prompt" not in result
    assert "temperature" in result


def test_generate_filter_drops_documents(plain_backend: LocalHFBackend) -> None:
    opts = {"documents": [{"text": "hello"}], "do_sample": False}
    result = plain_backend._make_backend_specific_and_remove(opts)
    assert "documents" not in result
    assert "do_sample" in result


def test_generate_filter_drops_template_only_reasoning_kwargs(
    plain_backend: LocalHFBackend,
) -> None:
    """Chat-template variables specific to non-Granite models must also be dropped.

    These variables are referenced by the chat templates of models shipped in
    ``mellea/backends/model_ids.py`` (Qwen3, Llama 3.x/4, gpt-oss, etc.) but are
    NOT accepted by ``transformers``' ``generate``. If forwarded through, they
    would raise ``TypeError`` at inference time.
    """
    template_only_kwargs = [
        "enable_thinking",  # Qwen3
        "thinking",  # Granite
        "controls",  # Granite
        "available_tools",  # Granite
        "custom_tools",  # Llama 3.x / 4
        "tools_in_user_message",  # Llama 3.x / 4
        "builtin_tools",  # Llama 3.3
        "reasoning_effort",  # gpt-oss
        "model_identity",  # gpt-oss
    ]
    opts: dict[str, Any] = dict.fromkeys(template_only_kwargs, "x")
    opts["temperature"] = 0.7  # control: a real generate() kwarg
    result = plain_backend._make_backend_specific_and_remove(opts)
    for kw in template_only_kwargs:
        assert kw not in result, f"{kw!r} leaked through to generate() kwargs"
    assert result["temperature"] == 0.7


def test_for_generate_false_skips_filter(plain_backend: LocalHFBackend) -> None:
    """``for_generate=False`` skips the generate-kwargs filter.

    This is the path used by ``_filter_for_chat_template`` — chat-template
    variables must survive the rename step so the chat-template allowlist can
    examine them.
    """
    opts = {"thinking": True, "guardian_config": {"x": 1}, ModelOption.TEMPERATURE: 0.5}
    result = plain_backend._make_backend_specific_and_remove(opts, for_generate=False)
    assert result["thinking"] is True
    assert result["guardian_config"] == {"x": 1}
    assert result["temperature"] == 0.5


def test_generate_kwargs_allowlist_no_chat_template_only_overlap() -> None:
    """No chat-template-only variable shipped by Mellea's models is in the allowlist.

    This is the regression guard for the bug class fixed in this change: if a
    Jinja variable referenced by one of the templates in
    ``mellea/backends/model_ids.py`` is *also* a real ``generate()`` kwarg name,
    the allowlist would let it leak through and we'd need a different mechanism
    (an explicit denylist or a per-call-site decision). Today there is no
    overlap, and this test will fail loudly the day that changes.
    """
    chat_template_only = {
        # Granite (3.x and 4.0-tiny-preview)
        "guardian_config",
        "thinking",
        "controls",
        "available_tools",
        # Llama 3.x / 4
        "custom_tools",
        "tools_in_user_message",
        "builtin_tools",
        # Qwen3
        "enable_thinking",
        # gpt-oss
        "model_identity",
        "reasoning_effort",
        # Universal
        "add_generation_prompt",
        "documents",
    }
    overlap = chat_template_only & _GENERATE_KWARGS_ALLOWLIST
    assert not overlap, (
        f"chat-template-only variables {sorted(overlap)} are also accepted by "
        "transformers' generate(); the generate-kwargs allowlist filter alone is "
        "insufficient. Add a denylist for the overlapping names."
    )


def test_generate_kwargs_allowlist_includes_known_generate_kwargs() -> None:
    """Common ``generate()`` kwargs Mellea relies on must be in the allowlist.

    A canary against an upstream rename or removal of a kwarg that Mellea
    forwards from caller-supplied ``model_options``. Mellea-injected kwargs
    (``stopping_criteria``, ``streamer``, etc.) are passed explicitly at the
    call sites and don't need to be in this set, but generation-tuning kwargs
    that callers customize do.
    """
    expected = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "do_sample",
        "num_beams",
        "repetition_penalty",
        "return_dict_in_generate",
        "output_scores",
        "use_cache",
        "stop_strings",
        "stopping_criteria",
        "streamer",
    }
    missing = expected - _GENERATE_KWARGS_ALLOWLIST
    assert not missing, (
        f"expected generate() kwargs {sorted(missing)} not found in allowlist; "
        "did transformers rename or remove them?"
    )


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
    """Granite's chat template exposes 'thinking' as a Jinja var.

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

    # This is a Granite-specific Jinja variable that must survive the filter.
    # If it is absent the filter is over-aggressive and will break Granite
    # think-mode. It serves as a signal for params that shouldn't be dropped.
    assert "thinking" in allowlist, (
        f"'thinking' missing from allowlist; got: {sorted(allowlist)}"
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
