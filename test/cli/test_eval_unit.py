# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for eval runner pure-logic helpers — no backend, no model required.

Covers InputEvalResult, TestEvalResult, parse_judge_output, _extract_first_json,
and create_session model resolution (backends are stubbed).
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from cli.eval.runner import (
    InputEvalResult,
    TestEvalResult,
    _extract_first_json,
    _resolved_model_name,
    create_session,
    parse_judge_output,
)
from mellea.backends import ModelOption, model_ids
from mellea.helpers.server_type import _ServerType
from mellea.stdlib.components.unit_test_eval import TestBasedEval

# --- InputEvalResult ---


def test_input_eval_result_to_dict():
    r = InputEvalResult(
        input_text="What is 2+2?",
        model_output="4",
        validation_passed=True,
        score=1,
        validation_reason="Correct answer",
    )
    d = r.to_dict()
    assert d["input"] == "What is 2+2?"
    assert d["model_output"] == "4"
    assert d["passed"] is True
    assert d["score"] == 1
    assert d["justification"] == "Correct answer"


def test_input_eval_result_to_dict_failed():
    r = InputEvalResult("q", "wrong", False, 0, "Incorrect")
    d = r.to_dict()
    assert d["passed"] is False
    assert d["score"] == 0


# --- TestEvalResult ---


def _make_test_eval() -> TestBasedEval:
    return TestBasedEval(
        source="test_source",
        name="test_name",
        instructions="Judge if correct",
        inputs=["input1", "input2"],
        test_id="test-001",
    )


def _make_input_results(passed: list[bool]) -> list[InputEvalResult]:
    return [
        InputEvalResult(f"q{i}", f"a{i}", p, 1 if p else 0, "reason")
        for i, p in enumerate(passed)
    ]


def test_test_eval_result_passed_count():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, False])
    r = TestEvalResult(eval_spec, results)
    assert r.passed_count == 1


def test_test_eval_result_pass_rate():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, False])
    r = TestEvalResult(eval_spec, results)
    assert r.pass_rate == pytest.approx(0.5)


def test_test_eval_result_pass_rate_empty():
    eval_spec = _make_test_eval()
    r = TestEvalResult(eval_spec, [])
    assert r.pass_rate == 0.0


def test_test_eval_result_all_pass():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, True])
    r = TestEvalResult(eval_spec, results)
    assert r.pass_rate == pytest.approx(1.0)


def test_test_eval_result_to_dict_structure():
    eval_spec = _make_test_eval()
    results = _make_input_results([True, False])
    r = TestEvalResult(eval_spec, results)
    d = r.to_dict()
    assert d["test_id"] == "test-001"
    assert d["source"] == "test_source"
    assert d["name"] == "test_name"
    assert d["instructions"] == "Judge if correct"
    assert len(d["input_results"]) == 2
    assert d["passed"] == 1
    assert d["total_count"] == 2
    assert d["pass_rate"] == pytest.approx(0.5)


# --- parse_judge_output ---


def test_parse_json_score_and_justification():
    output = '{"score": 1, "justification": "Correct answer"}'
    score, reason = parse_judge_output(output)
    assert score == 1
    assert reason == "Correct answer"


def test_parse_json_embedded_in_text():
    output = 'Based on my review: {"score": 0, "justification": "Wrong answer"} end.'
    score, reason = parse_judge_output(output)
    assert score == 0
    assert reason == "Wrong answer"


def test_parse_score_from_plain_text():
    output = "Score: 1\nThe answer is correct."
    score, reason = parse_judge_output(output)
    assert score == 1
    assert reason == output


def test_parse_no_score_returns_none():
    output = "I cannot determine the score."
    score, reason = parse_judge_output(output)
    assert score is None
    assert reason == output


def test_parse_invalid_json_falls_back_to_regex():
    output = 'Almost JSON: {"score": 1, but broken}'
    score, reason = parse_judge_output(output)
    # Regex fallback should find "score": 1 and return the full raw text as justification
    assert score == 1
    assert reason == output


def test_parse_zero_score():
    output = '{"score": 0, "justification": "Failed"}'
    score, reason = parse_judge_output(output)
    assert score == 0
    assert reason == "Failed"


def test_parse_nested_json_preserves_justification():
    output = '{"score": 1, "justification": "Correct", "reasoning": {"detail": "step-by-step"}}'
    score, reason = parse_judge_output(output)
    assert score == 1
    assert reason == "Correct"


def test_parse_json_score_no_justification_key():
    output = '{"score": 1}'
    score, reason = parse_judge_output(output)
    assert score == 1
    assert reason == output


def test_parse_json_justification_null():
    output = '{"score": 0, "justification": null}'
    score, reason = parse_judge_output(output)
    assert score == 0
    assert reason == output


def test_parse_second_json_when_first_lacks_score():
    output = '{"context": "intro"} {"score": 1, "justification": "Looks good"}'
    score, reason = parse_judge_output(output)
    assert score == 1
    assert reason == "Looks good"


# --- _extract_first_json ---


def test_extract_first_json_finds_score_object():
    assert _extract_first_json('{"score": 1, "justification": "ok"}') == {
        "score": 1,
        "justification": "ok",
    }


def test_extract_first_json_skips_object_without_score():
    text = '{"foo": "bar"} {"score": 0}'
    assert _extract_first_json(text) == {"score": 0}


def test_extract_first_json_no_json_returns_none():
    assert _extract_first_json("plain text, no JSON here") is None


def test_extract_first_json_no_score_key_returns_none():
    assert _extract_first_json('{"justification": "no score anywhere"}') is None


# --- create_session model resolution ---

_BACKEND_CLASSES = {
    "ollama": ("mellea.backends.ollama", "OllamaModelBackend"),
    "openai": ("mellea.backends.openai", "OpenAIBackend"),
    "hf": ("mellea.backends.huggingface", "LocalHFBackend"),
    "watsonx": ("mellea.backends.watsonx", "WatsonxAIBackend"),
    "litellm": ("mellea.backends.litellm", "LiteLLMBackend"),
}


@pytest.fixture
def stub_backend(monkeypatch):
    """Replace the backend module for `name` with a stub and return the class mock.

    Stubbing the module (rather than the class) keeps these tests independent of
    optional extras such as torch or the watsonx SDK. Keyword arguments to `_stub`
    set attributes on the stubbed instance, e.g. `_server_type`.
    """

    def _stub(name: str, **instance_attributes: object) -> MagicMock:
        module_name, class_name = _BACKEND_CLASSES[name]
        cls = MagicMock(name=class_name)
        instance = cls.return_value
        instance.model_options = {}
        if name == "openai" and "_server_type" not in instance_attributes:
            # The real backend resolves an unset OPENAI_BASE_URL as the OpenAI
            # platform, so that is the stub default too.
            instance._server_type = _ServerType.OPENAI
        for key, value in instance_attributes.items():
            setattr(instance, key, value)
        module = types.ModuleType(module_name)
        setattr(module, class_name, cls)
        monkeypatch.setitem(sys.modules, module_name, module)
        return cls

    with patch("cli.eval.runner.mellea.MelleaSession"):
        yield _stub


@pytest.mark.parametrize("backend", ["ollama", "openai", "watsonx", "litellm"])
def test_create_session_without_model_uses_backend_default(stub_backend, backend):
    cls = stub_backend(backend)
    create_session(backend=backend, model=None, max_tokens=8)
    assert "model_id" not in cls.call_args.kwargs


def test_create_session_hf_without_model_falls_back_to_granite_4_2(stub_backend):
    cls = stub_backend("hf")
    create_session(backend="hf", model=None, max_tokens=8)
    assert cls.call_args.kwargs["model_id"] is model_ids.IBM_GRANITE_4_2_3B


def test_create_session_string_model_passes_through(stub_backend):
    cls = stub_backend("ollama")
    create_session(backend="ollama", model="granite4.2:8b", max_tokens=8)
    assert cls.call_args.kwargs["model_id"] == "granite4.2:8b"


def test_create_session_attribute_name_resolves_to_model_identifier(stub_backend):
    cls = stub_backend("ollama")
    create_session(backend="ollama", model="IBM_GRANITE_4_2_8B", max_tokens=8)
    assert cls.call_args.kwargs["model_id"] is model_ids.IBM_GRANITE_4_2_8B


def test_create_session_litellm_string_model_passes_through(stub_backend):
    cls = stub_backend("litellm")
    create_session(backend="litellm", model="ollama_chat/granite4.2:3b", max_tokens=8)
    assert cls.call_args.kwargs["model_id"] == "ollama_chat/granite4.2:3b"


# --- create_session thinking default ---

_UNSET = object()  # THINKING is not in the options at all


@pytest.mark.parametrize("backend", ["ollama", "openai", "hf", "watsonx", "litellm"])
def test_create_session_builds_backend_with_token_cap_only(stub_backend, backend):
    """Only the token cap is passed at construction; thinking is set afterwards."""
    cls = stub_backend(backend)
    create_session(backend=backend, model=None, max_tokens=8)
    assert cls.call_args.kwargs["model_options"] == {ModelOption.MAX_NEW_TOKENS: 8}


@pytest.mark.parametrize("backend", ["ollama", "hf"])
def test_create_session_local_backends_turn_thinking_off(stub_backend, backend):
    cls = stub_backend(backend)
    create_session(backend=backend, model=None, max_tokens=8)
    assert cls.return_value.model_options[ModelOption.THINKING] is False


def test_create_session_watsonx_leaves_thinking_unset(stub_backend):
    cls = stub_backend("watsonx")
    create_session(backend="watsonx", model=None, max_tokens=8)
    assert cls.return_value.model_options.get(ModelOption.THINKING, _UNSET) is _UNSET


@pytest.mark.parametrize(
    ("server_type", "expected"),
    [
        pytest.param(_ServerType.OPENAI, _UNSET, id="openai-platform"),
        pytest.param(_ServerType.LOCALHOST, False, id="local-server"),
        pytest.param(_ServerType.UNKNOWN, False, id="unknown-server"),
    ],
)
def test_create_session_openai_thinking_follows_server(
    stub_backend, server_type, expected
):
    cls = stub_backend("openai", _server_type=server_type)
    # A model is named so the off-platform cases reach the thinking logic
    # instead of the missing-model error.
    create_session(backend="openai", model="granite4.2:3b", max_tokens=8)
    options = cls.return_value.model_options
    assert options.get(ModelOption.THINKING, _UNSET) is expected


@pytest.mark.parametrize(
    ("model", "resolved_model_id", "expected"),
    [
        pytest.param(
            None,
            "ollama_chat/granite4.2:3b",
            False,
            id="backend-default-is-ollama-routed",
        ),
        pytest.param(
            "ollama_chat/granite4.2:3b",
            "ollama_chat/granite4.2:3b",
            False,
            id="ollama-route",
        ),
        pytest.param("openai/gpt-4o", "openai/gpt-4o", _UNSET, id="non-ollama-route"),
    ],
)
def test_create_session_litellm_thinking_follows_route(
    stub_backend, model, resolved_model_id, expected
):
    cls = stub_backend("litellm", _model_id=resolved_model_id)
    create_session(backend="litellm", model=model, max_tokens=8)
    options = cls.return_value.model_options
    assert options.get(ModelOption.THINKING, _UNSET) is expected


# --- create_session openai model requirement ---


@pytest.mark.parametrize(
    "server_type",
    [_ServerType.LOCALHOST, _ServerType.UNKNOWN, _ServerType.REMOTE_VLLM],
    ids=["localhost", "unknown", "remote-vllm"],
)
def test_create_session_openai_without_model_raises_off_platform(
    stub_backend, server_type
):
    stub_backend("openai", _server_type=server_type)
    with pytest.raises(ValueError, match="--model"):
        create_session(backend="openai", model=None, max_tokens=8)


def test_create_session_openai_without_model_ok_on_platform(stub_backend):
    cls = stub_backend("openai", _server_type=_ServerType.OPENAI)
    create_session(backend="openai", model=None, max_tokens=8)
    assert "model_id" not in cls.call_args.kwargs


def test_create_session_openai_with_model_ok_off_platform(stub_backend):
    cls = stub_backend("openai", _server_type=_ServerType.LOCALHOST)
    create_session(backend="openai", model="granite4.2:3b", max_tokens=8)
    assert cls.call_args.kwargs["model_id"] == "granite4.2:3b"


# --- _resolved_model_name ---


def test_resolved_model_name_gives_the_backend_specific_name():
    session = types.SimpleNamespace(
        backend=types.SimpleNamespace(model_id=model_ids.IBM_GRANITE_4_2_3B)
    )
    assert _resolved_model_name(session, "ollama") == "granite4.2:3b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
