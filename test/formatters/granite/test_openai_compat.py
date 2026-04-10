# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying OpenAI SDK compatibility of ChatCompletion objects.

These tests ensure that ``ChatCompletion`` Pydantic models serialise to dicts
that pass OpenAI Python SDK request validation.  The SDK client is pointed at a
bogus URL so that the expected failure is ``APIConnectionError`` (network),
**not** a Pydantic/type validation error.  No network access is required.
"""

import json
import pathlib

import openai
import pytest

from mellea.formatters.granite import ChatCompletion, IntrinsicsRewriter

_INPUT_JSON_DIR = pathlib.Path(__file__).parent / "testdata" / "input_json"
_INPUT_YAML_DIR = pathlib.Path(__file__).parent / "testdata" / "input_yaml"
_INPUT_ARGS_DIR = pathlib.Path(__file__).parent / "testdata" / "input_args"

# All local input JSON files
_INPUT_FILES = sorted(_INPUT_JSON_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map input JSON stems to their YAML config + optional args.
# Only entries with a local YAML are included; others are tested as raw inputs.
_REWRITER_CONFIGS: dict[str, tuple[pathlib.Path, pathlib.Path | None]] = {
    "simple": (_INPUT_YAML_DIR / "answerability.yaml", None),
    "answerable": (_INPUT_YAML_DIR / "answerability.yaml", None),
    "unanswerable": (_INPUT_YAML_DIR / "answerability.yaml", None),
    "extra_params": (_INPUT_YAML_DIR / "answerability.yaml", None),
    "instruction": (
        _INPUT_YAML_DIR / "instruction.yaml",
        _INPUT_ARGS_DIR / "instruction.json",
    ),
}


def _fake_openai_client() -> openai.OpenAI:
    """Create an OpenAI client pointing at a bogus endpoint."""
    return openai.OpenAI(
        base_url="http://example.com:98765/not/a/valid/url",
        api_key="not_a_valid_api_key",
    )


# ---------------------------------------------------------------------------
# Tests: raw ChatCompletion → OpenAI SDK
# ---------------------------------------------------------------------------


class TestOpenAICompatRaw:
    """Verify raw (un-rewritten) ChatCompletion objects pass OpenAI SDK validation."""

    @pytest.fixture(params=[f.stem for f in _INPUT_FILES])
    def input_cc(self, request) -> ChatCompletion:
        path = _INPUT_JSON_DIR / f"{request.param}.json"
        with open(path, encoding="utf-8") as f:
            cc = ChatCompletion.model_validate_json(f.read())
        cc.model = "dummy_model_name"
        return cc

    def test_sdk_validation_passes(self, input_cc):
        """The SDK should accept the serialised dict and only fail on connection."""
        client = _fake_openai_client()
        with pytest.raises(openai.APIConnectionError):
            client.chat.completions.create(**input_cc.model_dump())


class TestOpenAICompatRewritten:
    """Verify IntrinsicsRewriter output passes OpenAI SDK validation."""

    @pytest.fixture(params=list(_REWRITER_CONFIGS))
    def rewritten_cc(self, request) -> ChatCompletion:
        stem = request.param
        yaml_file, args_file = _REWRITER_CONFIGS[stem]
        path = _INPUT_JSON_DIR / f"{stem}.json"

        with open(path, encoding="utf-8") as f:
            cc = ChatCompletion.model_validate_json(f.read())

        transform_kwargs: dict = {}
        if args_file and args_file.exists():
            with open(args_file, encoding="utf-8") as f:
                transform_kwargs = json.load(f)

        rewriter = IntrinsicsRewriter(config_file=yaml_file)
        rewritten = rewriter.transform(cc, **transform_kwargs)
        rewritten.model = "dummy_model_name"
        return rewritten

    def test_sdk_validation_passes(self, rewritten_cc):
        """Rewritten ChatCompletion should pass OpenAI SDK validation."""
        client = _fake_openai_client()
        with pytest.raises(openai.APIConnectionError):
            client.chat.completions.create(**rewritten_cc.model_dump())
