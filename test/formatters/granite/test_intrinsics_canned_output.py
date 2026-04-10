# SPDX-License-Identifier: Apache-2.0

"""Unit tests for IntrinsicsResultProcessor using canned model outputs.

These tests exercise the full output-processing pipeline (JSON parsing,
transformation rules, float-rounding comparison) against pre-recorded model
outputs stored in ``testdata/``.  They require **no GPU, no network, and no
model downloads** — only a local YAML config file and the three JSON artifacts
per scenario (canned input, model output, expected result).

Scenarios that need a YAML config from HuggingFace Hub are skipped unless the
YAML has been previously cached locally or a ``config_dict`` is embedded here.
"""

import copy
import json
import pathlib

import pytest

from mellea.formatters.granite import (
    ChatCompletion,
    ChatCompletionResponse,
    IntrinsicsResultProcessor,
)
from mellea.formatters.granite.intrinsics import json_util

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "testdata"
_CANNED_INPUT_DIR = _TEST_DATA_DIR / "test_canned_input"
_MODEL_OUTPUT_DIR = _TEST_DATA_DIR / "test_canned_output" / "model_output"
_EXPECTED_DIR = _TEST_DATA_DIR / "test_canned_output" / "expected_result"
_YAML_DIR = _TEST_DATA_DIR / "input_yaml"


# ---------------------------------------------------------------------------
# Float-tolerant comparison (extracted from test_intrinsics_formatters.py)
# ---------------------------------------------------------------------------


def _round_floats(json_data, num_digits: int = 2):
    """Round all floating-point numbers in a JSON value to facilitate comparisons.

    Handles floats, float-encoded strings, and JSON objects/arrays encoded as
    strings (recursive).
    """
    result = copy.deepcopy(json_data)
    for path in json_util.scalar_paths(result):
        value = json_util.fetch_path(result, path)
        if isinstance(value, float):
            json_util.replace_path(result, path, round(value, num_digits))
        elif isinstance(value, str):
            try:
                str_as_float = float(value)
                json_util.replace_path(result, path, round(str_as_float, num_digits))
            except ValueError:
                pass

            if value and value[0] in ("{", "["):
                try:
                    str_as_json = json.loads(value)
                    rounded_json = _round_floats(str_as_json, num_digits)
                    rounded_json_as_str = json.dumps(rounded_json)
                    json_util.replace_path(result, path, rounded_json_as_str)
                except json.JSONDecodeError:
                    pass
    return result


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Each tuple: (short_name, yaml_source)
# yaml_source is either a Path to a local YAML file or a dict that can be
# passed directly as ``config_dict``.

_ANSWERABILITY_YAML = _YAML_DIR / "answerability.yaml"

_SCENARIOS: list[tuple[str, pathlib.Path | dict]] = []

# Answerability scenarios — local YAML available
for _name in (
    "answerability_simple",
    "answerability_answerable",
    "answerability_unanswerable",
):
    if (_CANNED_INPUT_DIR / f"{_name}.json").exists():
        _SCENARIOS.append((_name, _ANSWERABILITY_YAML))

# Query-rewrite — inline config (simple nest transformation)
_QUERY_REWRITE_CONFIG: dict = {
    "model": None,
    "response_format": {
        "type": "object",
        "properties": {"rewritten_question": {"type": "string"}},
        "required": ["rewritten_question"],
    },
    "transformations": None,
    "instruction": None,
    "parameters": {"max_completion_tokens": 256},
    "sentence_boundaries": None,
}

if (_CANNED_INPUT_DIR / "query_rewrite.json").exists():
    _SCENARIOS.append(("query_rewrite", _QUERY_REWRITE_CONFIG))

# Query-clarification — inline config (no transformations)
_QUERY_CLARIFICATION_CONFIG: dict = {
    "model": None,
    "response_format": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
    "transformations": None,
    "instruction": None,
    "parameters": {"max_completion_tokens": 256},
    "sentence_boundaries": None,
}

if (_CANNED_INPUT_DIR / "query_clarification.json").exists():
    _SCENARIOS.append(("query_clarification", _QUERY_CLARIFICATION_CONFIG))


def _scenario_ids():
    return [s[0] for s in _SCENARIOS]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCannedOutput:
    """Regression tests: process canned model outputs and compare to expected results."""

    @pytest.fixture(params=range(len(_SCENARIOS)), ids=_scenario_ids())
    def scenario(self, request):
        name, yaml_source = _SCENARIOS[request.param]
        input_file = _CANNED_INPUT_DIR / f"{name}.json"
        output_file = _MODEL_OUTPUT_DIR / f"{name}.json"
        expected_file = _EXPECTED_DIR / f"{name}.json"

        for f in (input_file, output_file, expected_file):
            if not f.exists():
                pytest.skip(f"Missing test data: {f}")

        return name, yaml_source, input_file, output_file, expected_file

    def test_transform_matches_expected(self, scenario):
        name, yaml_source, input_file, output_file, expected_file = scenario

        # Build processor from local YAML or inline dict
        if isinstance(yaml_source, dict):
            processor = IntrinsicsResultProcessor(config_dict=yaml_source)
        else:
            processor = IntrinsicsResultProcessor(config_file=yaml_source)

        with open(input_file, encoding="utf-8") as f:
            model_input = ChatCompletion.model_validate_json(f.read())
        with open(output_file, encoding="utf-8") as f:
            model_output = ChatCompletionResponse.model_validate_json(f.read())

        transformed = processor.transform(model_output, model_input)
        transformed_json = _round_floats(json.loads(transformed.model_dump_json()))

        with open(expected_file, encoding="utf-8") as f:
            expected = ChatCompletionResponse.model_validate_json(f.read())
        expected_json = _round_floats(json.loads(expected.model_dump_json()))

        assert transformed_json == expected_json, f"Canned output mismatch for '{name}'"


class TestCannedOutputParsing:
    """Verify that canned test data files parse as valid Pydantic models."""

    @staticmethod
    def _canned_input_files():
        return sorted(_CANNED_INPUT_DIR.glob("*.json"))

    @staticmethod
    def _model_output_files():
        return sorted(_MODEL_OUTPUT_DIR.glob("*.json"))

    @staticmethod
    def _expected_result_files():
        return sorted(_EXPECTED_DIR.glob("*.json"))

    @pytest.fixture(params=[p.name for p in sorted(_CANNED_INPUT_DIR.glob("*.json"))])
    def canned_input_file(self, request):
        return _CANNED_INPUT_DIR / request.param

    @pytest.fixture(params=[p.name for p in sorted(_MODEL_OUTPUT_DIR.glob("*.json"))])
    def model_output_file(self, request):
        return _MODEL_OUTPUT_DIR / request.param

    @pytest.fixture(params=[p.name for p in sorted(_EXPECTED_DIR.glob("*.json"))])
    def expected_result_file(self, request):
        return _EXPECTED_DIR / request.param

    def test_canned_input_validates(self, canned_input_file):
        """All canned input files must parse as valid ChatCompletion objects."""
        with open(canned_input_file, encoding="utf-8") as f:
            cc = ChatCompletion.model_validate_json(f.read())
        assert cc.messages is not None

    def test_model_output_validates(self, model_output_file):
        """All model output files must parse as valid ChatCompletionResponse objects."""
        with open(model_output_file, encoding="utf-8") as f:
            resp = ChatCompletionResponse.model_validate_json(f.read())
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content is not None

    def test_expected_result_validates(self, expected_result_file):
        """All expected result files must parse as valid ChatCompletionResponse objects."""
        with open(expected_result_file, encoding="utf-8") as f:
            resp = ChatCompletionResponse.model_validate_json(f.read())
        assert len(resp.choices) >= 1
