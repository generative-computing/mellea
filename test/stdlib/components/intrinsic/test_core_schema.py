# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for requirement_check schema validation in core.py.

These tests mock call_intrinsic so they run without a GPU or HF backend.
"""

import math
from unittest.mock import patch

import pytest

from mellea.backends.adapters import AdapterSchemaMismatchError
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext

_CTX = ChatContext()
_BACKEND = object()
_REQUIREMENT = "must be polite"
_PATCH = "mellea.stdlib.components.intrinsic.core.call_intrinsic"


def _call(result_json: dict) -> float:
    with patch(_PATCH, return_value=result_json):
        return core.requirement_check(_CTX, _BACKEND, _REQUIREMENT)  # type: ignore[arg-type]


def test_valid_score_returned():
    assert _call({"requirement_check": {"score": 0.8}}) == pytest.approx(0.8)


def test_missing_requirement_check_key_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"other_field": 0.9})


def test_null_requirement_check_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": None})


def test_list_requirement_check_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": []})


def test_missing_score_key_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"other_key": 0.9}})


def test_null_score_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": None}})


def test_string_score_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": "0.9"}})


def test_bool_score_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": True}})


def test_nan_score_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": math.nan}})


def test_inf_score_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": math.inf}})


def test_score_above_range_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": 1.5}})


def test_score_below_range_raises():
    with pytest.raises(AdapterSchemaMismatchError):
        _call({"requirement_check": {"score": -0.1}})


def test_boundary_score_zero():
    assert _call({"requirement_check": {"score": 0.0}}) == pytest.approx(0.0)


def test_boundary_score_one():
    assert _call({"requirement_check": {"score": 1.0}}) == pytest.approx(1.0)
