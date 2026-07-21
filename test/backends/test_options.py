# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from mellea.backends import ModelOption
from mellea.backends._options import resolve_model_options


def test_resolve_model_options_call_wins_over_helper_and_backend():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.TEMPERATURE: 0.0},
        remap={},
        helper_defaults={ModelOption.TEMPERATURE: 0.5},
        call_options={ModelOption.TEMPERATURE: 0.9},
    )
    assert resolved[ModelOption.TEMPERATURE] == 0.9


def test_resolve_model_options_helper_wins_over_backend_when_no_call_override():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.TEMPERATURE: 0.0},
        remap={},
        helper_defaults={ModelOption.TEMPERATURE: 0.5},
        call_options=None,
    )
    assert resolved[ModelOption.TEMPERATURE] == 0.5


def test_resolve_model_options_backend_used_when_nothing_else_set():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.TEMPERATURE: 0.0}, remap={}, call_options=None
    )
    assert resolved[ModelOption.TEMPERATURE] == 0.0


def test_resolve_model_options_applies_remap_to_backend_and_call_options():
    resolved = resolve_model_options(
        backend_defaults={"temp": 0.0},
        remap={"temp": ModelOption.TEMPERATURE},
        call_options={"temp": 0.7},
    )
    assert resolved == {ModelOption.TEMPERATURE: 0.7}


def test_resolve_model_options_none_call_options_keeps_helper_and_backend_merged():
    resolved = resolve_model_options(
        backend_defaults={ModelOption.CONTEXT_WINDOW: 4096},
        remap={},
        helper_defaults={ModelOption.TEMPERATURE: 0.0},
        call_options=None,
    )
    assert resolved == {ModelOption.CONTEXT_WINDOW: 4096, ModelOption.TEMPERATURE: 0.0}


def test_resolve_model_options_unrelated_keys_are_preserved():
    resolved = resolve_model_options(
        backend_defaults={"backend_only": 1},
        remap={},
        helper_defaults={"helper_only": 2},
        call_options={"call_only": 3},
    )
    assert resolved == {"backend_only": 1, "helper_only": 2, "call_only": 3}
