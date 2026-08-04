# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical `Span` union alias and its re-exports.

`Span = Component | CBlock | ModelOutputThunk` is the single source of truth
for the three node-shaped types that flow through the library (context nodes,
`parts()` elements, formatter/backend inputs, sampling actions). This module
pins its membership, its export surface, and the `SampleActionType` re-export.
"""

from typing import get_args

import mellea.core as core
from mellea.core import CBlock, Component, ModelOutputThunk, SampleActionType, Span


def test_span_members():
    """`Span` is exactly the three-type union, order-insensitive."""
    assert set(get_args(Span)) == {Component, CBlock, ModelOutputThunk}


def test_span_exported_from_core():
    """`Span` is importable from `mellea.core` and listed in `__all__`."""
    assert "Span" in core.__all__
    assert core.Span is Span


def test_sample_action_type_is_span():
    """`SampleActionType` is a re-export of `Span` (same object), not a copy.

    Preserves the historical `mellea.core.SampleActionType` import path from #356
    while collapsing the duplicate union definition into `Span`.
    """
    assert SampleActionType is Span
    assert "SampleActionType" in core.__all__
