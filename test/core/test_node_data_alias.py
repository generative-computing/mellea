# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical `NodeData` union alias and its re-exports.

`NodeData = Component | CBlock | ModelOutputThunk` is the single source of truth
for the three node-shaped types that flow through the library (context nodes,
`parts()` elements, formatter/backend inputs, sampling actions). This module
pins its membership, its export surface, and the `SampleActionType` re-export.
"""

from typing import get_args

import mellea.core as core
from mellea.core import CBlock, Component, ModelOutputThunk, NodeData, SampleActionType


def test_node_data_members():
    """`NodeData` is exactly the three-type union, order-insensitive."""
    assert set(get_args(NodeData)) == {Component, CBlock, ModelOutputThunk}


def test_node_data_exported_from_core():
    """`NodeData` is importable from `mellea.core` and listed in `__all__`."""
    assert "NodeData" in core.__all__
    assert core.NodeData is NodeData


def test_sample_action_type_is_node_data():
    """`SampleActionType` is a re-export of `NodeData` (same object), not a copy.

    Preserves the historical `mellea.core.SampleActionType` import path from #356
    while collapsing the duplicate union definition into `NodeData`.
    """
    assert SampleActionType is NodeData
    assert "SampleActionType" in core.__all__
