# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures with postponed annotations (PEP 563), used by test_genstub_unit.py.

Kept in a separate module because `from __future__ import annotations` is a
module-level switch — isolating it here keeps the rest of the test suite on
normal (resolved) annotations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Requirement:
    """A single extracted requirement."""

    id: str
    text: str


def extract_requirements(product_description: str) -> list[Requirement]:
    """Extract requirements from a product description."""


def greet(name: str) -> str:
    """Say hello."""
