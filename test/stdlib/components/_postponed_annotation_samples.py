# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample functions with postponed annotations (PEP 563), for regression tests.

Kept in a separate module because `from __future__ import annotations` is a
module-level switch — isolating it here keeps the rest of the test suite on
normal (resolved) annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decimal import Decimal


@dataclass
class Requirement:
    """A single extracted requirement."""

    id: str
    text: str


def extract_requirements(product_description: str) -> list[Requirement]:
    """Extract requirements from a product description."""


def greet(name: str) -> str:
    """Say hello."""


def price_item(item: Requirement, quantity: int) -> Decimal:
    """Price an item, returning a TYPE_CHECKING-only type.

    `Decimal` is imported only under `if TYPE_CHECKING:`, so resolving the whole
    signature raises `NameError` while every parameter resolves cleanly.
    """
