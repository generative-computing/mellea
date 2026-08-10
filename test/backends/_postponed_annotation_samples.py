# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample functions with postponed annotations (PEP 563), for regression tests.

Isolated in its own module because `from __future__ import annotations` is a
module-level directive — it cannot be scoped to a single test function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decimal import Decimal


@dataclass
class Address:
    """A custom, non-builtin parameter type."""

    city: str


@dataclass
class Period:
    """A custom parameter type for testing."""

    name: str


@dataclass
class Region:
    """A custom parameter type deliberately not imported by the test module.

    Used to prove annotations resolve in this module's namespace rather than
    the namespace of a decorator applied elsewhere.
    """

    code: str


def send_letter(to: Address) -> str:
    """Send a letter to the given address.

    Args:
        to: the destination address
    """
    return "sent"


def tc_only_return_builtin_param(query: str) -> Decimal:
    """TYPE_CHECKING-only return with builtin params.

    Args:
        query: query string
    """
    return Decimal("0")


def tc_return_custom_param(period: Period) -> Decimal:
    """TYPE_CHECKING-only return with custom param.

    Args:
        period: the period
    """
    return Decimal("0")


def tc_return_region(region: Region) -> Decimal:
    """TYPE_CHECKING-only return with a param type local to this module.

    Args:
        region: the region
    """
    return Decimal("0")


def unresolvable_param(query: NonExistentType) -> str:  # type: ignore[name-defined]  # noqa: F821
    """Unresolvable parameter annotation.

    Args:
        query: query string
    """
    return "ok"
