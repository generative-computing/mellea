# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample functions with postponed annotations (PEP 563), for regression tests.

Isolated in its own module because `from __future__ import annotations` is a
module-level directive — it cannot be scoped to a single test function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Address:
    """A custom, non-builtin parameter type."""

    city: str


def send_letter(to: Address) -> str:
    """Send a letter to the given address.

    Args:
        to: the destination address
    """
    return "sent"
