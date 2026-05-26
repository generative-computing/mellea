"""Placeholder module for ``AdapterBasedComponent``.

IBM is retiring the term "Intrinsic" but has not yet confirmed a replacement.
This module re-exports :class:`~mellea.stdlib.components.intrinsic.Intrinsic`
under the provisional name ``AdapterBasedComponent`` so downstream code can
begin migrating to the new name before the old one is removed.

The old import path ``mellea.stdlib.components.intrinsic`` remains valid.
"""

from ..intrinsic import Intrinsic as AdapterBasedComponent

__all__ = ["AdapterBasedComponent"]
