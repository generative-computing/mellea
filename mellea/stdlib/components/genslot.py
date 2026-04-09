"""Backward-compatibility shim — use ``mellea.stdlib.components.genstub`` instead.

.. deprecated::
    The ``genslot`` module has been renamed to ``genstub``.  This shim will be
    removed in a future release.
"""

import warnings as _warnings

_warnings.warn(
    "mellea.stdlib.components.genslot has been renamed to "
    "mellea.stdlib.components.genstub. Please update your imports. "
    "If you have the m cli installed, you can use m fix genslots. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical module so existing code keeps working.
from .genstub import *  # noqa: F403, E402

# Old class-name aliases — the module-level warning above already told the user
# to migrate, so no per-name warning here.
from .genstub import (  # noqa: E402
    AsyncGenerativeStub as AsyncGenerativeSlot,
    GenerativeStub as GenerativeSlot,
    SyncGenerativeStub as SyncGenerativeSlot,
    __all__,
)
