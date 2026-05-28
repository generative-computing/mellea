"""Advisory registry of known adapter roles.

:data:`KNOWN_ROLES` is a frozenset of role strings derived from the intrinsics
catalog. It is advisory only: callers are warned (not rejected) when a role
outside this set is used, so that custom adapters and pre-release intrinsics
are not blocked.

Deriving from the catalog (rather than hand-copying) keeps the two registries
in sync automatically — adding a new entry to ``catalog.py`` automatically
registers it as a known role.
"""

from .catalog import _INTRINSICS_CATALOG_ENTRIES

KNOWN_ROLES: frozenset[str] = frozenset(e.name for e in _INTRINSICS_CATALOG_ENTRIES)
