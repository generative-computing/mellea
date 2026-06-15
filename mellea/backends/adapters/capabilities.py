"""Advisory registry of known adapter capabilities.

:data:`KNOWN_CAPABILITIES` is a frozenset of capability strings derived from the
intrinsics catalog. It is advisory only: callers are warned (not rejected) when a
capability outside this set is used, so that custom adapters and pre-release
intrinsics are not blocked.

Deriving from the catalog (rather than hand-copying) keeps the two registries in
sync automatically — adding a new entry to ``catalog.py`` automatically registers
its :attr:`~mellea.backends.adapters.catalog.IntriniscsCatalogEntry.effective_capability`
as a known capability.
"""

from .catalog import _INTRINSICS_CATALOG_ENTRIES

KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    e.effective_capability for e in _INTRINSICS_CATALOG_ENTRIES
)
