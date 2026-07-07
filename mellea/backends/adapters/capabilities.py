"""Advisory registry of known adapter capabilities.

:data:`KNOWN_CAPABILITIES` is a frozenset of capability strings derived from the
adapter function catalog. It is advisory only: callers are warned (not rejected) when a
capability outside this set is used, so that custom adapters and pre-release
adapter functions are not blocked.

Deriving from the catalog (rather than hand-copying) keeps the two registries in
sync automatically — adding a new entry to ``catalog.py`` automatically registers
its :attr:`~mellea.backends.adapters.catalog.IntrinsicsCatalogEntry.effective_capability`
as a known capability.
"""

from .catalog import _INTRINSICS_CATALOG_ENTRIES

KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    e.effective_capability for e in _INTRINSICS_CATALOG_ENTRIES
)
