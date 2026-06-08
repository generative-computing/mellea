"""Advisory registry of known adapter capabilities.

:data:`KNOWN_CAPABILITIES` is a frozenset of capability strings derived from the
intrinsics catalog. It is advisory only: callers are warned (not rejected) when a
capability outside this set is used, so that custom adapters and pre-release
intrinsics are not blocked.

Deriving from the catalog (rather than hand-copying) keeps the two registries in
sync automatically — adding a new entry to ``catalog.py`` automatically registers
it as a known capability.

.. note::
    Capabilities are currently derived from catalog entry *names*. Decoupling the
    capability vocabulary from entry names (an explicit ``capability`` field on
    ``CatalogEntry``) is tracked as a follow-up under Epic #929.
"""

from .catalog import _INTRINSICS_CATALOG_ENTRIES

KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    e.name for e in _INTRINSICS_CATALOG_ENTRIES
)
