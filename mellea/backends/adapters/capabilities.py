# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory registry of known adapter capabilities.

:data:`KNOWN_CAPABILITIES` is a frozenset of capability strings derived from the
adapter function catalog, plus the decision capability. It is advisory only:
callers are warned (not rejected) when a capability outside this set is used,
so that custom adapters and pre-release adapter functions are not blocked.

Deriving from the catalog (rather than hand-copying) keeps the two registries in
sync automatically; adding a new entry to `catalog.py` automatically registers
its :attr:`~mellea.backends.adapters.catalog.IntrinsicsCatalogEntry.effective_capability`
as a known capability. Capabilities that ship in Granite Switch checkpoints
(the one-token decision capability) have no catalog entry; the catalog pins Hub
repos for local-file adapters; so it is unioned in directly.
"""

from .catalog import _INTRINSICS_CATALOG_ENTRIES

DECISION_CAPABILITY = "decision"
"""Capability name of the one-token decision capability.

All published Granite Switch checkpoints name the adapter `decision`.
Mellea uses this constant everywhere: the carried-config merge, the
io-contract registry, the capability registry, and the decision helpers."""

KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    e.effective_capability for e in _INTRINSICS_CATALOG_ENTRIES
) | {DECISION_CAPABILITY}
