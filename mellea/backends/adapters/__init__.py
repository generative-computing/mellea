"""Classes and Functions for Backend Adapters."""

from ._core import (
    Adapter,
    AdapterSchemaMismatchError,
    EmbeddedBinding,
    Identity,
    IOContract,
    LocalFileBinding,
    ServerMediatedBinding,
    WeightsBinding,
)
from .adapter import (
    AdapterMixin,
    AdapterType,
    EmbeddedIntrinsicAdapter,
    IntrinsicAdapter,
    LocalHFAdapter,
    fetch_intrinsic_metadata,
    get_adapter_for_intrinsic,
)
from .roles import KNOWN_ROLES

__all__ = [
    "KNOWN_ROLES",
    "Adapter",
    "AdapterMixin",
    "AdapterSchemaMismatchError",
    "AdapterType",
    "EmbeddedBinding",
    "EmbeddedIntrinsicAdapter",
    "IOContract",
    "Identity",
    "IntrinsicAdapter",
    "LocalFileBinding",
    "LocalHFAdapter",
    "ServerMediatedBinding",
    "WeightsBinding",
    "fetch_intrinsic_metadata",
    "get_adapter_for_intrinsic",
]
