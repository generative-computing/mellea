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
from .capabilities import KNOWN_CAPABILITIES
from .catalog import validate_revision

__all__ = [
    "KNOWN_CAPABILITIES",
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
    "validate_revision",
]
