# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classes and Functions for Backend Adapters."""

from ._core import (
    Adapter,
    AdapterSchemaMismatchError,
    EmbeddedActivationRequest,
    EmbeddedBinding,
    Identity,
    IOContract,
    LocalFileBinding,
    ServerMediatedBinding,
    WeightsBinding,
)
from .adapter import (
    AdapterInput,
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
from .io_contracts import get_io_contract

__all__ = [
    "KNOWN_CAPABILITIES",
    "Adapter",
    "AdapterInput",
    "AdapterMixin",
    "AdapterSchemaMismatchError",
    "AdapterType",
    "EmbeddedActivationRequest",
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
    "get_io_contract",
    "validate_revision",
]
