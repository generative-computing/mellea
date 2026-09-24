# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve OTLP protocol and endpoint for one telemetry signal."""

import os


def otlp_protocol_endpoint(signal: str) -> tuple[str, str | None]:
    """Return the signal's transport and endpoint, preserving HTTP URL rules."""
    prefix = f"OTEL_EXPORTER_OTLP_{signal.upper()}"
    protocol = os.getenv(f"{prefix}_PROTOCOL") or os.getenv(
        "OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"
    )
    specific_endpoint = os.getenv(f"{prefix}_ENDPOINT")
    general_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    endpoint = specific_endpoint or general_endpoint
    if protocol == "http/protobuf" and not specific_endpoint and general_endpoint:
        endpoint = f"{general_endpoint.rstrip('/')}/v1/{signal}"
    return protocol, endpoint
