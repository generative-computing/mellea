"""Security module for mellea.

This module provides security features for tracking and managing the security
level of content blocks and components in the mellea library.
"""

from .core import (
    AccessType,
    SecLevel,
    SecurityMetadata,
    SecurityError,
    privileged,
    declassify,
    taint_sources,
)

__all__ = [
    "AccessType",
    "SecLevel",
    "SecurityMetadata", 
    "SecurityError",
    "privileged",
    "declassify",
    "taint_sources",
]