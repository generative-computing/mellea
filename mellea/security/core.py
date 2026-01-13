"""Core security functionality for mellea.

This module provides the fundamental security classes and functions for
tracking security levels of content blocks and enforcing security policies.
"""

import abc
import functools
from collections.abc import Callable
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from mellea.stdlib.base import CBlock, Component

T = TypeVar("T")


class SecLevelType(str, Enum):
    """Security level type constants."""

    NONE = "none"
    CLASSIFIED = "classified"
    TAINTED_BY = "tainted_by"


class AccessType(Generic[T], abc.ABC):
    """Abstract base class for access-based security.

    This trait allows integration with IAM systems and provides fine-grained
    access control based on entitlements rather than coarse security levels.
    """

    @abc.abstractmethod
    def has_access(self, entitlement: T | None) -> bool:
        """Check if the given entitlement has access.

        Args:
            entitlement: The entitlement to check (e.g., user role, IAM identifier)

        Returns:
            True if the entitlement has access, False otherwise
        """


class SecLevel(Generic[T]):
    """Security level with access-based control and taint tracking.

    SecLevel := None | Classified of AccessType | TaintedBy of (list[CBlock | Component] | None)
    """

    def __init__(self, level_type: SecLevelType | str, data: Any = None):
        """Initialize security level.

        Args:
            level_type: Type of security level (SecLevelType enum or string)
            data: Associated data (AccessType for classified, list[CBlock|Component] for tainted_by)
        """
        # Convert string to enum if needed for backward compatibility
        if isinstance(level_type, str):
            level_type = SecLevelType(level_type)
        self.level_type = level_type
        self.data = data

    @classmethod
    def none(cls) -> "SecLevel":
        """Create a SecLevel with no restrictions (safe)."""
        return cls(SecLevelType.NONE)

    @classmethod
    def classified(cls, access_type: AccessType[T]) -> "SecLevel":
        """Create a SecLevel with classified access requirements."""
        return cls(SecLevelType.CLASSIFIED, access_type)

    @classmethod
    def tainted_by(
        cls, sources: "CBlock | Component | list[CBlock | Component] | None"
    ) -> "SecLevel":
        """Create a SecLevel tainted by one or more CBlocks or Components.

        Args:
            sources: A single CBlock/Component, a list of CBlocks/Components, or None for root nodes.
                    If a single source is provided, it will be converted to a list internally.

        Returns:
            SecLevel with TAINTED_BY type
        """
        # Normalize to list: convert single source to list, None to empty list
        if sources is None:
            sources_list: list[CBlock | Component] = []
        elif isinstance(sources, list):
            sources_list = sources
        else:
            sources_list = [sources]

        return cls(SecLevelType.TAINTED_BY, sources_list)

    def is_tainted(self) -> bool:
        """Check if this security level represents tainted content.

        Returns:
            True if tainted, False otherwise
        """
        return self.level_type == SecLevelType.TAINTED_BY

    def is_classified(self) -> bool:
        """Check if this security level represents classified content.

        Returns:
            True if classified, False otherwise
        """
        return self.level_type == SecLevelType.CLASSIFIED

    def get_access_type(self) -> AccessType[T] | None:
        """Get the AccessType for classified content.

        Returns:
            The AccessType if this is classified, None otherwise
        """
        if self.level_type == SecLevelType.CLASSIFIED:
            return self.data
        return None

    def get_taint_sources(self) -> "list[CBlock | Component]":
        """Get all sources of taint if this is a tainted level.

        Returns:
            List of CBlocks or Components that tainted this content, empty list if not tainted
        """
        if self.level_type == SecLevelType.TAINTED_BY:
            if isinstance(self.data, list):
                return self.data
            # Handle legacy single-source format (shouldn't happen in new code)
            return [self.data] if self.data is not None else []
        return []


class SecurityError(Exception):
    """Exception raised for security-related errors."""


@runtime_checkable
class TaintChecking(Protocol):
    """Protocol for objects that can provide security level information.

    This protocol allows uniform access to security levels without
    relying on hasattr checks or _meta dictionary access.
    """

    @property
    def sec_level(self) -> "SecLevel | None":
        """Get the security level for this object.

        Returns:
            SecLevel if present, None otherwise
        """
        ...


def taint_sources(action: "Component | CBlock", ctx: Any) -> "list[CBlock | Component]":
    """Compute taint sources from action and context.

    This function examines the action and context to determine what
    security sources might be present. It performs recursive analysis
    of Component parts and shallow analysis of context to identify
    potential taint sources and returns the actual objects that are tainted.

    Args:
        action: The action component or content block
        ctx: The context containing previous interactions

    Returns:
        List of tainted CBlocks or Components
    """
    from mellea.stdlib.base import (
        CBlock,
        Component,
    )  # Import here to avoid circular dependency

    sources = []

    # Check if action has security level and is tainted
    if isinstance(action, TaintChecking):
        sec_level = action.sec_level
        if sec_level is not None and sec_level.is_tainted():
            sources.append(action)

    # For Components, check their constituent parts for taint
    # Use pattern matching: CBlock doesn't have parts, Components do
    match action:
        case CBlock():
            # CBlock doesn't have parts, nothing to do
            pass
        case _ if isinstance(action, Component):
            # Component is @runtime_checkable, so isinstance() works
            # If it's a Component, it has parts() method by protocol definition
            parts = action.parts()
            for part in parts:
                # Check if the part itself is tainted
                if isinstance(part, TaintChecking):
                    sec_level = part.sec_level
                    if sec_level is not None and sec_level.is_tainted():
                        sources.append(part)
                # Recursively check Component parts for nested taint sources
                # (Components can contain other Components with tainted CBlocks)
                if isinstance(part, Component):
                    nested_sources = taint_sources(part, None)
                    sources.extend(nested_sources)

    # Check context for tainted content (shallow check of recent items, but recursive within each)
    if hasattr(ctx, "as_list"):
        try:
            context_items = ctx.as_list(
                last_n_components=5
            )  # Limit to recent items for performance
            for item in context_items:
                # Recursively check each context item (same as action check)
                # Only append if item is actually a CBlock or Component (not just TaintChecking)
                if isinstance(item, CBlock | Component) and isinstance(
                    item, TaintChecking
                ):
                    sec_level = item.sec_level
                    if sec_level is not None and sec_level.is_tainted():
                        sources.append(item)
                # Recursively check Component parts in context items
                if isinstance(item, Component):
                    nested_sources = taint_sources(item, None)
                    sources.extend(nested_sources)
        except Exception:
            # If context analysis fails, continue without it
            pass

    return sources


F = TypeVar("F", bound=Callable[..., Any])


def privileged(func: F) -> F:
    """Decorator to mark functions that require safe (non-tainted, non-classified) input.

    Functions decorated with @privileged will raise SecurityError if
    called with tainted or classified content blocks.

    Args:
        func: The function to decorate

    Returns:
        The decorated function

    Raises:
        SecurityError: If the function is called with tainted or classified content
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check all arguments for marked content (tainted or classified)
        for arg in args:
            if isinstance(arg, TaintChecking):
                sec_level = arg.sec_level
                if sec_level is not None:
                    if sec_level.is_tainted():
                        taint_sources = sec_level.get_taint_sources()
                        if taint_sources:
                            source_names = ", ".join(
                                type(s).__name__ for s in taint_sources
                            )
                            source_info = f" (tainted by: {source_names})"
                        else:
                            source_info = ""
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"tainted content{source_info}"
                        )
                    elif sec_level.is_classified():
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"classified content"
                        )

        # Check keyword arguments for marked content (tainted or classified)
        for key, value in kwargs.items():
            if isinstance(value, TaintChecking):
                sec_level = value.sec_level
                if sec_level is not None:
                    if sec_level.is_tainted():
                        taint_sources = sec_level.get_taint_sources()
                        if taint_sources:
                            source_names = ", ".join(
                                type(s).__name__ for s in taint_sources
                            )
                            source_info = f" (tainted by: {source_names})"
                        else:
                            source_info = ""
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"tainted content in argument '{key}'{source_info}"
                        )
                    elif sec_level.is_classified():
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"classified content in argument '{key}'"
                        )

        return func(*args, **kwargs)

    return wrapper  # type: ignore


def declassify(cblock: "CBlock") -> "CBlock":
    """Create a declassified version of a CBlock (non-mutating).

    This function creates a new CBlock with the same content but marked
    as safe (SecLevel.none()). The original CBlock is not modified.

    Args:
        cblock: The CBlock to declassify

    Returns:
        A new CBlock with safe security level
    """
    from mellea.stdlib.base import CBlock  # Import here to avoid circular dependency

    # Return new CBlock with same content but safe security metadata
    return CBlock(
        cblock.value,
        cblock._meta.copy() if cblock._meta else None,
        sec_level=SecLevel.none(),
    )
