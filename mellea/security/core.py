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
    from ..core.base import CBlock, Component

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


def _collect_sources_by_predicate(
    action: "Component | CBlock", ctx: Any, predicate: Callable[["SecLevel"], bool]
) -> "list[CBlock | Component]":
    """Recursively collect CBlocks/Components whose sec_level satisfies predicate.

    Shared logic for taint_sources and classified_sources. Walks action and
    context (shallow), recursing into Component.parts().
    """
    from ..core.base import (
        CBlock,
        Component,
    )  # Import here to avoid circular dependency

    sources = []

    if isinstance(action, TaintChecking):
        sec_level = action.sec_level
        if sec_level is not None and predicate(sec_level):
            sources.append(action)

    match action:
        case CBlock():
            pass
        case _ if isinstance(action, Component):
            parts = action.parts()
            for part in parts:
                if isinstance(part, TaintChecking):
                    sec_level = part.sec_level
                    if sec_level is not None and predicate(sec_level):
                        sources.append(part)
                if isinstance(part, Component):
                    nested = _collect_sources_by_predicate(part, None, predicate)
                    sources.extend(nested)

    if hasattr(ctx, "as_list"):
        try:
            context_items = ctx.as_list(last_n_components=5)
            for item in context_items:
                if isinstance(item, CBlock | Component) and isinstance(
                    item, TaintChecking
                ):
                    sec_level = item.sec_level
                    if sec_level is not None and predicate(sec_level):
                        sources.append(item)
                if isinstance(item, Component):
                    nested = _collect_sources_by_predicate(item, None, predicate)
                    sources.extend(nested)
        except Exception:
            pass

    return sources


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
    return _collect_sources_by_predicate(action, ctx, lambda sec: sec.is_tainted())


def classified_sources(
    action: "Component | CBlock", ctx: Any = None
) -> "list[CBlock | Component]":
    """Compute classified sources from action and context.

    Recursively examines the action and context (same structure as
    taint_sources) and returns all CBlocks or Components that have
    classified security level.

    Args:
        action: The action component or content block
        ctx: Optional context containing previous interactions (shallow scan)

    Returns:
        List of classified CBlocks or Components
    """
    return _collect_sources_by_predicate(action, ctx, lambda sec: sec.is_classified())


F = TypeVar("F", bound=Callable[..., Any])


def _raise_if_privilege_violation(
    obj: Any, func_name: str, arg_name: str | None = None
) -> None:
    """Raise SecurityError if obj or any nested part is tainted or classified.

    Uses taint_sources() and classified_sources() for recursive detection.
    """
    suffix = f" in argument '{arg_name}'" if arg_name else ""

    sources = taint_sources(obj, None)
    if sources:
        source_names = ", ".join(type(s).__name__ for s in sources)
        raise SecurityError(
            f"Function {func_name} requires safe input, but received "
            f"tainted content (tainted by: {source_names}){suffix}"
        )

    sources = classified_sources(obj, None)
    if sources:
        source_names = ", ".join(type(s).__name__ for s in sources)
        raise SecurityError(
            f"Function {func_name} requires safe input, but received "
            f"classified content (sources: {source_names}){suffix}"
        )


def privileged(func: F) -> F:
    """Decorator to mark functions that require safe (non-tainted, non-classified) input.

    Functions decorated with @privileged will raise SecurityError if
    called with tainted or classified content blocks. Checks are performed
    recursively: if any argument is a Component, its parts (and their parts)
    are also checked for taint or classified content.

    Args:
        func: The function to decorate

    Returns:
        The decorated function

    Raises:
        SecurityError: If the function is called with tainted or classified content
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            _raise_if_privilege_violation(arg, func.__name__, None)
        for key, value in kwargs.items():
            _raise_if_privilege_violation(value, func.__name__, key)
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
    from ..core.base import CBlock  # Import here to avoid circular dependency

    # Return new CBlock with same content but safe security metadata
    return CBlock(
        cblock.value,
        cblock._meta.copy() if cblock._meta else None,
        sec_level=SecLevel.none(),
    )
