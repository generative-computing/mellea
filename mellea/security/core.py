"""Core security functionality for mellea.

This module provides the fundamental security classes and functions for
tracking security levels of content blocks and enforcing security policies.
"""

import abc
import functools
from enum import Enum
from typing import Any, Callable, Generic, TypeVar, Union

from mellea.stdlib.base import CBlock, Component


T = TypeVar('T')


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
        pass


class SecLevel(Generic[T]):
    """Security level with access-based control and taint tracking.
    
    SecLevel := None | Classified of AccessType | TaintedBy of (CBlock | Component)
    """
    
    def __init__(self, level_type: SecLevelType | str, data: Any = None):
        """Initialize security level.
        
        Args:
            level_type: Type of security level (SecLevelType enum or string)
            data: Associated data (AccessType for classified, CBlock/Component/None for tainted_by)
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
    def tainted_by(cls, source: Union[CBlock, Component, None]) -> "SecLevel":
        """Create a SecLevel tainted by a specific CBlock, Component, or None for root nodes."""
        return cls(SecLevelType.TAINTED_BY, source)
    
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
    
    def get_taint_source(self) -> Union[CBlock, Component, None]:
        """Get the source of taint if this is a tainted level.
        
        Returns:
            The CBlock or Component that tainted this content, or None
        """
        if self.level_type == SecLevelType.TAINTED_BY:
            return self.data
        return None


class SecurityMetadata:
    """Metadata for tracking security properties of content blocks."""
    
    def __init__(self, sec_level: SecLevel):
        """Initialize security metadata with a SecLevel.
        
        Args:
            sec_level: The security level for this content
        """
        self.sec_level = sec_level
    
    def is_tainted(self) -> bool:
        """Check if this security level represents tainted content.
        
        Returns:
            True if tainted, False otherwise
        """
        return self.sec_level.is_tainted()
    
    def is_classified(self) -> bool:
        """Check if this security level represents classified content.
        
        Returns:
            True if classified, False otherwise
        """
        return self.sec_level.is_classified()
    
    def get_access_type(self) -> AccessType[Any] | None:
        """Get the AccessType for classified content.
        
        Returns:
            The AccessType if this is classified, None otherwise
        """
        return self.sec_level.get_access_type()
    
    def get_taint_source(self) -> Union[CBlock, Component, None]:
        """Get the source of taint if this is a tainted level.
        
        Returns:
            The CBlock or Component that tainted this content, or None
        """
        return self.sec_level.get_taint_source()


class SecurityError(Exception):
    """Exception raised for security-related errors."""
    pass


def taint_sources(action: Union[Component, CBlock], ctx: Any) -> list[Union[CBlock, Component]]:
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
    sources = []
    
    # Check if action has security metadata and is tainted
    if hasattr(action, '_meta') and '_security' in action._meta:
        security_meta = action._meta['_security']
        if isinstance(security_meta, SecurityMetadata) and security_meta.is_tainted():
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
            try:
                parts = action.parts()
                for part in parts:
                    if hasattr(part, '_meta') and '_security' in part._meta:
                        security_meta = part._meta['_security']
                        if isinstance(security_meta, SecurityMetadata) and security_meta.is_tainted():
                            sources.append(part)
            except Exception:
                # If parts() fails, continue without it
                pass
        case _:
            # For other types that might have parts(), check with hasattr
            try:
                if hasattr(action, 'parts'):
                    parts = action.parts()
                    for part in parts:
                        if hasattr(part, '_meta') and '_security' in part._meta:
                            security_meta = part._meta['_security']
                            if isinstance(security_meta, SecurityMetadata) and security_meta.is_tainted():
                                sources.append(part)
            except Exception:
                # If parts() fails or doesn't exist, continue without it
                pass
    
    # Check context for tainted content (shallow check)
    if hasattr(ctx, 'as_list'):
        try:
            context_items = ctx.as_list(last_n_components=5)  # Limit to recent items
            for item in context_items:
                if hasattr(item, '_meta') and '_security' in item._meta:
                    security_meta = item._meta['_security']
                    if isinstance(security_meta, SecurityMetadata) and security_meta.is_tainted():
                        sources.append(item)
        except Exception:
            # If context analysis fails, continue without it
            pass
    
    return sources


F = TypeVar('F', bound=Callable[..., Any])


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
            if isinstance(arg, CBlock) and hasattr(arg, '_meta') and '_security' in arg._meta:
                security_meta = arg._meta['_security']
                if isinstance(security_meta, SecurityMetadata):
                    if security_meta.is_tainted():
                        taint_source = security_meta.get_taint_source()
                        source_info = f" (tainted by: {type(taint_source).__name__})" if taint_source else ""
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"tainted content{source_info}"
                        )
                    elif security_meta.is_classified():
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"classified content"
                        )
        
        # Check keyword arguments for marked content (tainted or classified)
        for key, value in kwargs.items():
            if isinstance(value, CBlock) and hasattr(value, '_meta') and '_security' in value._meta:
                security_meta = value._meta['_security']
                if isinstance(security_meta, SecurityMetadata):
                    if security_meta.is_tainted():
                        taint_source = security_meta.get_taint_source()
                        source_info = f" (tainted by: {type(taint_source).__name__})" if taint_source else ""
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"tainted content in argument '{key}'{source_info}"
                        )
                    elif security_meta.is_classified():
                        raise SecurityError(
                            f"Function {func.__name__} requires safe input, but received "
                            f"classified content in argument '{key}'"
                        )
        
        return func(*args, **kwargs)
    
    return wrapper  # type: ignore


def declassify(cblock: CBlock) -> CBlock:
    """Create a declassified version of a CBlock (non-mutating).
    
    This function creates a new CBlock with the same content but marked
    as safe (SecLevel.none()). The original CBlock is not modified.
    
    Args:
        cblock: The CBlock to declassify
        
    Returns:
        A new CBlock with safe security level
    """
    # Create new meta dict with safe security
    new_meta = cblock._meta.copy() if cblock._meta else {}
    new_meta['_security'] = SecurityMetadata(SecLevel.none())
    
    # Return new CBlock with same content but new security metadata
    return CBlock(cblock.value, new_meta)
