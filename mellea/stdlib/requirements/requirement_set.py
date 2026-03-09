"""RequirementSet: A composable collection of requirements (guardrails).

This module provides utilities for managing, combining, and reusing multiple
requirements as a cohesive unit, making it easier to maintain consistent
guardrail policies across an application.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

from ...core import Requirement


class RequirementSet:
    """A composable collection of requirements (guardrails).

    Provides a fluent API for building, combining, and managing
    multiple requirements as a reusable unit. RequirementSet instances
    are iterable and can be used anywhere a list of requirements is expected.

    Examples:
        Basic usage:
        >>> from mellea.stdlib.requirements import RequirementSet
        >>> from mellea.stdlib.requirements.guardrails import no_pii, json_valid
        >>>
        >>> basic_safety = RequirementSet([no_pii(), no_harmful_content()])
        >>> result = m.instruct("Generate text", requirements=basic_safety)

        Fluent API:
        >>> reqs = (RequirementSet()
        ...     .add(no_pii())
        ...     .add(json_valid())
        ...     .add(max_length(500)))

        Composition:
        >>> safety = RequirementSet([no_pii(), no_harmful_content()])
        >>> format = RequirementSet([json_valid(), max_length(500)])
        >>> combined = safety + format

        In-place modification:
        >>> reqs = RequirementSet([no_pii()])
        >>> reqs += RequirementSet([json_valid()])
    """

    def __init__(self, requirements: list[Requirement] | None = None):
        """Initialize RequirementSet with optional list of requirements.

        Args:
            requirements: Optional list of Requirement instances

        Raises:
            TypeError: If any item in requirements is not a Requirement instance
        """
        self._requirements: list[Requirement] = []
        if requirements:
            for req in requirements:
                if not isinstance(req, Requirement):
                    raise TypeError(
                        f"All items must be Requirement instances, got {type(req).__name__}"
                    )
                self._requirements.append(req)

    def add(self, requirement: Requirement) -> RequirementSet:
        """Add a requirement and return a new RequirementSet (fluent API).

        This method returns a new RequirementSet instance, leaving the original
        unchanged (functional/immutable style).

        Args:
            requirement: Requirement instance to add

        Returns:
            New RequirementSet with the added requirement

        Raises:
            TypeError: If requirement is not a Requirement instance

        Examples:
            >>> reqs = RequirementSet().add(no_pii()).add(json_valid())
        """
        if not isinstance(requirement, Requirement):
            raise TypeError(
                f"Expected Requirement instance, got {type(requirement).__name__}"
            )
        new_set = self.copy()
        new_set._requirements.append(requirement)
        return new_set

    def remove(self, requirement: Requirement) -> RequirementSet:
        """Remove a requirement and return a new RequirementSet (fluent API).

        This method returns a new RequirementSet instance, leaving the original
        unchanged. If the requirement is not found, returns a copy unchanged.

        Args:
            requirement: Requirement instance to remove

        Returns:
            New RequirementSet without the specified requirement

        Examples:
            >>> reqs = RequirementSet([no_pii(), json_valid()])
            >>> reqs_without_pii = reqs.remove(no_pii())
        """
        new_set = self.copy()
        try:
            new_set._requirements.remove(requirement)
        except ValueError:
            pass  # Requirement not found, return copy unchanged
        return new_set

    def extend(self, requirements: list[Requirement]) -> RequirementSet:
        """Add multiple requirements and return a new RequirementSet (fluent API).

        Args:
            requirements: List of Requirement instances to add

        Returns:
            New RequirementSet with all requirements added

        Raises:
            TypeError: If any item is not a Requirement instance

        Examples:
            >>> reqs = RequirementSet().extend([no_pii(), json_valid(), max_length(500)])
        """
        new_set = self.copy()
        for req in requirements:
            if not isinstance(req, Requirement):
                raise TypeError(
                    f"All items must be Requirement instances, got {type(req).__name__}"
                )
            new_set._requirements.append(req)
        return new_set

    def __add__(self, other: RequirementSet) -> RequirementSet:
        """Combine two RequirementSets using + operator.

        Creates a new RequirementSet containing requirements from both sets.

        Args:
            other: Another RequirementSet to combine with

        Returns:
            New RequirementSet containing requirements from both sets

        Raises:
            TypeError: If other is not a RequirementSet

        Examples:
            >>> safety = RequirementSet([no_pii()])
            >>> format = RequirementSet([json_valid()])
            >>> combined = safety + format
        """
        if not isinstance(other, RequirementSet):
            raise TypeError(
                f"Can only add RequirementSet to RequirementSet, got {type(other).__name__}"
            )
        new_set = self.copy()
        new_set._requirements.extend(other._requirements)
        return new_set

    def __iadd__(self, other: RequirementSet) -> RequirementSet:
        """In-place addition using += operator.

        Modifies the current RequirementSet by adding requirements from other.

        Args:
            other: Another RequirementSet to add

        Returns:
            Self (modified in place)

        Raises:
            TypeError: If other is not a RequirementSet

        Examples:
            >>> reqs = RequirementSet([no_pii()])
            >>> reqs += RequirementSet([json_valid()])
        """
        if not isinstance(other, RequirementSet):
            raise TypeError(
                f"Can only add RequirementSet to RequirementSet, got {type(other).__name__}"
            )
        self._requirements.extend(other._requirements)
        return self

    def __len__(self) -> int:
        """Return the number of requirements in the set.

        Returns:
            Number of requirements

        Examples:
            >>> reqs = RequirementSet([no_pii(), json_valid()])
            >>> len(reqs)
            2
        """
        return len(self._requirements)

    def __iter__(self) -> Iterator[Requirement]:
        """Make RequirementSet iterable.

        This allows RequirementSet to be used anywhere a list of requirements
        is expected, such as in m.instruct(requirements=...).

        Returns:
            Iterator over requirements

        Examples:
            >>> reqs = RequirementSet([no_pii(), json_valid()])
            >>> for req in reqs:
            ...     print(req.description)
        """
        return iter(self._requirements)

    def __repr__(self) -> str:
        """Return string representation of RequirementSet.

        Returns:
            String showing the number of requirements

        Examples:
            >>> reqs = RequirementSet([no_pii(), json_valid()])
            >>> repr(reqs)
            'RequirementSet(2 requirements)'
        """
        return f"RequirementSet({len(self._requirements)} requirements)"

    def __str__(self) -> str:
        """Return detailed string representation.

        Returns:
            String listing all requirement descriptions
        """
        if not self._requirements:
            return "RequirementSet(empty)"

        descriptions = [
            req.description or "No description" for req in self._requirements
        ]
        return (
            f"RequirementSet({len(self._requirements)} requirements):\n  - "
            + "\n  - ".join(descriptions)
        )

    def copy(self) -> RequirementSet:
        """Create a deep copy of the RequirementSet.

        Returns:
            New RequirementSet with copied requirements

        Examples:
            >>> original = RequirementSet([no_pii()])
            >>> copy = original.copy()
            >>> copy.add(json_valid())  # Doesn't affect original
        """
        new_set = RequirementSet()
        new_set._requirements = deepcopy(self._requirements)
        return new_set

    def to_list(self) -> list[Requirement]:
        """Convert to a plain list of requirements.

        Returns:
            List of Requirement instances

        Examples:
            >>> reqs = RequirementSet([no_pii(), json_valid()])
            >>> req_list = reqs.to_list()
        """
        return list(self._requirements)

    def clear(self) -> RequirementSet:
        """Remove all requirements and return a new empty RequirementSet.

        Returns:
            New empty RequirementSet

        Examples:
            >>> reqs = RequirementSet([no_pii(), json_valid()])
            >>> empty = reqs.clear()
            >>> len(empty)
            0
        """
        return RequirementSet()

    def is_empty(self) -> bool:
        """Check if the RequirementSet is empty.

        Returns:
            True if no requirements, False otherwise

        Examples:
            >>> reqs = RequirementSet()
            >>> reqs.is_empty()
            True
        """
        return len(self._requirements) == 0
