#!/usr/bin/env python3
"""Test cross-reference functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from decorate_api_mdx import add_cross_references, extract_type_references


def test_extract_type_references():
    """Test type reference extraction."""
    content = """
    # Example Module

    This module uses `Backend` and `Session` classes.

    ```python
    def foo(backend: Backend) -> Session:
        return Session(backend)
    ```

    You can also use Optional[Backend] or List[Session].
    """

    refs = extract_type_references(content)
    print("Extracted references:", refs)

    # Should find Backend and Session
    assert "Backend" in refs, "Should find Backend"
    assert "Session" in refs, "Should find Session"

    print("✅ extract_type_references test passed")


def test_add_cross_references():
    """Test cross-reference link generation."""
    content = """
    # Example Module

    This module uses `Backend` for LLM calls.
    """

    # Mock source directory (won't actually resolve, but tests the logic)
    source_dir = Path.cwd() / "mellea"
    module_path = "mellea.stdlib.session"

    # This will run but won't find symbols (that's OK for this test)
    result = add_cross_references(content, module_path, source_dir)

    print("Original content:")
    print(content)
    print("\nProcessed content:")
    print(result)

    print("✅ add_cross_references test passed (no errors)")


def test_directory_index_same_dir_link():
    """Cross-refs from a directory-index file prefix the dir name."""
    # backends/backends.mdx is served at /api/mellea/backends (not /backends/backends).
    # A link to the sibling backends/backend.mdx must include the dir name so it
    # resolves from the parent URL base /api/mellea/.
    symbol_cache = {"FormatterBackend": "mellea.backends.backend"}
    content = "Uses `FormatterBackend` as the base."
    result = add_cross_references(
        content, "mellea.backends.backends", Path("."), symbol_cache=symbol_cache
    )
    assert "backends/backend#class-formatterbackend" in result, (
        f"Expected 'backends/backend#class-formatterbackend' in: {result}"
    )
    print("✅ directory-index same-dir link test passed")


def test_regular_file_same_dir_link():
    """Cross-refs from a regular file in the same directory use bare filename."""
    symbol_cache = {"FormatterBackend": "mellea.backends.backend"}
    content = "Uses `FormatterBackend` as the base."
    result = add_cross_references(
        content, "mellea.backends.model_ids", Path("."), symbol_cache=symbol_cache
    )
    # model_ids.mdx is at /api/mellea/backends/model_ids — same dir as backend.mdx
    assert "backend#class-formatterbackend" in result, (
        f"Expected 'backend#class-formatterbackend' in: {result}"
    )
    # Must NOT have the directory prefix (that would add an extra level)
    assert "backends/backend#" not in result, (
        f"Should not have 'backends/backend#' in: {result}"
    )
    print("✅ regular-file same-dir link test passed")


def test_directory_index_cross_package_link():
    """Cross-refs from a directory-index file to a different package adjust levels."""
    # core/core.mdx is at /api/mellea/core (dir-index).
    # Link to stdlib/session.mdx should be 'stdlib/session' (not '../stdlib/session').
    symbol_cache = {"Session": "mellea.stdlib.session"}
    content = "Uses `Session`."
    result = add_cross_references(
        content, "mellea.core.core", Path("."), symbol_cache=symbol_cache
    )
    assert "stdlib/session#class-session" in result, (
        f"Expected 'stdlib/session#class-session' in: {result}"
    )
    assert "../stdlib/session" not in result, (
        f"Should not have '../stdlib/session' in: {result}"
    )
    print("✅ directory-index cross-package link test passed")


if __name__ == "__main__":
    print("Testing cross-reference functions...")
    print("=" * 60)

    test_extract_type_references()
    print()
    test_add_cross_references()
    print()
    test_directory_index_same_dir_link()
    print()
    test_regular_file_same_dir_link()
    print()
    test_directory_index_cross_package_link()

    print("=" * 60)
    print("All tests passed!")
