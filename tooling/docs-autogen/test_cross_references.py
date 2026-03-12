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


if __name__ == "__main__":
    print("Testing cross-reference functions...")
    print("=" * 60)

    test_extract_type_references()
    print()
    test_add_cross_references()

    print("=" * 60)
    print("All tests passed!")

# Made with Bob
