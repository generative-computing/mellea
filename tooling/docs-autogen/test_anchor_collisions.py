#!/usr/bin/env python3
"""Test anchor collision detection."""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from validate import validate_anchor_collisions


def test_no_collisions():
    """Test file with no anchor collisions."""
    content = """---
title: Test Module
---

## class Backend

This is the Backend class.

## function generative

This is the generative function.

### Backend.generate

Method for generating.
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.mdx"
        test_file.write_text(content)

        error_count, errors = validate_anchor_collisions(Path(tmpdir))

        assert error_count == 0, f"Expected no errors, got {error_count}: {errors}"
        print("✅ No collisions test passed")


def test_with_collisions():
    """Test file with anchor collisions."""
    content = """---
title: Test Module
---

## class Backend

This is the Backend class.

## Class Backend

This is a duplicate heading that will cause a collision.
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.mdx"
        test_file.write_text(content)

        error_count, errors = validate_anchor_collisions(Path(tmpdir))

        assert error_count > 0, "Expected collision errors"
        assert "class-backend" in errors[0].lower(), (
            "Should detect 'class-backend' collision"
        )
        print("✅ Collision detection test passed")


if __name__ == "__main__":
    print("Testing anchor collision detection...")
    print("=" * 60)

    test_no_collisions()
    print()
    test_with_collisions()

    print("=" * 60)
    print("All tests passed!")

# Made with Bob
