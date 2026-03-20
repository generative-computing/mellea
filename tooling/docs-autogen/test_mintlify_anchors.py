#!/usr/bin/env python3
"""Test Mintlify anchor generation algorithm.

Mintlify generates anchors from headings for internal linking.
This script tests the algorithm to ensure our cross-references work correctly.
"""


def mintlify_anchor(heading: str) -> str:
    """Generate Mintlify-style anchor from heading.

    Based on observed behavior:
    - Lowercase
    - Replace spaces with hyphens
    - Remove special characters except hyphens
    - Remove leading/trailing hyphens

    Args:
        heading: Heading text (e.g., "class Backend")

    Returns:
        Anchor string (e.g., "class-backend")
    """
    import re

    # Lowercase
    anchor = heading.lower()

    # Replace spaces with hyphens
    anchor = anchor.replace(" ", "-")

    # Remove special characters except hyphens and alphanumeric
    anchor = re.sub(r"[^a-z0-9-]", "", anchor)

    # Remove multiple consecutive hyphens
    anchor = re.sub(r"-+", "-", anchor)

    # Remove leading/trailing hyphens
    anchor = anchor.strip("-")

    return anchor


def test_mintlify_anchors():
    """Test anchor generation with known examples."""
    test_cases = [
        ("class Backend", "class-backend"),
        ("function generative", "function-generative"),
        ("Backend.__init__", "backendinit"),
        ("@generative decorator", "generative-decorator"),
        ("Session.add_message()", "sessionaddmessage"),
        ("Type[Backend]", "typebackend"),
    ]

    print("Testing Mintlify anchor generation:")
    print("=" * 60)

    all_passed = True
    for heading, expected in test_cases:
        result = mintlify_anchor(heading)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        print(f"{status} '{heading}' -> '{result}' (expected: '{expected}')")

    print("=" * 60)
    print(f"Result: {'All tests passed!' if all_passed else 'Some tests failed'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(test_mintlify_anchors())
