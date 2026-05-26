"""Regression tests for docs/examples/ collection hooks.

These hooks have regressed twice (#794, #796). This test ensures:
- Support files (__init__.py, helpers.py, conftest.py) are never collected
- Real examples with markers ARE collected
- No example is collected twice (duplicate guard)
"""

import subprocess


def test_example_collection_sanity():
    """Verify example collection excludes support files and avoids duplicates."""
    result = subprocess.run(
        ["uv", "run", "pytest", "docs/examples/", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    lines = result.stdout.splitlines()
    # Collected test IDs are lines before the blank/summary lines
    collected = [line for line in lines if "::" in line]

    # Support files must never appear as collected tests
    for item in collected:
        filename = item.split("::")[0].rsplit("/", 1)[-1]
        assert filename != "__init__.py", f"__init__.py collected as test: {item}"
        assert filename != "helpers.py", f"helpers.py collected as test: {item}"
        assert filename != "conftest.py", f"conftest.py collected as test: {item}"

    # Sanity floor — we have ~79 examples today; 50 is a safe lower bound
    assert len(collected) >= 50, (
        f"Only {len(collected)} examples collected — expected at least 50. "
        "Collection hooks may be broken."
    )

    # No duplicates — each test ID should appear exactly once
    seen = set()
    for item in collected:
        assert item not in seen, f"Duplicate collection detected: {item}"
        seen.add(item)
