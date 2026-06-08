"""Filename validation utilities for the decompose pipeline.

Provides `validate_filename`, which checks that a candidate output filename
contains only safe characters (alphanumeric, underscores, hyphens, periods, and
spaces) and falls within a reasonable length limit. Used to prevent path-traversal
or shell-injection issues when writing decomposition output files.
"""


def validate_filename(candidate_str: str) -> bool:
    """Check whether a string is safe to use as an output filename.

    Permits alphanumeric characters, underscores, hyphens, periods, and spaces;
    the first character must be alphanumeric, an underscore, or a period.
    Also enforces a maximum length of 250 characters.

    Args:
        candidate_str: The filename candidate to validate.

    Returns:
        `True` if the string is a safe, valid filename; `False` otherwise.
    """
    import re

    # Allows alphanumeric characters, underscore, hyphen, period, and space.
    # Enforces the first character to be alphanumeric, underscore, or period.
    # Anchors ^ and $ ensure the entire string matches the pattern.
    FILENAME_PATTERN = r"^[a-zA-Z0-9_.][a-zA-Z0-9_.\- ]+$"

    # Check if the "filename" matches the pattern and is within a reasonable length
    # (e.g., 1 to 250 characters, a common limit that considers 5 more character for extension)
    if re.fullmatch(FILENAME_PATTERN, candidate_str) and 1 <= len(candidate_str) <= 250:
        return True
    return False
