def validate_filename(candidate_str: str) -> bool:
    import re

    # Pattern allows alphanumeric characters, underscore, hyphen, period, and space.
    # Anchors ^ and $ ensure the entire string matches the pattern.
    FILENAME_PATTERN = r"^[a-zA-Z0-9_.][a-zA-Z0-9_.\- ]+$"

    # Check if the "filename" matches the pattern and is within a reasonable length
    # (e.g., 1 to 250 characters, a common limit that considers 5 more character for extension)
    if re.fullmatch(FILENAME_PATTERN, candidate_str) and 1 <= len(candidate_str) <= 250:
        return True
    return False
