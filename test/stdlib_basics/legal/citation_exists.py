from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult

import json
import os
import re

def normalize_case_name(name) -> str:
    """
    Converts a case name to a standard format.

    Args:
        name: A string representing the case name.

    Returns:
        A normalized case name.
    """
    # 1. Lowercase everything
    name = name.lower()

    # 2. Normalize 'vs', 'vs.', 'v', 'versus' to 'v.'
    name = re.sub(r'\b(vs\.?|versus|v)(?!\.)\b', 'v.', name)

    # 3. Remove all non-alphanumeric characters except periods, spaces, and apostrophes
    name = re.sub(r"[^a-z0-9.& ']+", '', name)

    # 4. Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name)

    return name.strip()

def citation_exists(ctx: Context, folder_path: str) -> ValidationResult:
    """
    Given a case name and a directory, checks all of the CasesMetadata files in that directory
    to see if the given case name can be found in it.

    Args:
        ctx: Context
        folder_path: a string representing the path to the database directory 

    Returns:
        A validation result  indicating if a match was found between given case name and database
    """
    # not sure about this line
    case_name = ctx.last_output().value

    if not case_name or not isinstance(case_name, str):
        return ValidationResult(False, reason="No case name provided in output")
    
    normalized_input = normalize_case_name(case_name)

    # Search in all files in the folder
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        return ValidationResult(False, reason=f"Folder '{folder_path}' not found")

    case_names = set()
    case_name_abb = set()

    # Iterate over all json files in the directory and extract info
    for file in files:
        if not file.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder_path, file), 'r') as f:
                data = json.load(f)
        
            # Collect all case names (including abbreviated version)
            for case in data:
                if 'name' in case:
                    case_names.add(normalize_case_name(case['name']))
                if 'name_abbreviation' in case:
                    case_name_abb.add(normalize_case_name(case['name_abbreviation']))
        
        except Exception as e:
            return ValidationResult(False, reason=f"Error loading '{file}': {e!s}")

        # Check both name and name_abbreviation
    if normalized_input in case_names or normalized_input in case_name_abb:
        return ValidationResult(True, reason=f"'{case_name}' found in database")
    else:
        return ValidationResult(False, reason=f"'{case_name}' not found in database")


class CaseNameExistsInDatabase(Requirement):
    """
    Checks if the output case name exists in the provided case metadata database.
    """

    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        super().__init__(
            description="The case name should exist in the provided case metadata database.",
            validation_fn=lambda ctx: citation_exists(ctx, self._folder_path),
        )