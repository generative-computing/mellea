import json
import os

from mellea.stdlib.requirement import Requirement


def load_jsons_from_folder(folder_path: str) -> list[dict]:
    """Load all JSON files in the folder into a list of dicts."""
    all_entries = []
    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name)) as file:
            data = json.load(file)
            all_entries.extend(data)
    return all_entries


def get_court_from_case(case_name: str, case_metadata: list[dict]) -> str:
    """Given a case name and case metadata, return the court name."""
    for entry in case_metadata:
        if case_name.lower() in entry["name"].lower():
            return entry["court"]["name"]
    raise ValueError(f"Court not found for case: {case_name}")


def is_appellate_court(court_name: str) -> bool:
    """Determine if a court is an appellate court based on its name."""
    lowered_name = court_name.lower()
    exceptions = ["pennsylvania superior court", "pennsylvania commonwealth court"]
    keywords = ["supreme", "appeal", "appellate"]
    return (
        any(keyword in lowered_name for keyword in keywords)
        or lowered_name in exceptions
    )


class IsAppellateCase(Requirement):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        super().__init__(
            description="The result should be an appellate court case.",
            validation_fn=lambda x: is_appellate_court(
                get_court_from_case(x, folder_path)
            ),
        )


class IsDistrictCase(Requirement):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        super().__init__(
            description="The result should be a district court case.",
            validation_fn=lambda x: not is_appellate_court(
                get_court_from_case(x, folder_path)
            ),
        )
