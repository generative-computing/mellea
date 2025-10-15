import json
import os

from mellea.stdlib.requirement import Requirement


def get_court_from_case(case_name, folder_path):
    files = os.listdir(folder_path)
    for file_path in files:
        with open(os.path.join(folder_path, file_path)) as file:
            data = json.load(file)
            for entry in data:
                if case_name.lower() in entry["name"].lower():
                    return entry["court"]["name"]


def is_appellate_court(court_name):
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
