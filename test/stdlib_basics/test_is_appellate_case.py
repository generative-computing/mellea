import pytest

import os

from mellea.stdlib.reqlib.is_appellate_case import load_jsons_from_folder, get_court_from_case, is_appellate_court

from mellea import start_session
from mellea.stdlib.requirement import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


def test_is_appellate_court():
    assert is_appellate_court("Supreme Court of New Jersey") is True
    assert is_appellate_court("Tax Court of New Jersey") is False
    assert is_appellate_court("Pennsylvania Commonwealth Court") is True
    assert is_appellate_court("U.S. Court of Appeals for the First Circuit") is True
    assert is_appellate_court("Maryland Appellate Court") is True
    assert is_appellate_court("District Court of Maryland") is False


def test_appellate_case_session():
    case_name = "ARTHUR DeMOORS, PLAINTIFF-RESPONDENT, v. ATLANTIC CASUALTY INSURANCE COMPANY OF NEWARK, NEW JERSEY, A CORPORATION, DEFENDANT-APPELLANT"
    folder_path = os.path.join(os.path.dirname(__file__), "..", "data", "legal", "nj_case_metadata")
    folder_path = os.path.normpath(folder_path)

    m = start_session()
    case_metadata = load_jsons_from_folder(folder_path)
    appellate_case = m.instruct(
        f"Return the following string (only return the characters after the colon, no other words): {case_name}",
        requirements=[
            req(
                "The result should be an appellate court case",
                validation_fn=simple_validate(
                    lambda x: is_appellate_court(get_court_from_case(x, case_metadata))
                ),
            )
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        return_sampling_results=True,
    )
    assert "SUCCESS" in appellate_case
