import pytest

from citation_exists import normalize_case_name, citation_exists

# Mock context for testing citation_exists (not sure if this is implemented correctly)
class MockContext:
    def __init__(self, case_name):
        self._case_name = case_name

    def last_output(self):
        return type("MockOutput", (), {"value": self._case_name})()

# region: normalize_case_name tests

# NOTE: investigate how to handle apostrophe and ampersand
@pytest.mark.parametrize("raw_name,expected", [
    ("BOB VS SHMEEGUS", "bob v. shmeegus"),
    ("William Payne, Executor of John Payne v. William Dudley Executor of Fleet", "william payne executor of john payne v. william dudley executor of fleet"),
    ("Ozwald v. Dickinson's Ex'rs", "ozwald v. dickinson's ex'rs"),
    ("Fox & al. v. Cosby", "fox & al. v. cosby"),
    ("Groves v. Graves", "groves v. graves"),
    ("Ozwald, Deniston, & Co. v. Dickinson's Ex'rs", "ozwald deniston & co. v. dickinson's ex'rs"),
    ("Bobby- versus shmeegy", "bobby v. shmeegy")
])

def test_normalize_case_name(raw_name, expected):
    assert normalize_case_name(raw_name) == expected

# endregion

# region: citation_exists tests

@pytest.mark.parametrize("case_name,expected", [
    ("Bob v. Shmeegus", False),
    ("Gimli versus Legolas", False),
    ("Groves v. Graves", True),
    ("William Payne, Executor of John Payne v. William Dudley Executor of Fleet", True),
    ("Payne v. Dudley", True),
    ("Fox & al. v. Cosby", True),
    ("Fox v. Cosby", True),
])

def test_citation_exists(tmp_path, case_name, expected):
    # create mock context
    ctx = MockContext(case_name)

    # path to metadata folder
    db_folder = "cases_metadata"
    

    result = citation_exists(ctx, db_folder)
    assert result.as_bool() == expected, result.reason

# endregion