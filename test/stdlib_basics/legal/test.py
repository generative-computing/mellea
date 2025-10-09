import pytest

from citation_exists import normalize_case_name, citation_exists  # Replace `your_module` with actual module name

# Mock context for testing citation_exists
# NOTE: Not sure about how to test without context (can I make some fake context?)
class MockContext:
    def __init__(self, existing_cases):
        self.existing_cases = set(existing_cases)

    def case_exists(self, case_name, table_name):
        return case_name in self.existing_cases

# region: normalize_case_name tests

# NOTE: investigate apostrophe and ampersand business
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

@pytest.fixture
def mock_ctx():
    return MockContext(existing_cases={
        "Groves v. Graves",
        "William Payne, Executor of John Payne v. William Dudley Executor of Fleet",
        "Payne v. Dudley",
        "Fox & al. v. Cosby",
        "Fox v. Cosby",
    })

@pytest.mark.parametrize("case_name,expected", [
    # Not in database
    ("Bob v. Shmeegus", False),
    # In database (VA Reports, Vol 1)
    ("Groves v. Graves", True),
    ("William Payne, Executor of John Payne v. William Dudley Executor of Fleet", True),
    ("Payne v. Dudley", True),
    # In database (VA Reports, Vol 6)
    ("Fox & al. v. Cosby", True),
    ("Fox v. Cosby", True),
])
def test_citation_exists(mock_ctx, case_name, expected):
    assert citation_exists(mock_ctx, case_name, "cases_metadata") == expected

# endregion