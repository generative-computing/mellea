import os

import pytest

@pytest.fixture(scope="session")
def gh_run() -> int:
    return int(os.environ.get("CICD", 0))  # type: ignore


def pytest_runtest_setup(item):
    # Runs tests *not* marked with `@pytest.mark.qualitative` to run normally.
    if not item.get_closest_marker("qualitative"):
        return

    gh_run = int(os.environ.get("CICD", 0))

    if gh_run == 1:
        pytest.xfail(
            reason="Skipping qualitative test: got env variable CICD == 1. Used only in gh workflows."
        )
