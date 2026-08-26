# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Ollama flaky-rerun patterns in `test/conftest.py`.

The conftest applies `pytest.mark.flaky(only_rerun=OLLAMA_TIMEOUT_RERUN_PATTERNS)`
to every `ollama`-marked test so that transient Ollama stalls are retried while
real failures are not. pytest-rerunfailures matches each pattern with
`re.search` against `f"{excinfo.type.__name__}: {excinfo.value}"`.

These tests pin that match behaviour against the exception strings each
backend path actually produces, so the retry net cannot silently stop covering
a path (as it did for the LiteLLM path, whose `litellm.exceptions.Timeout`
message the original single `"ReadTimeout"` pattern never matched), and it
cannot start retrying the pytest-timeout watchdog kill, which has already
consumed the whole per-attempt budget.
"""

import re

from test.conftest import OLLAMA_TIMEOUT_RERUN_PATTERNS

# Match strings captured from the exception shapes each path raises, using the
# same f"{type.__name__}: {value}" construction pytest-rerunfailures uses.

NATIVE_READ_TIMEOUT = "ReadTimeout: "

# Captured from litellm 1.95.0 against a stalled OpenAI-compatible endpoint
# (socket accepted, no response) with a bounded request timeout:
LITELLM_TIMEOUT = (
    "Timeout: litellm.Timeout: APITimeoutError - Request timed out. "
    "Error_str: Request timed out. - timeout value=300.0, time taken=300.12 seconds"
)

# pytest-timeout watchdog kill of an attempt that consumed the 900 s budget.
WATCHDOG_KILL = "Failed: Timeout (>900.0s) from pytest-timeout."


def _matches_any(patterns: list[str], match_string: str) -> bool:
    return any(
        isinstance(pattern, str) and re.search(pattern, match_string)
        for pattern in patterns
    )


def test_native_ollama_readtimeout_is_rerunnable():
    """The native OllamaModelBackend timeout (httpx.ReadTimeout) must rerun."""
    assert _matches_any(OLLAMA_TIMEOUT_RERUN_PATTERNS, NATIVE_READ_TIMEOUT)


def test_litellm_openai_compatible_timeout_is_rerunnable():
    """The LiteLLM path's litellm.Timeout (APITimeoutError message) must rerun."""
    assert _matches_any(OLLAMA_TIMEOUT_RERUN_PATTERNS, LITELLM_TIMEOUT)


def test_pytest_timeout_watchdog_kill_is_not_rerunnable():
    """A watchdog kill already spent the attempt budget; rerunning is pure waste."""
    assert not _matches_any(OLLAMA_TIMEOUT_RERUN_PATTERNS, WATCHDOG_KILL)
