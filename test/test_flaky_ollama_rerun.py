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
from pathlib import Path

from test.conftest import OLLAMA_TIMEOUT_RERUN_PATTERNS

# The live "say hello" telemetry tests whose timeout shapes are pinned here.
METRICS_TEST_PATH = Path(__file__).parent / "telemetry" / "test_metrics_backend.py"

# Match strings captured from the exception shapes each path raises, using the
# same f"{type.__name__}: {value}" construction pytest-rerunfailures uses.

NATIVE_READ_TIMEOUT = "ReadTimeout: "

# Captured from litellm 1.95.0 against a stalled OpenAI-compatible endpoint
# (socket accepted, no response) with a bounded request timeout:
LITELLM_TIMEOUT = (
    "Timeout: litellm.Timeout: APITimeoutError - Request timed out. "
    "Error_str: Request timed out. - timeout value=300.0, time taken=300.12 seconds"
)

# The streaming stream-guard abort (mellea/helpers/async_helpers.py,
# DEFAULT_CHUNK_TIMEOUT=120.0): the builtin TimeoutError is raised verbatim at
# the consumer (mellea/core/base.py) when a stalled stream goes quiet for 120 s.
STREAM_GUARD_TIMEOUT = (
    "TimeoutError: Stream timed out after 120.0s without a chunk "
    "(covers time-to-first-token and inter-chunk gaps). "
    "Set ModelOption.STREAM_TIMEOUT to a larger value or None to disable."
)

# Raised by asyncio.wait_for when the test-level 300 s total-stream budget
# (test/telemetry/test_metrics_backend.py) is exhausted: no message.
STREAM_WAIT_FOR_TIMEOUT = "TimeoutError: "

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


def test_streaming_stream_guard_timeout_is_rerunnable():
    """A stalled stream aborted by the 120 s chunk guard must rerun."""
    assert _matches_any(OLLAMA_TIMEOUT_RERUN_PATTERNS, STREAM_GUARD_TIMEOUT)


def test_streaming_total_budget_timeout_is_rerunnable():
    """A stalled stream that exhausts the 300 s wait_for budget must rerun.

    asyncio.wait_for raises the builtin TimeoutError with an empty message;
    the per-chunk guard only bounds inter-chunk gaps, so a stream that keeps
    the connection alive but never finishes needs this total-time backstop.
    """
    assert _matches_any(OLLAMA_TIMEOUT_RERUN_PATTERNS, STREAM_WAIT_FOR_TIMEOUT)


def test_pytest_timeout_watchdog_kill_is_not_rerunnable():
    """A watchdog kill already spent the attempt budget; rerunning is pure waste."""
    assert not _matches_any(OLLAMA_TIMEOUT_RERUN_PATTERNS, WATCHDOG_KILL)


def test_live_openai_compat_tests_send_the_max_tokens_cap():
    """The /v1 live tests must cap output with the raw `max_tokens` key.

    ModelOption.MAX_NEW_TOKENS maps to `max_completion_tokens` on the
    OpenAI/LiteLLM paths, but the CI-pinned Ollama 0.32.2 /v1 handler only
    maps `max_tokens` -> num_predict (openai/openai.go at v0.32.2) and
    silently ignores max_completion_tokens. With the sentinel, the
    generation ran uncapped on CI for 15 m (run 33048969379, 3.12 lane);
    local Ollama 0.33.0 accepts both, which is why the sentinel looked
    fine locally.
    """
    src = METRICS_TEST_PATH.read_text()
    assert src.count('"max_tokens": 64') == 2  # openai + litellm live tests


def test_live_tests_bound_the_avalue_consumption():
    """All four live consumption paths must stay inside the 300 s budget.

    astream() returns as soon as its queue drains, so a long stream is
    finished by avalue(); without the bound it rides to the 900 s pytest
    watchdog (run 33048969379: a 15 m stream escaped the astream bound).
    """
    src = METRICS_TEST_PATH.read_text()
    assert src.count("asyncio.wait_for(mot.avalue(), timeout=300.0)") == 4
