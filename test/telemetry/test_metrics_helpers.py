"""Unit tests for classify_error()."""

from unittest.mock import MagicMock

import pytest

from mellea.telemetry.metrics import (
    ERROR_TYPE_AUTH,
    ERROR_TYPE_CONTENT_POLICY,
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_RATE_LIMIT,
    ERROR_TYPE_SERVER_ERROR,
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_TRANSPORT_ERROR,
    ERROR_TYPE_UNKNOWN,
    classify_error,
)

# ---------------------------------------------------------------------------
# classify_error() — stdlib exceptions
# ---------------------------------------------------------------------------


def test_classify_timeout_error():
    assert classify_error(TimeoutError("timed out")) == ERROR_TYPE_TIMEOUT


def test_classify_connection_error():
    assert classify_error(ConnectionError("refused")) == ERROR_TYPE_TRANSPORT_ERROR


def test_classify_connection_refused_error():
    assert (
        classify_error(ConnectionRefusedError("refused")) == ERROR_TYPE_TRANSPORT_ERROR
    )


def test_classify_unknown_exception():
    assert classify_error(ValueError("something went wrong")) == ERROR_TYPE_UNKNOWN


def test_classify_runtime_error_is_unknown():
    assert classify_error(RuntimeError("unexpected")) == ERROR_TYPE_UNKNOWN


# ---------------------------------------------------------------------------
# classify_error() — name-based heuristics (no openai import needed)
# ---------------------------------------------------------------------------


def test_classify_name_rate_limit():
    class RateLimitError(Exception):
        pass

    assert classify_error(RateLimitError()) == ERROR_TYPE_RATE_LIMIT


def test_classify_name_timeout_in_class_name():
    class RequestTimeoutError(Exception):
        pass

    assert classify_error(RequestTimeoutError()) == ERROR_TYPE_TIMEOUT


def test_classify_name_auth_in_class_name():
    class AuthError(Exception):
        pass

    assert classify_error(AuthError()) == ERROR_TYPE_AUTH


def test_classify_name_server_in_class_name():
    class ServerError(Exception):
        pass

    assert classify_error(ServerError()) == ERROR_TYPE_SERVER_ERROR


def test_classify_name_transport_in_class_name():
    class TransportError(Exception):
        pass

    assert classify_error(TransportError()) == ERROR_TYPE_TRANSPORT_ERROR


def test_classify_name_content_policy_in_class_name():
    class ContentPolicyError(Exception):
        pass

    assert classify_error(ContentPolicyError()) == ERROR_TYPE_CONTENT_POLICY


# ---------------------------------------------------------------------------
# classify_error() — OpenAI SDK exceptions (mocked)
# ---------------------------------------------------------------------------


def _make_openai_mock():
    """Build a minimal openai module mock with the exception classes we test."""
    openai = MagicMock()

    class _Base(Exception):
        pass

    class RateLimitError(_Base):
        pass

    class APITimeoutError(_Base):
        pass

    class AuthenticationError(_Base):
        pass

    class PermissionDeniedError(_Base):
        pass

    class BadRequestError(_Base):
        def __init__(self, msg="", code=None):
            super().__init__(msg)
            self.code = code

    class APIConnectionError(_Base):
        pass

    class InternalServerError(_Base):
        pass

    openai.RateLimitError = RateLimitError
    openai.APITimeoutError = APITimeoutError
    openai.AuthenticationError = AuthenticationError
    openai.PermissionDeniedError = PermissionDeniedError
    openai.BadRequestError = BadRequestError
    openai.APIConnectionError = APIConnectionError
    openai.InternalServerError = InternalServerError

    return openai


@pytest.fixture
def openai_mock(monkeypatch):
    mock = _make_openai_mock()
    monkeypatch.setitem(__import__("sys").modules, "openai", mock)
    return mock


def test_classify_openai_rate_limit(openai_mock):
    assert classify_error(openai_mock.RateLimitError("429")) == ERROR_TYPE_RATE_LIMIT


def test_classify_openai_timeout(openai_mock):
    assert classify_error(openai_mock.APITimeoutError()) == ERROR_TYPE_TIMEOUT


def test_classify_openai_auth(openai_mock):
    assert classify_error(openai_mock.AuthenticationError("401")) == ERROR_TYPE_AUTH


def test_classify_openai_permission_denied(openai_mock):
    assert classify_error(openai_mock.PermissionDeniedError("403")) == ERROR_TYPE_AUTH


def test_classify_openai_bad_request_invalid(openai_mock):
    assert (
        classify_error(openai_mock.BadRequestError("400", code=None))
        == ERROR_TYPE_INVALID_REQUEST
    )


def test_classify_openai_bad_request_content_policy(openai_mock):
    assert (
        classify_error(
            openai_mock.BadRequestError("400", code="content_policy_violation")
        )
        == ERROR_TYPE_CONTENT_POLICY
    )


def test_classify_openai_connection_error(openai_mock):
    assert (
        classify_error(openai_mock.APIConnectionError()) == ERROR_TYPE_TRANSPORT_ERROR
    )


def test_classify_openai_internal_server_error(openai_mock):
    assert (
        classify_error(openai_mock.InternalServerError("500"))
        == ERROR_TYPE_SERVER_ERROR
    )
