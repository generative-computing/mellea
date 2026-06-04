# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mellea.formatters.granite.retrievers.util._urlretrieve_with_retry."""

import http.client
import urllib.error
from unittest.mock import call, patch

import pytest

pytest.importorskip(
    "pyarrow", reason="pyarrow not installed — install mellea[granite_retriever]"
)

from mellea.formatters.granite.retrievers.util import (
    _urlretrieve_with_retry,  # type: ignore[reportAttributeAccessIssue]
)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://example.com",
        code=code,
        msg=str(code),
        hdrs=http.client.HTTPMessage(),
        fp=None,
    )


def test_success_first_attempt():
    with (
        patch("urllib.request.urlretrieve") as mock_retrieve,
        patch("time.sleep") as mock_sleep,
    ):
        _urlretrieve_with_retry("http://example.com/f", "/tmp/f")

    mock_retrieve.assert_called_once_with("http://example.com/f", "/tmp/f")
    mock_sleep.assert_not_called()


def test_retries_on_429_succeeds_second_attempt():
    with (
        patch("urllib.request.urlretrieve") as mock_retrieve,
        patch("time.sleep") as mock_sleep,
    ):
        mock_retrieve.side_effect = [_http_error(429), None]
        _urlretrieve_with_retry("http://example.com/f", "/tmp/f")

    assert mock_retrieve.call_count == 2
    mock_sleep.assert_called_once_with(2)  # 2**1


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_retries_on_5xx(code: int):
    with (
        patch("urllib.request.urlretrieve") as mock_retrieve,
        patch("time.sleep") as mock_sleep,
    ):
        mock_retrieve.side_effect = [_http_error(code), None]
        _urlretrieve_with_retry("http://example.com/f", "/tmp/f")

    assert mock_retrieve.call_count == 2
    mock_sleep.assert_called_once_with(2)


def test_non_retryable_raises_immediately():
    with (
        patch("urllib.request.urlretrieve") as mock_retrieve,
        patch("time.sleep") as mock_sleep,
    ):
        mock_retrieve.side_effect = _http_error(404)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _urlretrieve_with_retry("http://example.com/f", "/tmp/f")

    assert exc_info.value.code == 404
    mock_retrieve.assert_called_once()
    mock_sleep.assert_not_called()


def test_raises_after_all_retries_exhausted():
    with (
        patch("urllib.request.urlretrieve") as mock_retrieve,
        patch("time.sleep") as mock_sleep,
    ):
        mock_retrieve.side_effect = _http_error(429)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _urlretrieve_with_retry("http://example.com/f", "/tmp/f")

    assert exc_info.value.code == 429
    assert mock_retrieve.call_count == 3  # default max_attempts=3
    assert mock_sleep.call_args_list == [call(2), call(4)]  # 2**1, 2**2


def test_max_attempts_one_no_retry():
    with (
        patch("urllib.request.urlretrieve") as mock_retrieve,
        patch("time.sleep") as mock_sleep,
    ):
        mock_retrieve.side_effect = _http_error(429)
        with pytest.raises(urllib.error.HTTPError):
            _urlretrieve_with_retry("http://example.com/f", "/tmp/f", max_attempts=1)

    mock_retrieve.assert_called_once()
    mock_sleep.assert_not_called()
