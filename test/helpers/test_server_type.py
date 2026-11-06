# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from mellea.helpers import (
    is_vllm_server_with_structured_output,
    vllm_server_version,
    vllm_tokenize_single_tokens,
)
from mellea.helpers.server_type import _server_type, _ServerType

# --- _server_type ---


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://localhost:8000/v1", _ServerType.LOCALHOST),
        ("http://127.0.0.1:11434", _ServerType.LOCALHOST),
        ("http://[::1]:8080/v1", _ServerType.LOCALHOST),
        ("http://0.0.0.0:5000", _ServerType.LOCALHOST),
        ("https://api.openai.com/v1", _ServerType.OPENAI),
        ("https://my-company.example.com/v1", _ServerType.UNKNOWN),
        ("not-a-url", _ServerType.UNKNOWN),
    ],
)
def test_server_type_classification(url, expected):
    assert _server_type(url) == expected


# --- is_vllm_server_with_structured_output ---

BASE_URL = "http://localhost:8000/v1"
HEADERS = {"Authorization": "Bearer test-key"}


def _mock_version_response(version_string, status_code=200):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = 200 <= status_code < 400
    mock_resp.status_code = status_code
    mock_resp.text = json.dumps({"version": version_string})
    return mock_resp


@patch("mellea.helpers.server_type.requests.get")
def test_version_above_threshold(mock_get):
    mock_get.return_value = _mock_version_response("0.16.0")
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is True
    mock_get.assert_called_once_with(
        "http://localhost:8000/version",
        headers=HEADERS,
        timeout=5,
        allow_redirects=False,
    )


@patch("mellea.helpers.server_type.requests.get")
def test_version_equal_to_threshold(mock_get):
    mock_get.return_value = _mock_version_response("0.12.0")
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is True


@patch("mellea.helpers.server_type.requests.get")
def test_version_below_threshold(mock_get):
    mock_get.return_value = _mock_version_response("0.11.9")
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_version_much_lower(mock_get):
    mock_get.return_value = _mock_version_response("0.6.0")
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_release_candidate_version_above(mock_get):
    mock_get.return_value = _mock_version_response("0.16.0rc1.dev172+gddf69")
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is True


@patch("mellea.helpers.server_type.requests.get")
def test_http_error_response(mock_get):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_get.return_value = mock_resp
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_http_404_response(mock_get):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = False
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_no_version_key_in_response(mock_get):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = True
    mock_resp.text = json.dumps({"status": "ok"})
    mock_get.return_value = mock_resp
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_empty_json_response(mock_get):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = True
    mock_resp.text = json.dumps({})
    mock_get.return_value = mock_resp
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_invalid_json_response(mock_get):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = True
    mock_resp.text = "not json at all"
    mock_get.return_value = mock_resp
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_invalid_version_string(mock_get):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = True
    mock_resp.text = json.dumps({"version": "not-a-version"})
    mock_get.return_value = mock_resp
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_connection_error(mock_get):
    mock_get.side_effect = requests.ConnectionError("Connection refused")
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_timeout_error(mock_get):
    mock_get.side_effect = requests.Timeout("Request timed out")
    assert is_vllm_server_with_structured_output(BASE_URL, HEADERS) is False


@patch("mellea.helpers.server_type.requests.get")
def test_url_strips_v1_suffix(mock_get):
    mock_get.return_value = _mock_version_response("0.14.0")
    is_vllm_server_with_structured_output("http://myserver:9000/v1", HEADERS)
    mock_get.assert_called_once_with(
        "http://myserver:9000/version",
        headers=HEADERS,
        timeout=5,
        allow_redirects=False,
    )


@patch("mellea.helpers.server_type.requests.get")
def test_url_strips_v1_slash_suffix(mock_get):
    mock_get.return_value = _mock_version_response("0.14.0")
    is_vllm_server_with_structured_output("http://myserver:9000/v1/", HEADERS)
    mock_get.assert_called_once_with(
        "http://myserver:9000/version",
        headers=HEADERS,
        timeout=5,
        allow_redirects=False,
    )


@patch("mellea.helpers.server_type.requests.get")
def test_url_without_v1(mock_get):
    mock_get.return_value = _mock_version_response("0.14.0")
    is_vllm_server_with_structured_output("http://myserver:9000", HEADERS)
    mock_get.assert_called_once_with(
        "http://myserver:9000/version",
        headers=HEADERS,
        timeout=5,
        allow_redirects=False,
    )


# --- SSRF protection: link-local / cloud metadata hosts ---


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/v1",  # cloud metadata endpoint
        "http://169.254.169.254:8000/v1",
        "http://169.254.0.1/v1",  # link-local range
        "http://[fe80::1]/v1",  # IPv6 link-local
    ],
)
@patch("mellea.helpers.server_type.requests.get")
def test_link_local_host_blocked(mock_get, url):
    """Link-local / metadata endpoints must be denied before any request."""
    assert is_vllm_server_with_structured_output(url, HEADERS) is False
    mock_get.assert_not_called()


@patch("mellea.helpers.server_type.socket.getaddrinfo")
@patch("mellea.helpers.server_type.requests.get")
def test_hostname_resolving_to_link_local_blocked(mock_get, mock_getaddrinfo):
    """A hostname that resolves to a link-local address must be denied."""
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("169.254.169.254", 0))]
    assert (
        is_vllm_server_with_structured_output("http://metadata.internal/v1", HEADERS)
        is False
    )
    mock_get.assert_not_called()


@patch("mellea.helpers.server_type.socket.getaddrinfo")
@patch("mellea.helpers.server_type.requests.get")
def test_unresolvable_host_not_blocked(mock_get, mock_getaddrinfo):
    """An unresolvable host is not treated as link-local; request still attempted."""
    import socket as _socket

    mock_getaddrinfo.side_effect = _socket.gaierror("name resolution failed")
    mock_get.return_value = _mock_version_response("0.14.0")
    assert (
        is_vllm_server_with_structured_output("http://unresolvable.example/v1", HEADERS)
        is True
    )
    mock_get.assert_called_once()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])


# --- vllm_server_version ---


@patch("mellea.helpers.server_type.requests.get")
def test_vllm_server_version_parses(mock_get):
    mock_get.return_value = _mock_version_response("0.30.0")
    assert str(vllm_server_version(BASE_URL, HEADERS)) == "0.30.0"


@patch("mellea.helpers.server_type.requests.get")
def test_vllm_server_version_none_when_not_ok(mock_get):
    mock_get.return_value = _mock_version_response("0.30.0", status_code=404)
    assert vllm_server_version(BASE_URL, HEADERS) is None


@patch("mellea.helpers.server_type.requests.get", side_effect=OSError("down"))
def test_vllm_server_version_none_on_error(mock_get):
    assert vllm_server_version(BASE_URL, HEADERS) is None


@patch("mellea.helpers.server_type.requests.get")
def test_vllm_server_version_refuses_link_local(mock_get):
    assert vllm_server_version("http://169.254.169.254/v1", HEADERS) is None
    mock_get.assert_not_called()


# --- vllm_tokenize_single_tokens ---


def _tokenize_response(tokens, status_code=200):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.ok = 200 <= status_code < 400
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"tokens": tokens}
    return mock_resp


@patch("mellea.helpers.server_type.requests.post")
def test_tokenize_single_tokens_returns_ids_in_order(mock_post):
    mock_post.side_effect = [_tokenize_response([32]), _tokenize_response([33])]
    ids = vllm_tokenize_single_tokens(BASE_URL, HEADERS, "m", ["A", "B"])
    assert ids == [32, 33]
    first = mock_post.call_args_list[0]
    assert first.args[0] == "http://localhost:8000/tokenize"
    assert first.kwargs["json"] == {
        "model": "m",
        "prompt": "A",
        "add_special_tokens": False,
    }
    assert first.kwargs["allow_redirects"] is False


@patch("mellea.helpers.server_type.requests.post")
def test_tokenize_rejects_multi_token_label(mock_post):
    mock_post.return_value = _tokenize_response([1905, 7])
    with pytest.raises(ValueError, match="single-token"):
        vllm_tokenize_single_tokens(BASE_URL, HEADERS, "m", ["AB"])


@patch("mellea.helpers.server_type.requests.post")
def test_tokenize_raises_on_server_error(mock_post):
    mock_post.return_value = _tokenize_response([], status_code=404)
    with pytest.raises(ValueError, match="404"):
        vllm_tokenize_single_tokens(BASE_URL, HEADERS, "m", ["A"])


@patch("mellea.helpers.server_type.requests.post")
def test_tokenize_refuses_link_local(mock_post):
    with pytest.raises(ValueError, match="link-local"):
        vllm_tokenize_single_tokens("http://169.254.169.254/v1", HEADERS, "m", ["A"])
    mock_post.assert_not_called()
