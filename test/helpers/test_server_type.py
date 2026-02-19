import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from mellea.helpers import is_vllm_server_with_structured_output

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
    mock_get.assert_called_once_with("http://localhost:8000/version", headers=HEADERS)


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
    mock_get.assert_called_once_with("http://myserver:9000/version", headers=HEADERS)


@patch("mellea.helpers.server_type.requests.get")
def test_url_strips_v1_slash_suffix(mock_get):
    mock_get.return_value = _mock_version_response("0.14.0")
    is_vllm_server_with_structured_output("http://myserver:9000/v1/", HEADERS)
    mock_get.assert_called_once_with("http://myserver:9000/version", headers=HEADERS)


@patch("mellea.helpers.server_type.requests.get")
def test_url_without_v1(mock_get):
    mock_get.return_value = _mock_version_response("0.14.0")
    is_vllm_server_with_structured_output("http://myserver:9000", HEADERS)
    mock_get.assert_called_once_with("http://myserver:9000/version", headers=HEADERS)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
