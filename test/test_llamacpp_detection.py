# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the llama.cpp capability gate in `test/conftest.py`.

`_check_llamacpp_available` decides whether `llamacpp`-marked tests and examples get
collected, and it has to answer two questions with one request: is a server ready,
and is it actually llama.cpp? Port 8080 is contended, so a bare liveness probe can
pass against an unrelated service, and an OpenAI-compatible one (vLLM, TGI) would
then serve every request from a different model.

These tests pin that behaviour against captured responses, so the gate cannot
regress into a liveness-only check, and so `LLAMACPP_PROPS_KEYS` cannot drift away
from what a real llama-server sends.
"""

import email.message
import json
import urllib.error
import urllib.request

import pytest

from test.conftest import (
    LLAMACPP_DEFAULT_BASE_URL,
    LLAMACPP_PROPS_KEYS,
    _check_llamacpp_available,
)

# Captured from `GET /props` on llama-server (llama.cpp b7061, serving a Granite
# GGUF). Nested values are trimmed; every top-level key the server returned is kept.
LLAMACPP_PROPS = {
    "bos_token": "<|end_of_text|>",
    "build_info": "b7061-8f1d2e3",
    "chat_template": "{%- if messages[0]['role'] == 'system' %}",
    "chat_template_caps": {"tools": True},
    "cors_proxy_enabled": False,
    "default_generation_settings": {"params": {"temperature": 0.8, "top_k": 40}},
    "endpoint_metrics": False,
    "endpoint_props": True,
    "endpoint_slots": False,
    "eos_token": "<|end_of_text|>",
    "is_sleeping": False,
    "media_marker": "<__media__>",
    "modalities": {"vision": False, "audio": False},
    "model_alias": "granite-4.2-3b",
    "model_ftype": 15,
    "model_path": "/models/granite-4.2-3b-Q4_K_M.gguf",
    "total_slots": 1,
    "ui": {},
    "ui_settings": {},
}

# Bodies a 200 can carry from something that is not a ready llama-server. The first
# is a generic health payload, the second an OpenAI-compatible model listing (the
# shape that would otherwise silently swap the model under the tests), then a web
# page, a JSON array, and an empty body.
NON_LLAMACPP_BODIES = [
    pytest.param(b'{"status":"ok"}', id="generic-health-json"),
    pytest.param(
        b'{"object":"list","data":[{"id":"granite4:micro-h","object":"model"}]}',
        id="openai-model-listing",
    ),
    pytest.param(b"<html><body>It works</body></html>", id="html-page"),
    pytest.param(b"[]", id="json-array"),
    pytest.param(b"", id="empty-body"),
]


class _FakeResponse:
    """Stand-in for the context manager `urllib.request.urlopen` returns."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        return None


def _patch_urlopen(monkeypatch, handler) -> list[str]:
    """Route `urlopen` to `handler(url)`, recording every URL it was called with."""
    urls: list[str] = []

    def fake_urlopen(url, timeout=None):
        urls.append(url)
        return handler(url)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return urls


@pytest.fixture(autouse=True)
def _unset_base_url(monkeypatch):
    """Keep a developer's own `LLAMACPP_BASE_URL` out of the default-URL cases."""
    monkeypatch.delenv("LLAMACPP_BASE_URL", raising=False)


def test_props_keys_are_a_subset_of_a_real_props_payload():
    """The fingerprint must only require keys a real llama-server actually sends."""
    missing = [key for key in LLAMACPP_PROPS_KEYS if key not in LLAMACPP_PROPS]
    assert not missing, f"not returned by llama-server /props: {missing}"


def test_ready_llamacpp_server_passes_the_gate(monkeypatch):
    urls = _patch_urlopen(
        monkeypatch, lambda _: _FakeResponse(json.dumps(LLAMACPP_PROPS).encode())
    )
    assert _check_llamacpp_available() is True
    assert urls == ["http://127.0.0.1:8080/props"]


@pytest.mark.parametrize("body", NON_LLAMACPP_BODIES)
def test_foreign_server_on_the_port_fails_the_gate(monkeypatch, body):
    """A 200 from something that is not llama-server must not open the gate."""
    _patch_urlopen(monkeypatch, lambda _: _FakeResponse(body))
    assert _check_llamacpp_available() is False


def _http_error(status: int, reason: str) -> urllib.error.HTTPError:
    """The `HTTPError` `urlopen` raises for a non-2xx status."""
    return urllib.error.HTTPError(
        "http://127.0.0.1:8080/props", status, reason, email.message.Message(), None
    )


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_http_error(503, "Loading model"), id="503-loading-weights"),
        pytest.param(_http_error(404, "Not Found"), id="404-no-props-endpoint"),
        pytest.param(urllib.error.URLError("Connection refused"), id="refused"),
        pytest.param(TimeoutError("timed out"), id="timeout"),
    ],
)
def test_unusable_server_fails_the_gate(monkeypatch, error):
    """Loading, absent, and unreachable servers all read as unavailable.

    The 503 case is the one a socket check gets wrong: llama-server binds the port
    and accepts connections while it is still downloading and loading weights.
    """

    def raise_error(_):
        raise error

    _patch_urlopen(monkeypatch, raise_error)
    assert _check_llamacpp_available() is False


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        pytest.param(
            LLAMACPP_DEFAULT_BASE_URL, "http://127.0.0.1:8080/props", id="default"
        ),
        pytest.param(
            "http://127.0.0.1:9090/v1", "http://127.0.0.1:9090/props", id="custom-port"
        ),
        pytest.param(
            "http://gpu-box:8080/v1/", "http://gpu-box:8080/props", id="trailing-slash"
        ),
        pytest.param(
            "http://gpu-box:8080", "http://gpu-box:8080/props", id="no-v1-suffix"
        ),
    ],
)
def test_base_url_env_override_is_resolved_to_props(monkeypatch, base_url, expected):
    monkeypatch.setenv("LLAMACPP_BASE_URL", base_url)
    urls = _patch_urlopen(
        monkeypatch, lambda _: _FakeResponse(json.dumps(LLAMACPP_PROPS).encode())
    )
    assert _check_llamacpp_available() is True
    assert urls == [expected]
