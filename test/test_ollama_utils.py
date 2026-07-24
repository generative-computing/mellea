# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared Ollama eviction utilities in test/_ollama_utils.py."""

from unittest import mock

from test._ollama_utils import evict_all_loaded_ollama_models, resolve_ollama_base_url


class TestResolveOllamaBaseURL:
    """Tests for host/port resolution."""

    def test_host_with_port(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "example.com:9999")
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        assert resolve_ollama_base_url() == "http://example.com:9999"

    def test_bare_host_uses_ollama_port(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "example.com")
        monkeypatch.setenv("OLLAMA_PORT", "9000")
        assert resolve_ollama_base_url() == "http://example.com:9000"

    def test_zero_host_normalized_to_localhost(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:12345")
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        assert resolve_ollama_base_url() == "http://127.0.0.1:12345"

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        assert resolve_ollama_base_url() == "http://127.0.0.1:11434"


class TestEvictAllLoadedOllamaModels:
    """Tests for the core evict-all-loaded-models logic."""

    def _mock_ps_response(self, models):
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"models": models}
        return resp

    def test_evicts_each_loaded_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        ps_resp = self._mock_ps_response(
            [{"name": "granite4.1:3b"}, {"model": "granite3.2-vision"}]
        )

        with (
            mock.patch(
                "test._ollama_utils.requests.get", return_value=ps_resp
            ) as mock_get,
            mock.patch("test._ollama_utils.requests.post") as mock_post,
        ):
            evict_all_loaded_ollama_models()

        mock_get.assert_called_once_with("http://127.0.0.1:11434/api/ps", timeout=5)
        assert mock_post.call_count == 2
        posted_models = {
            call.kwargs["json"]["model"] for call in mock_post.call_args_list
        }
        assert posted_models == {"granite4.1:3b", "granite3.2-vision"}
        for call in mock_post.call_args_list:
            assert call.kwargs["json"]["keep_alive"] == 0
            assert call.args[0] == "http://127.0.0.1:11434/api/generate"

    def test_no_loaded_models_makes_no_post(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        ps_resp = self._mock_ps_response([])

        with (
            mock.patch("test._ollama_utils.requests.get", return_value=ps_resp),
            mock.patch("test._ollama_utils.requests.post") as mock_post,
        ):
            evict_all_loaded_ollama_models()

        mock_post.assert_not_called()

    def test_ps_query_failure_warns_and_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        warnings: list[str] = []

        with (
            mock.patch(
                "test._ollama_utils.requests.get", side_effect=RuntimeError("boom")
            ),
            mock.patch("test._ollama_utils.requests.post") as mock_post,
        ):
            evict_all_loaded_ollama_models(on_warning=warnings.append)

        mock_post.assert_not_called()
        assert len(warnings) == 1
        assert "boom" in warnings[0]

    def test_per_model_failure_isolated(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        ps_resp = self._mock_ps_response([{"name": "model-a"}, {"name": "model-b"}])
        warnings: list[str] = []

        def post_side_effect(url, **kwargs):
            if kwargs["json"]["model"] == "model-a":
                raise RuntimeError("evict failed")
            return mock.Mock()

        with (
            mock.patch("test._ollama_utils.requests.get", return_value=ps_resp),
            mock.patch(
                "test._ollama_utils.requests.post", side_effect=post_side_effect
            ) as mock_post,
        ):
            evict_all_loaded_ollama_models(on_warning=warnings.append)

        # Both models attempted despite the first failing.
        assert mock_post.call_count == 2
        assert len(warnings) == 1
        assert "model-a" in warnings[0]

    def test_info_callback_invoked_on_success(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        ps_resp = self._mock_ps_response([{"name": "model-a"}])
        infos: list[str] = []

        with (
            mock.patch("test._ollama_utils.requests.get", return_value=ps_resp),
            mock.patch("test._ollama_utils.requests.post", return_value=mock.Mock()),
        ):
            evict_all_loaded_ollama_models(on_info=infos.append)

        assert any("model-a" in msg for msg in infos)

    def test_callbacks_default_to_noop(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        ps_resp = self._mock_ps_response([{"name": "model-a"}])

        # No callbacks passed — must not raise.
        with (
            mock.patch("test._ollama_utils.requests.get", return_value=ps_resp),
            mock.patch("test._ollama_utils.requests.post", return_value=mock.Mock()),
        ):
            evict_all_loaded_ollama_models()
