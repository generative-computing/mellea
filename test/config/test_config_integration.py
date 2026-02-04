"""Integration tests for Mellea configuration with start_session()."""

import os
from pathlib import Path

import pytest

from mellea.config import clear_config_cache, init_project_config
from mellea.stdlib.session import start_session


@pytest.fixture(autouse=True)
def clear_cache_and_env():
    """Clear config cache and environment variables before each test."""
    clear_config_cache()

    # Store original env vars
    original_env = {}
    env_vars = [
        "OPENAI_API_KEY",
        "WATSONX_API_KEY",
        "WATSONX_PROJECT_ID",
        "WATSONX_URL",
    ]

    for var in env_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore original env vars
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    for var, value in original_env.items():
        os.environ[var] = value

    clear_config_cache()


@pytest.fixture
def temp_project_dir(tmp_path, monkeypatch):
    """Create a temporary project directory."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)

    # Change to project directory
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    yield project_dir

    # Restore original directory
    os.chdir(original_cwd)


class TestSessionWithConfig:
    """Test start_session() with configuration files."""

    @pytest.mark.ollama
    def test_session_uses_project_config(self, temp_project_dir):
        """Test that start_session() uses project config."""
        # Create project config
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:1b"

[backend.model_options]
temperature = 0.8
max_tokens = 100
""")

        # Start session without explicit parameters
        with start_session() as session:
            assert session.backend.model_id == "llama3.2:1b"

    @pytest.mark.ollama
    def test_session_explicit_overrides_config(self, temp_project_dir):
        """Test that explicit parameters override config."""
        # Create project config
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:1b"
""")

        # Start session with explicit model_id
        with start_session(model_id="granite-4-micro:3b") as session:
            assert session.backend.model_id == "granite-4-micro:3b"

    @pytest.mark.ollama
    def test_session_without_config(self, temp_project_dir):
        """Test that start_session() works without config files."""
        # No config files created

        # Start session with defaults
        with start_session() as session:
            # Should use default backend and model
            assert session.backend is not None
            assert session.ctx is not None

    @pytest.mark.ollama
    def test_session_credentials_from_config(self, temp_project_dir):
        """Test that credentials from config are applied to environment."""
        # Create project config with credentials
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"

[credentials]
openai_api_key = "sk-test-from-config"
watsonx_api_key = "wx-test-from-config"
""")

        # Start session
        with start_session() as _session:
            # Credentials should be in environment
            assert os.environ.get("OPENAI_API_KEY") == "sk-test-from-config"
            assert os.environ.get("WATSONX_API_KEY") == "wx-test-from-config"

    @pytest.mark.ollama
    def test_session_env_overrides_config_credentials(
        self, temp_project_dir, monkeypatch
    ):
        """Test that environment variables override config credentials."""
        # Set environment variable
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        # Create project config with different credential
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"

[credentials]
openai_api_key = "sk-from-config"
""")

        # Start session
        with start_session() as _session:
            # Environment variable should take precedence
            assert os.environ.get("OPENAI_API_KEY") == "sk-from-env"


class TestConfigPrecedence:
    """Test configuration precedence in real scenarios."""

    @pytest.mark.ollama
    def test_explicit_overrides_project(self, temp_project_dir):
        """Test complete precedence: explicit > project > default."""
        # Create project config
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:1b"

[backend.model_options]
temperature = 0.7
""")

        # Test 1: No explicit params - uses project config
        with start_session() as session:
            assert session.backend.model_id == "llama3.2:1b"

        # Test 2: Explicit model_id - overrides project config
        with start_session(model_id="granite-4-micro:3b") as session:
            assert session.backend.model_id == "granite-4-micro:3b"


class TestConfigWithDifferentBackends:
    """Test configuration with different backend types."""

    @pytest.mark.ollama
    def test_ollama_backend_from_config(self, temp_project_dir):
        """Test Ollama backend configuration."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:1b"

[backend.kwargs]
base_url = "http://localhost:11434"
""")

        with start_session() as session:
            assert session.backend.model_id == "llama3.2:1b"

    @pytest.mark.openai
    @pytest.mark.requires_api_key
    def test_openai_backend_from_config(self, temp_project_dir, monkeypatch):
        """Test OpenAI backend configuration."""
        # Set API key in environment
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "openai"
model_id = "gpt-3.5-turbo"

[backend.model_options]
temperature = 0.7
max_tokens = 100
""")

        with start_session() as session:
            assert session.backend.model_id == "gpt-3.5-turbo"


class TestConfigCaching:
    """Test that config caching works correctly with sessions."""

    @pytest.mark.ollama
    def test_config_cached_across_sessions(self, temp_project_dir):
        """Test that config is cached and reused across multiple sessions."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:1b"
""")

        # First session
        with start_session() as session1:
            model1 = session1.backend.model_id

        # Second session - should use cached config
        with start_session() as session2:
            model2 = session2.backend.model_id

        assert model1 == model2 == "llama3.2:1b"

    @pytest.mark.ollama
    def test_config_cache_cleared(self, temp_project_dir):
        """Test that clearing cache forces config reload."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:1b"
""")

        # First session
        with start_session() as session1:
            model1 = session1.backend.model_id

        # Modify config
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:3b"
""")

        # Clear cache
        clear_config_cache()

        # Second session - should reload config
        with start_session() as session2:
            model2 = session2.backend.model_id

        assert model1 == "llama3.2:1b"
        assert model2 == "llama3.2:3b"
