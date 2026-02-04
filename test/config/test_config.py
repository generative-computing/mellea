"""Unit tests for Mellea configuration module."""

import os
from pathlib import Path

import pytest

from mellea.config import (
    BackendConfig,
    CredentialsConfig,
    MelleaConfig,
    apply_credentials_to_env,
    clear_config_cache,
    find_config_file,
    init_project_config,
    load_config,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear config cache before each test."""
    clear_config_cache()
    yield
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


class TestConfigModels:
    """Test Pydantic config models."""

    def test_backend_config_defaults(self):
        """Test BackendConfig with default values."""
        config = BackendConfig()
        assert config.name is None
        assert config.model_id is None
        assert config.model_options == {}
        assert config.kwargs == {}

    def test_backend_config_with_values(self):
        """Test BackendConfig with explicit values."""
        config = BackendConfig(
            name="ollama",
            model_id="llama3.2:1b",
            model_options={"temperature": 0.7},
            kwargs={"base_url": "http://localhost:11434"},
        )
        assert config.name == "ollama"
        assert config.model_id == "llama3.2:1b"
        assert config.model_options["temperature"] == 0.7
        assert config.kwargs["base_url"] == "http://localhost:11434"

    def test_credentials_config_defaults(self):
        """Test CredentialsConfig with default values."""
        config = CredentialsConfig()
        assert config.openai_api_key is None
        assert config.watsonx_api_key is None
        assert config.watsonx_project_id is None
        assert config.watsonx_url is None

    def test_mellea_config_defaults(self):
        """Test MelleaConfig with default values."""
        config = MelleaConfig()
        assert isinstance(config.backend, BackendConfig)
        assert isinstance(config.credentials, CredentialsConfig)
        assert config.context_type is None
        assert config.log_level is None


class TestConfigDiscovery:
    """Test configuration file discovery."""

    def test_find_config_file_none(self, temp_project_dir):
        """Test finding config when none exists."""
        config_path = find_config_file()
        assert config_path is None

    def test_find_config_file_project(self, temp_project_dir):
        """Test finding project config file."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("[backend]\nname = 'openai'")

        config_path = find_config_file()
        assert config_path == project_config

    def test_find_config_file_parent_dir(self, temp_project_dir):
        """Test finding config in parent directory."""
        # Create config in project dir
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("[backend]\nname = 'ollama'")

        # Create and cd to subdirectory
        subdir = temp_project_dir / "src" / "module"
        subdir.mkdir(parents=True)
        os.chdir(subdir)

        config_path = find_config_file()
        assert config_path == project_config


class TestConfigLoading:
    """Test configuration loading and parsing."""

    def test_load_config_empty(self, temp_project_dir):
        """Test loading config when no file exists."""
        config, path = load_config()
        assert isinstance(config, MelleaConfig)
        assert path is None
        assert config.backend.name is None

    def test_load_config_basic(self, temp_project_dir):
        """Test loading a basic config file."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[backend]
name = "ollama"
model_id = "llama3.2:1b"

[backend.model_options]
temperature = 0.8
max_tokens = 2048
""")

        config, path = load_config()
        assert path == project_config
        assert config.backend.name == "ollama"
        assert config.backend.model_id == "llama3.2:1b"
        assert config.backend.model_options["temperature"] == 0.8
        assert config.backend.model_options["max_tokens"] == 2048

    def test_load_config_with_credentials(self, temp_project_dir):
        """Test loading config with credentials."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
[credentials]
openai_api_key = "sk-test123"
watsonx_api_key = "wx-test456"
watsonx_project_id = "proj-789"
""")

        config, _path = load_config()
        assert config.credentials.openai_api_key == "sk-test123"
        assert config.credentials.watsonx_api_key == "wx-test456"
        assert config.credentials.watsonx_project_id == "proj-789"

    def test_load_config_with_general_settings(self, temp_project_dir):
        """Test loading config with general settings."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("""
context_type = "chat"
log_level = "DEBUG"
""")

        config, _path = load_config()
        assert config.context_type == "chat"
        assert config.log_level == "DEBUG"

    def test_load_config_invalid_toml(self, temp_project_dir):
        """Test loading invalid TOML raises error."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("invalid toml [[[")

        with pytest.raises(ValueError, match="Error loading config"):
            load_config()

    def test_load_config_caching(self, temp_project_dir):
        """Test that config is cached after first load."""
        project_config = temp_project_dir / "mellea.toml"
        project_config.write_text("[backend]\nname = 'ollama'")

        # First load
        config1, path1 = load_config()

        # Second load should return cached version
        config2, path2 = load_config()

        assert config1 is config2
        assert path1 == path2


class TestCredentialApplication:
    """Test credential application to environment."""

    def test_apply_credentials_to_env(self, monkeypatch):
        """Test applying credentials to environment variables."""
        # Clear any existing env vars
        for key in ["OPENAI_API_KEY", "WATSONX_API_KEY", "WATSONX_PROJECT_ID"]:
            monkeypatch.delenv(key, raising=False)

        config = MelleaConfig(
            credentials=CredentialsConfig(
                openai_api_key="sk-test123",
                watsonx_api_key="wx-test456",
                watsonx_project_id="proj-789",
            )
        )

        apply_credentials_to_env(config)

        assert os.environ["OPENAI_API_KEY"] == "sk-test123"
        assert os.environ["WATSONX_API_KEY"] == "wx-test456"
        assert os.environ["WATSONX_PROJECT_ID"] == "proj-789"

    def test_apply_credentials_respects_existing_env(self, monkeypatch):
        """Test that existing env vars are not overwritten."""
        monkeypatch.setenv("OPENAI_API_KEY", "existing-key")

        config = MelleaConfig(credentials=CredentialsConfig(openai_api_key="new-key"))

        apply_credentials_to_env(config)

        # Should keep existing value
        assert os.environ["OPENAI_API_KEY"] == "existing-key"

    def test_apply_credentials_skips_none(self, monkeypatch):
        """Test that None credentials are not set."""
        for key in ["OPENAI_API_KEY", "WATSONX_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        config = MelleaConfig(
            credentials=CredentialsConfig(openai_api_key=None, watsonx_api_key=None)
        )

        apply_credentials_to_env(config)

        assert "OPENAI_API_KEY" not in os.environ
        assert "WATSONX_API_KEY" not in os.environ


class TestBackendModelOptionsHierarchy:
    """Test per-backend model options."""

    def test_generic_options_only(self):
        """Test that generic options are returned when no backend-specific options exist."""
        config = BackendConfig(model_options={"temperature": 0.7, "max_tokens": 100})
        result = config.get_model_options_for_backend("ollama")
        assert result == {"temperature": 0.7, "max_tokens": 100}

    def test_backend_specific_options(self):
        """Test that backend-specific options are returned."""
        config = BackendConfig(
            model_options={"temperature": 0.7, "ollama": {"num_ctx": 4096}}
        )
        result = config.get_model_options_for_backend("ollama")
        assert result == {"temperature": 0.7, "num_ctx": 4096}

    def test_backend_specific_overrides_generic(self):
        """Test that backend-specific options override generic options."""
        config = BackendConfig(
            model_options={
                "temperature": 0.7,
                "ollama": {"temperature": 0.9, "num_ctx": 4096},
            }
        )
        result = config.get_model_options_for_backend("ollama")
        assert result == {"temperature": 0.9, "num_ctx": 4096}

    def test_different_backends_get_different_options(self):
        """Test that different backends get their own specific options."""
        config = BackendConfig(
            model_options={
                "temperature": 0.7,
                "ollama": {"num_ctx": 4096},
                "openai": {"presence_penalty": 0.5},
            }
        )
        ollama_result = config.get_model_options_for_backend("ollama")
        openai_result = config.get_model_options_for_backend("openai")

        assert ollama_result == {"temperature": 0.7, "num_ctx": 4096}
        assert openai_result == {"temperature": 0.7, "presence_penalty": 0.5}

    def test_backend_without_specific_options(self):
        """Test that a backend without specific options gets only generic options."""
        config = BackendConfig(
            model_options={"temperature": 0.7, "ollama": {"num_ctx": 4096}}
        )
        result = config.get_model_options_for_backend("openai")
        assert result == {"temperature": 0.7}

    def test_empty_model_options(self):
        """Test with empty model options."""
        config = BackendConfig(model_options={})
        result = config.get_model_options_for_backend("ollama")
        assert result == {}


class TestConfigInitialization:
    """Test config file initialization."""

    def test_init_project_config(self, temp_project_dir):
        """Test creating project config file."""
        config_path = init_project_config()

        assert config_path.exists()
        assert config_path == temp_project_dir / "mellea.toml"

        # Verify content is valid TOML
        content = config_path.read_text()
        assert "[backend]" in content

    def test_init_project_config_exists(self, temp_project_dir):
        """Test that init fails if project config exists without force."""
        config_path = temp_project_dir / "mellea.toml"
        config_path.write_text("existing")

        with pytest.raises(FileExistsError, match="already exists"):
            init_project_config(force=False)

    def test_init_project_config_force(self, temp_project_dir):
        """Test that force overwrites existing project config."""
        config_path = temp_project_dir / "mellea.toml"
        config_path.write_text("existing")

        new_path = init_project_config(force=True)

        assert new_path == config_path
        content = config_path.read_text()
        assert "existing" not in content
        assert "[backend]" in content
