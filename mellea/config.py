r"""Configuration file support for Mellea.

This module provides support for TOML configuration files to set default
backends, models, credentials, and other options without hardcoding them.

Configuration files are searched in the following order:
1. Project-specific: ./mellea.toml (current dir and parents)
2. User config: ~/.config/mellea/config.toml (Linux/macOS) or
   %APPDATA%\mellea\config.toml (Windows)

Values are applied with the following precedence:
1. Explicit parameters passed to start_session()
2. Project config file (if exists)
3. User config file (if exists)
4. Built-in defaults
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

# Import tomllib for Python 3.11+, tomli for Python 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "tomli is required for Python 3.10. Install it with: pip install tomli"
        )


class BackendConfig(BaseModel):
    """Configuration for backend settings."""

    name: str | None = None
    model_id: str | None = None
    model_options: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class CredentialsConfig(BaseModel):
    """Configuration for API credentials."""

    openai_api_key: str | None = None
    watsonx_api_key: str | None = None
    watsonx_project_id: str | None = None
    watsonx_url: str | None = None


class MelleaConfig(BaseModel):
    """Main configuration model for Mellea."""

    backend: BackendConfig = Field(default_factory=BackendConfig)
    credentials: CredentialsConfig = Field(default_factory=CredentialsConfig)
    context_type: str | None = None
    log_level: str | None = None


# Global cache for loaded config
_config_cache: tuple[MelleaConfig, Path | None] | None = None


def get_user_config_dir() -> Path:
    r"""Get the user configuration directory following XDG Base Directory spec.

    Returns:
        Path to user config directory (~/.config/mellea on Linux/macOS,
        %APPDATA%\mellea on Windows)
    """
    if sys.platform == "win32":
        # Windows: use APPDATA
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "mellea"
        # Fallback to user home
        return Path.home() / "AppData" / "Roaming" / "mellea"
    else:
        # Linux/macOS: use XDG_CONFIG_HOME or ~/.config
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / "mellea"
        return Path.home() / ".config" / "mellea"


def find_config_file() -> Path | None:
    """Find configuration file in standard locations.

    Searches in order:
    1. ./mellea.toml (current directory and parents)
    2. ~/.config/mellea/config.toml (or Windows equivalent)

    Returns:
        Path to config file if found, None otherwise
    """
    # Search for project config (current dir and parents)
    current = Path.cwd()
    for parent in [current, *current.parents]:
        project_config = parent / "mellea.toml"
        if project_config.exists():
            return project_config

    # Search for user config
    user_config = get_user_config_dir() / "config.toml"
    if user_config.exists():
        return user_config

    return None


def load_config(config_path: Path | None = None) -> tuple[MelleaConfig, Path | None]:
    """Load configuration from file.

    Args:
        config_path: Optional explicit path to config file. If None, searches
            standard locations.

    Returns:
        Tuple of (MelleaConfig, config_path). config_path is None if no config
        file was found.
    """
    global _config_cache

    # Return cached config if available and no explicit path provided
    if _config_cache is not None and config_path is None:
        return _config_cache

    # Find config file if not explicitly provided
    if config_path is None:
        config_path = find_config_file()

    # No config file found - return empty config
    if config_path is None:
        config = MelleaConfig()
        _config_cache = (config, None)
        return config, None

    # Load and parse config file
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Parse into Pydantic model
        config = MelleaConfig(**data)
        _config_cache = (config, config_path)
        return config, config_path

    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}") from e


def get_config_path() -> Path | None:
    """Get the path to the currently loaded config file.

    Returns:
        Path to config file if one was loaded, None otherwise
    """
    if _config_cache is None:
        load_config()
    return _config_cache[1] if _config_cache else None


def apply_credentials_to_env(config: MelleaConfig) -> None:
    """Apply credentials from config to environment variables.

    Only sets environment variables if they are not already set and the
    credential is present in the config.

    Args:
        config: Configuration containing credentials
    """
    creds = config.credentials

    # Map config fields to environment variable names
    env_mappings = {
        "openai_api_key": "OPENAI_API_KEY",
        "watsonx_api_key": "WATSONX_API_KEY",
        "watsonx_project_id": "WATSONX_PROJECT_ID",
        "watsonx_url": "WATSONX_URL",
    }

    for config_field, env_var in env_mappings.items():
        value = getattr(creds, config_field)
        if value is not None and env_var not in os.environ:
            os.environ[env_var] = value


def init_user_config(force: bool = False) -> Path:
    """Create example user configuration file.

    Args:
        force: If True, overwrite existing config file

    Returns:
        Path to created config file

    Raises:
        FileExistsError: If config file exists and force=False
    """
    config_dir = get_user_config_dir()
    config_path = config_dir / "config.toml"

    if config_path.exists() and not force:
        raise FileExistsError(
            f"Config file already exists at {config_path}. Use --force to overwrite."
        )

    # Create config directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Example config content
    example_config = """# Mellea User Configuration
# This file sets global defaults for all projects.
# Project-specific configs (./mellea.toml) override these settings.

[backend]
# Default backend to use (ollama, openai, huggingface, vllm, watsonx, litellm)
name = "ollama"

# Default model ID
model_id = "granite-4-micro:3b"

# Default model options (temperature, max_tokens, etc.)
[backend.model_options]
temperature = 0.7
max_tokens = 2048

# Backend-specific options
[backend.kwargs]
# base_url = "http://localhost:11434"  # For Ollama

[credentials]
# API keys (environment variables take precedence)
# openai_api_key = "sk-..."
# watsonx_api_key = "..."
# watsonx_project_id = "..."
# watsonx_url = "https://us-south.ml.cloud.ibm.com"

# General settings
# context_type = "simple"  # or "chat"
# log_level = "INFO"
"""

    # Write config file
    with open(config_path, "w") as f:
        f.write(example_config)

    return config_path


def init_project_config(force: bool = False) -> Path:
    """Create example project configuration file.

    Args:
        force: If True, overwrite existing config file

    Returns:
        Path to created config file

    Raises:
        FileExistsError: If config file exists and force=False
    """
    config_path = Path.cwd() / "mellea.toml"

    if config_path.exists() and not force:
        raise FileExistsError(
            f"Config file already exists at {config_path}. Use --force to overwrite."
        )

    # Example project config content
    example_config = """# Mellea Project Configuration
# This file overrides user config (~/.config/mellea/config.toml) for this project.

[backend]
# Project-specific model
model_id = "llama3.2:1b"

[backend.model_options]
temperature = 0.9

# Project-specific context type
context_type = "chat"
"""

    # Write config file
    with open(config_path, "w") as f:
        f.write(example_config)

    return config_path


def clear_config_cache() -> None:
    """Clear the cached configuration.

    Useful for testing or when config files change during runtime.
    """
    global _config_cache
    _config_cache = None
