# Mellea Configuration Guide

This guide explains how to use configuration files to set default backends, models, credentials, and other options for Mellea.

## Quick Start

Create a user configuration file:

```bash
m config init
```

This creates `~/.config/mellea/config.toml` with example settings. Edit this file to set your preferences.

For project-specific settings:

```bash
m config init-project
```

This creates `./mellea.toml` in your current directory.

## Configuration Hierarchy

Mellea searches for configuration files in this order:

1. **Project config**: `./mellea.toml` (current directory and parent directories)
2. **User config**: `~/.config/mellea/config.toml` (Linux/macOS) or `%APPDATA%\mellea\config.toml` (Windows)

### Precedence Rules

Values are applied with the following precedence (highest to lowest):

1. **Explicit parameters** passed to `start_session()`
2. **Project config** (`./mellea.toml`)
3. **User config** (`~/.config/mellea/config.toml`)
4. **Built-in defaults**

This means you can set global defaults in your user config and override them per-project or per-call.

## Configuration File Format

Configuration files use [TOML](https://toml.io/) format. Here's a complete example:

```toml
# ~/.config/mellea/config.toml

[backend]
# Backend to use: "ollama", "hf", "openai", "watsonx", "litellm"
name = "ollama"

# Model identifier
model_id = "granite-4-micro:3b"

# Model options (temperature, max_tokens, etc.)
[backend.model_options]
temperature = 0.7
max_tokens = 2048
top_p = 0.9

# Backend-specific options
[backend.kwargs]
# For Ollama:
# base_url = "http://localhost:11434"

# For OpenAI:
# organization = "org-..."

[credentials]
# API keys (environment variables take precedence)
# openai_api_key = "sk-..."
# watsonx_api_key = "..."
# watsonx_project_id = "..."
# watsonx_url = "https://us-south.ml.cloud.ibm.com"

# General settings
context_type = "simple"  # or "chat"
log_level = "INFO"       # DEBUG, INFO, WARNING, ERROR
```

## Configuration Options

### Backend Settings

#### `backend.name`
- **Type**: String
- **Options**: `"ollama"`, `"hf"`, `"openai"`, `"watsonx"`, `"litellm"`
- **Default**: `"ollama"`
- **Description**: The backend to use for model inference

#### `backend.model_id`
- **Type**: String
- **Default**: `"granite-4-micro:3b"`
- **Description**: Model identifier. Format depends on backend:
  - Ollama: `"llama3.2:1b"`, `"granite-4-micro:3b"`
  - OpenAI: `"gpt-4"`, `"gpt-3.5-turbo"`
  - HuggingFace: `"microsoft/DialoGPT-medium"`
  - Watsonx: Model ID from IBM Watsonx catalog

#### `backend.model_options`
- **Type**: Dictionary
- **Default**: `{}`
- **Description**: Model-specific options. Common options:
  - `temperature` (float): Sampling temperature (0.0-2.0)
  - `max_tokens` (int): Maximum tokens to generate
  - `top_p` (float): Nucleus sampling threshold
  - `top_k` (int): Top-k sampling parameter
  - `frequency_penalty` (float): Frequency penalty (OpenAI)
  - `presence_penalty` (float): Presence penalty (OpenAI)

#### `backend.kwargs`
- **Type**: Dictionary
- **Default**: `{}`
- **Description**: Backend-specific constructor arguments:
  - Ollama: `base_url`, `timeout`
  - OpenAI: `organization`, `base_url`
  - HuggingFace: `device`, `torch_dtype`

### Credentials

#### `credentials.openai_api_key`
- **Type**: String
- **Default**: None
- **Description**: OpenAI API key. Environment variable `OPENAI_API_KEY` takes precedence.

#### `credentials.watsonx_api_key`
- **Type**: String
- **Default**: None
- **Description**: IBM Watsonx API key. Environment variable `WATSONX_API_KEY` takes precedence.

#### `credentials.watsonx_project_id`
- **Type**: String
- **Default**: None
- **Description**: IBM Watsonx project ID. Environment variable `WATSONX_PROJECT_ID` takes precedence.

#### `credentials.watsonx_url`
- **Type**: String
- **Default**: None
- **Description**: IBM Watsonx API URL. Environment variable `WATSONX_URL` takes precedence.

### General Settings

#### `context_type`
- **Type**: String
- **Options**: `"simple"`, `"chat"`
- **Default**: `"simple"`
- **Description**: Default context type for sessions
  - `"simple"`: Each interaction is independent
  - `"chat"`: Maintains conversation history

#### `log_level`
- **Type**: String
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- **Default**: `"INFO"`
- **Description**: Logging level for Mellea

## Example Configurations

### Local Development with Ollama

```toml
# ~/.config/mellea/config.toml
[backend]
name = "ollama"
model_id = "llama3.2:1b"

[backend.model_options]
temperature = 0.8
max_tokens = 4096

context_type = "chat"
log_level = "DEBUG"
```

### Production with OpenAI

```toml
# ~/.config/mellea/config.toml
[backend]
name = "openai"
model_id = "gpt-4"

[backend.model_options]
temperature = 0.7
max_tokens = 2048

[credentials]
openai_api_key = "sk-..."  # Better: use environment variable

context_type = "chat"
log_level = "INFO"
```

### Project-Specific Override

```toml
# ./mellea.toml (in your project directory)
[backend]
# Override user config for this project
model_id = "llama3.2:3b"

[backend.model_options]
temperature = 0.9  # More creative for this project

context_type = "simple"
```

### HuggingFace Local Models

```toml
# ~/.config/mellea/config.toml
[backend]
name = "hf"
model_id = "microsoft/DialoGPT-medium"

[backend.kwargs]
device = "cuda"  # or "cpu"
torch_dtype = "float16"

[backend.model_options]
temperature = 0.8
max_tokens = 512
```

## CLI Commands

### View Current Configuration

```bash
# Show effective configuration
m config show

# Show which config file is being used
m config path

# Show all possible config locations
m config where
```

### Initialize Configuration

```bash
# Create user config
m config init

# Create project config
m config init-project

# Force overwrite existing config
m config init --force
m config init-project --force
```

## Security Best Practices

### Credentials Management

1. **Use Environment Variables**: For CI/CD and production, use environment variables instead of config files:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export WATSONX_API_KEY="..."
   ```

2. **Don't Commit Credentials**: The `.gitignore` file excludes `mellea.toml` and `.mellea.toml` by default. User config (`~/.config/mellea/config.toml`) is outside your repository.

3. **File Permissions**: Ensure config files with credentials have restricted permissions:
   ```bash
   chmod 600 ~/.config/mellea/config.toml
   ```

4. **Use Separate Configs**: Keep credentials in user config, not project config:
   - User config: API keys and credentials
   - Project config: Model settings and preferences

### Example: Secure Setup

**User config** (`~/.config/mellea/config.toml`):
```toml
[credentials]
openai_api_key = "sk-..."
watsonx_api_key = "..."
```

**Project config** (`./mellea.toml`, safe to commit):
```toml
[backend]
name = "openai"
model_id = "gpt-4"

[backend.model_options]
temperature = 0.7
```

## Programmatic Usage

Configuration is automatically loaded when you call `start_session()`:

```python
from mellea import start_session

# Uses config file settings
with start_session() as session:
    response = session.instruct("Hello!")

# Override config with explicit parameters
with start_session(backend_name="openai", model_id="gpt-4") as session:
    response = session.instruct("Hello!")

# Merge model_options with config
with start_session(model_options={"temperature": 0.9}) as session:
    # Config temperature is overridden to 0.9
    response = session.instruct("Hello!")
```

## Troubleshooting

### Config Not Loading

1. Check which config is being used:
   ```bash
   m config path
   ```

2. Verify config syntax:
   ```bash
   python -c "import tomllib; tomllib.load(open('mellea.toml', 'rb'))"
   ```

3. Check for typos in field names (case-sensitive)

### Credentials Not Working

1. Environment variables take precedence over config files
2. Check if credentials are set in environment:
   ```bash
   echo $OPENAI_API_KEY
   ```

3. Verify credentials are in the correct section:
   ```toml
   [credentials]  # Not [backend.credentials]
   openai_api_key = "..."
   ```

### Model Not Found

1. Verify model ID format for your backend
2. For Ollama, ensure model is pulled:
   ```bash
   ollama pull llama3.2:1b
   ```

3. Check backend-specific model naming conventions

## Advanced Topics

### Multiple Profiles

While not directly supported, you can use multiple config files:

```bash
# Development
cp ~/.config/mellea/config-dev.toml ~/.config/mellea/config.toml

# Production
cp ~/.config/mellea/config-prod.toml ~/.config/mellea/config.toml
```

Or use environment-specific project configs:

```bash
# Use different configs per environment
cp mellea-dev.toml mellea.toml  # For development
cp mellea-prod.toml mellea.toml  # For production
```

### Config in CI/CD

For CI/CD pipelines, use environment variables instead of config files:

```yaml
# GitHub Actions example
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  WATSONX_API_KEY: ${{ secrets.WATSONX_API_KEY }}
```

### Dynamic Configuration

For dynamic configuration, use explicit parameters:

```python
import os
from mellea import start_session

# Load config from custom source
backend = os.getenv("MELLEA_BACKEND", "ollama")
model = os.getenv("MELLEA_MODEL", "llama3.2:1b")

with start_session(backend_name=backend, model_id=model) as session:
    response = session.instruct("Hello!")
```

## Migration Guide

### From Hardcoded Settings

**Before:**
```python
from mellea import start_session

with start_session("ollama", "llama3.2:1b") as session:
    response = session.instruct("Hello!")
```

**After (with config):**
```toml
# ~/.config/mellea/config.toml
[backend]
name = "ollama"
model_id = "llama3.2:1b"
```

```python
from mellea import start_session

# Uses config automatically
with start_session() as session:
    response = session.instruct("Hello!")
```

### From Environment Variables

**Before:**
```bash
export MELLEA_BACKEND="ollama"
export MELLEA_MODEL="llama3.2:1b"
```

**After:**
```toml
# ~/.config/mellea/config.toml
[backend]
name = "ollama"
model_id = "llama3.2:1b"
```

Environment variables for credentials still work and take precedence.

## See Also

- [Mellea Documentation](../README.md)
- [TOML Specification](https://toml.io/)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
