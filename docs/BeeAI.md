# BeeAI Platform Integration with Mellea

This comprehensive guide covers the complete BeeAI Platform integration with Mellea, providing GUI-based chat interfaces, trace visualization, and production deployment capabilities.

## Overview

The BeeAI Platform (BAIP) is an open-source GUI for chat interfaces and agentic systems. Mellea's integration provides:

- **GUI Chat Interface**: Web-based chat UI without custom frontend development
- **Trace Visualization**: Built-in traces that correspond to Mellea program execution
- **Agent Discovery**: Publish and discover Mellea programs as BeeAI agents
- **Production Deployment**: Enterprise-ready deployment with BeeAI Platform infrastructure
- **Framework Agnostic**: Works with any LLM provider and deployment scenario

## Quick Start

### 1. Install BeeAI CLI

```bash
# Install using uv (recommended)
uv tool install beeai-cli

# Or use the Mellea helper command
m gui install
```

### 2. Install Mellea with BeeAI Support

```bash
# Install with BeeAI support
pip install "mellea[beeai]"

# Or if already installed, add BeeAI framework
pip install beeai-framework
```

### 3. Start Local Chat Interface

```bash
# Start BeeAI Platform with a Mellea program
m gui chat docs/examples/beeai/platform_example.py

# Or start platform without a specific program
m gui chat --port 8080 --host localhost
```

### 4. Access Web Interface

Open your browser to `http://localhost:8080/ui` to access the BeeAI Platform web interface.

## Implementation Details

### BeeAI Platform Backend

The `BeeAIPlatformBackend` extends the standard BeeAI backend with platform-specific features:

```python
from mellea.backends.beeai_platform import BeeAIPlatformBackend
from mellea.backends.formatter import TemplateFormatter

formatter = TemplateFormatter(model_id="granite3.3:2b")
backend = BeeAIPlatformBackend(
    model_id="granite3.3:2b",
    formatter=formatter,
    trace_granularity="generate",  # Options: none, generate, component, all
    enable_traces=True,
    base_url="http://localhost:11434",  # For local Ollama
)
```

### Trace System Features

#### Trace Granularity Levels

Control the level of detail in trace collection:

- **`none`**: No traces collected (minimal overhead)
- **`generate`**: Trace model generation calls only (default)
- **`component`**: Trace component-level operations
- **`all`**: Maximum detail including all operations

#### Trace Data Structure

Each trace captures:
- **Timing**: Start time, end time, duration
- **Input Data**: Prompts, model options, context information
- **Output Data**: Generated responses, parsed results, tool calls
- **Metadata**: Backend info, provider, timestamps
- **Hierarchical Nesting**: Parent-child trace relationships

#### Working with Traces

```python
# Save traces to file
trace_file = backend.save_traces("my_traces.json")

# Get trace summary
summary = backend.get_trace_summary()
print(f"Collected {summary['total_traces']} traces")

# Clear traces
backend.clear_traces()
```

### Agent Manifests

Create BeeAI agent manifests for your Mellea programs:

```bash
# Create manifest for a Mellea program
m gui manifest my_program.py --agent-name "MyAgent" --description "My custom agent"
```

This creates a manifest file that describes how to run your Mellea program as a BeeAI agent:

```json
{
  "name": "MyAgent",
  "description": "My custom agent",
  "type": "mellea_agent",
  "endpoints": {
    "chat": {
      "path": "/chat",
      "method": "POST"
    }
  },
  "capabilities": ["chat", "traces", "session_management", "mellea"]
}
```

## CLI Commands

### Complete CLI Interface

The `m gui` command provides a complete interface for BeeAI Platform integration:

```bash
# Get help
m gui --help

# Start chat interface
m gui chat [script] [options]

# Check installation status
m gui status

# Create agent manifest
m gui manifest script.py [options]

# Install BeeAI CLI
m gui install
```

### Chat Command Options

```bash
m gui chat [script_path] \
  --port 8080 \
  --host localhost \
  --trace-granularity generate \
  --auto-manifest true
```

**Options**:
- `script_path`: Optional path to Mellea program
- `--port`: Port to run BeeAI Platform (default: 8080)
- `--host`: Host to bind to (default: localhost)
- `--trace-granularity`: Trace detail level (default: generate)
- `--auto-manifest`: Automatically create agent manifest (default: true)

## Example Programs

### Basic Chat Agent

```python
#!/usr/bin/env python3
"""Simple BeeAI Platform chat agent example."""

import os
from mellea.stdlib.session import MelleaSession
from mellea.stdlib.base import CBlock
from mellea.stdlib.chat import Message
from mellea.backends.beeai_platform import BeeAIPlatformBackend
from mellea.backends.formatter import TemplateFormatter

def create_chat_agent():
    """Create a BeeAI Platform backend configured for chat."""
    formatter = TemplateFormatter(model_id="granite3.3:2b")
    
    backend = BeeAIPlatformBackend(
        model_id="granite3.3:2b",
        formatter=formatter,
        base_url="http://localhost:11434",
        trace_granularity="generate",
        enable_traces=True,
    )
    
    return backend

def chat_with_agent(message: str) -> str:
    """Chat with the agent and return response."""
    backend = create_chat_agent()
    session = MelleaSession(backend=backend)
    
    # Add system message
    system_msg = Message(role="system", content="You are a helpful AI assistant.")
    session.ctx.add(system_msg)
    
    # Add user message and generate response
    user_msg = Message(role="user", content=message)
    session.ctx.add(user_msg)
    
    result = session.backend.generate_from_context(
        action=user_msg,
        ctx=session.ctx,
        model_options={"temperature": 0.7, "max_tokens": 512}
    )
    
    return result.value

# For OpenAI-compatible serving (used by m serve)
def serve(input, requirements=None, model_options=None):
    if isinstance(input, list) and len(input) > 0:
        message = input[-1].get("content", str(input))
    else:
        message = str(input)
    
    response = chat_with_agent(message)
    return CBlock(response)

if __name__ == "__main__":
    # Interactive demo
    print("🤖 BeeAI Platform Chat Demo")
    print("💡 Run 'm gui chat' for web interface")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            response = chat_with_agent(user_input)
            print(f"🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Demo ended!")
```

## Production Deployment

### Option 1: Self-Hosted BeeAI Platform

#### Requirements

- Docker or Kubernetes cluster
- Persistent storage for traces and agent data
- Load balancer (for high availability)
- SSL/TLS certificates (for HTTPS)

#### Deployment Steps

1. **Deploy BeeAI Platform**:

```bash
# Using Docker Compose
git clone https://github.com/i-am-bee/beeai-platform
cd beeai-platform
docker-compose up -d

# Using Kubernetes (Helm)
helm repo add beeai https://charts.beeai.dev
helm install beeai-platform beeai/beeai-platform \
  --set ingress.enabled=true \
  --set ingress.host=your-domain.com \
  --set persistence.enabled=true
```

2. **Configure Environment Variables**:

```bash
# .env file for production
BEEAI_DATABASE_URL=postgresql://user:pass@db:5432/beeai
BEEAI_REDIS_URL=redis://redis:6379
BEEAI_SECRET_KEY=your-secret-key
BEEAI_ALLOWED_HOSTS=your-domain.com
BEEAI_DEBUG=false
```

3. **Deploy Mellea Programs as Agents**:

```python
# production_agent.py
import os
from mellea.backends.beeai_platform import BeeAIPlatformBackend

def create_production_backend():
    return BeeAIPlatformBackend(
        model_id=os.getenv("MODEL_ID", "gpt-4"),
        formatter=TemplateFormatter(model_id=os.getenv("MODEL_ID")),
        api_key=os.getenv("OPENAI_API_KEY"),
        provider=os.getenv("PROVIDER", "openai"),
        base_url=os.getenv("BASE_URL"),
        trace_granularity=os.getenv("TRACE_GRANULARITY", "generate"),
        enable_traces=True,
    )
```

### Option 2: Client-Provided Credentials

For scenarios where clients bring their own LLM API keys:

#### Frontend Configuration

```javascript
// Configure BeeAI UI to accept user credentials
const config = {
  providers: {
    openai: {
      requiresApiKey: true,
      keyField: "api_key",
      baseUrlField: "base_url"
    },
    anthropic: {
      requiresApiKey: true,
      keyField: "api_key"
    },
    watsonx: {
      requiresApiKey: true,
      keyField: "api_key",
      projectIdField: "project_id"
    }
  }
};
```

#### Backend Integration

```python
def create_backend_with_user_creds(user_config):
    """Create backend using user-provided credentials."""
    
    provider = user_config.get("provider", "openai")
    api_key = user_config.get("api_key")
    base_url = user_config.get("base_url")
    
    if not api_key:
        raise ValueError("API key is required")
    
    backend_config = {
        "model_id": user_config.get("model_id", "gpt-3.5-turbo"),
        "api_key": api_key,
        "provider": provider,
        "trace_granularity": "generate",
        "enable_traces": True,
    }
    
    if base_url:
        backend_config["base_url"] = base_url
    
    formatter = TemplateFormatter(model_id=backend_config["model_id"])
    backend_config["formatter"] = formatter
    
    return BeeAIPlatformBackend(**backend_config)
```

### Option 3: Enterprise Deployment with vLLM

For high-performance local model serving:

#### 1. Deploy vLLM Server

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model "microsoft/DialoGPT-medium" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "custom-model"
```

#### 2. Configure Mellea Backend

```python
backend = BeeAIPlatformBackend(
    model_id="custom-model",
    formatter=TemplateFormatter(model_id="custom-model"),
    base_url="http://vllm-server:8000/v1",
    provider="openai",  # vLLM uses OpenAI-compatible API
    trace_granularity="all",
    enable_traces=True,
)
```

#### 3. Docker Compose Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    command: >
      --model microsoft/DialoGPT-medium
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  beeai-platform:
    image: beeai/platform:latest
    ports:
      - "8080:8080"
    environment:
      - VLLM_BASE_URL=http://vllm-server:8000/v1
    depends_on:
      - vllm-server

  mellea-agent:
    build: .
    environment:
      - BASE_URL=http://vllm-server:8000/v1
      - MODEL_ID=custom-model
      - PROVIDER=openai
    depends_on:
      - vllm-server
      - beeai-platform
```

### Option 4: IBM watsonx Integration

For IBM watsonx.ai deployment:

```python
def create_watsonx_backend():
    return BeeAIPlatformBackend(
        model_id="ibm/granite-13b-chat-v2",
        formatter=TemplateFormatter(model_id="ibm/granite-13b-chat-v2"),
        api_key=os.getenv("WATSONX_API_KEY"),
        base_url="https://us-south.ml.cloud.ibm.com",
        provider="watsonx",
        model_options={
            "project_id": os.getenv("WATSONX_PROJECT_ID"),
            "deployment_id": os.getenv("WATSONX_DEPLOYMENT_ID"),
        },
        trace_granularity="generate",
        enable_traces=True,
    )
```

## Security Considerations

### 1. API Key Management

- Use environment variables for API keys
- Implement key rotation policies
- Use secrets management systems (HashiCorp Vault, Kubernetes Secrets)

### 2. Network Security

```bash
# Configure firewall rules
ufw allow 8080/tcp  # BeeAI Platform
ufw allow 443/tcp   # HTTPS
ufw deny 8000/tcp   # Block vLLM direct access
```

### 3. Authentication & Authorization

```python
# Add authentication middleware
from beeai_platform.middleware import AuthMiddleware

app.add_middleware(
    AuthMiddleware,
    jwt_secret=os.getenv("JWT_SECRET"),
    allowed_users=["user@company.com"],
)
```

## Monitoring & Observability

### 1. Trace Analytics

```python
def analyze_traces(trace_file):
    with open(trace_file) as f:
        traces = json.load(f)
    
    total_traces = len(traces["traces"])
    avg_duration = sum(t.get("duration_ms", 0) for t in traces["traces"]) / total_traces
    
    return {
        "total_traces": total_traces,
        "average_duration_ms": avg_duration,
        "backend": traces["backend"],
    }
```

### 2. Health Checks

```python
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        # Check BeeAI Platform
        platform_response = requests.get("http://localhost:8080/health", timeout=5)
        platform_healthy = platform_response.status_code == 200
        
        # Check model backend
        backend = create_production_backend()
        test_result = backend.generate_from_context(
            action=CBlock("test"),
            ctx=Context(),
        )
        backend_healthy = test_result.value is not None
        
        if platform_healthy and backend_healthy:
            return {"status": "healthy"}
        else:
            raise HTTPException(status_code=503, detail="Service unhealthy")
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")
```

## Technical Architecture

### Implementation Components

1. **BeeAIPlatformBackend** (`mellea/backends/beeai_platform.py`)
   - Extended BeeAI backend with platform-specific features
   - Configurable trace levels and data collection
   - Graceful degradation when BeeAI framework unavailable

2. **BeeAITraceContext** 
   - Manages hierarchical trace collection and serialization
   - Thread-safe trace stack management
   - JSON export for BeeAI Platform visualization

3. **CLI Interface** (`cli/gui/commands.py`)
   - Complete command set for platform management
   - Auto-installation of dependencies
   - Status checking and troubleshooting

4. **Agent Manifests**
   - Automatic generation from Mellea programs
   - BeeAI Platform discovery integration
   - Standard metadata and capability description

### Key Features

- **Automatic trace collection** during model generation
- **Hierarchical trace nesting** for complex operations
- **JSON export** of traces for BeeAI Platform visualization
- **Configurable granularity levels** with sensible defaults
- **Compatible with existing Mellea patterns**
- **Framework agnostic** - works with any LLM provider
- **Production ready** with security and monitoring

## Troubleshooting

### Common Issues

1. **BeeAI CLI Not Found**
   ```bash
   # Solution: Install BeeAI CLI
   uv tool install beeai-cli
   # Or
   m gui install
   ```

2. **Port Already in Use**
   ```bash
   # Solution: Use different port
   m gui chat --port 8081
   ```

3. **Model Not Found**
   ```bash
   # Solution: Check model availability
   ollama list  # For local models
   # Or verify API credentials for cloud models
   ```

4. **Traces Not Appearing**
   ```python
   # Solution: Ensure traces are enabled
   backend = BeeAIPlatformBackend(
       enable_traces=True,
       trace_granularity="generate"  # or higher
   )
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create backend with verbose tracing
backend = BeeAIPlatformBackend(
    model_id="your-model",
    formatter=formatter,
    trace_granularity="all",
    enable_traces=True,
)
```

## Best Practices

1. **Trace Management**: Set appropriate trace granularity for your use case
2. **Resource Monitoring**: Monitor memory usage with high trace volumes
3. **Error Handling**: Implement proper error handling for production deployments
4. **Testing**: Test agent manifests locally before production deployment
5. **Documentation**: Document your agent's capabilities in the manifest
6. **Security**: Use environment variables for sensitive configuration
7. **Performance**: Use "generate" granularity for production unless debugging

## Benefits Summary

### For Individual Developers
- **Instant GUI**: Web-based chat interface without frontend development
- **Visual Traces**: Built-in visualization of Mellea program execution
- **Local Development**: Works with local models (Ollama) out of the box
- **Easy Sharing**: Generate agent manifests for collaboration

### For Teams
- **Centralized Platform**: Single BeeAI instance for entire team
- **Agent Discovery**: Searchable catalog of Mellea agents
- **Standardized Interface**: Consistent user experience across agents
- **Enterprise Ready**: Production deployment with security and monitoring

### Framework Integration
- **Zero Vendor Lock-in**: Works with any LLM provider
- **Minimal Code Changes**: Drop-in replacement for existing backends
- **Backward Compatible**: Existing Mellea programs work unchanged
- **Extensible**: Easy to add custom trace data and metadata

## Dependencies

### Required
- `mellea` (core framework)
- `typer` (CLI interface)
- `fastapi` (web serving)

### Optional
- `beeai-framework` - For full BeeAI backend functionality
- `beeai-cli` - For platform management (auto-installable via `m gui install`)
- `requests` - For health checks and HTTP operations

### Installation Commands

```bash
# Core installation
pip install "mellea[beeai]"

# Development setup
pip install -e ".[beeai,dev]"

# Full installation with all optional dependencies
pip install "mellea[all]"
```

## Support and Resources

- **BeeAI Platform**: https://github.com/i-am-bee/beeai-platform
- **Documentation**: https://docs.beeai.dev
- **Agent Communication Protocol**: https://agentcommunicationprotocol.dev
- **Discord Community**: https://discord.gg/beeai
- **Mellea Framework**: Core Mellea documentation

---

This comprehensive guide provides everything needed to integrate Mellea with the BeeAI Platform, from local development to enterprise production deployment.
