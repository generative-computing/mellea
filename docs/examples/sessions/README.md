# Session Examples

This directory contains examples of creating and customizing Mellea sessions.

## Files

### creating_a_new_type_of_session.py
Demonstrates how to create custom session types with specialized behavior.

**Key Features:**
- Extending the base session class
- Custom context types
- Specialized session methods
- Session configuration patterns

## Concepts Demonstrated

- **Session Customization**: Creating domain-specific session types
- **Context Management**: Using different context types
- **Backend Configuration**: Setting up backends for sessions
- **Session Lifecycle**: Managing session state and resources

## Basic Session Creation

```python
from mellea import start_session

# Default session (Ollama backend, SimpleContext)
m = start_session()

# With specific model
m = start_session(model_id="ibm-granite/granite-3.2-8b-instruct")

# With chat context
from mellea.stdlib.context import ChatContext
m = start_session(ctx=ChatContext())

# With model options
from mellea.backends import ModelOption
m = start_session(model_options={
    ModelOption.MAX_NEW_TOKENS: 200,
    ModelOption.TEMPERATURE: 0.7
})
```

## Custom Session Pattern

```python
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import ChatContext

class MyCustomSession(MelleaSession):
    def __init__(self, **kwargs):
        backend = OllamaModelBackend(
            model_id="my-model",
            **kwargs
        )
        super().__init__(backend=backend, ctx=ChatContext())
    
    def custom_method(self, prompt: str):
        """Add custom functionality."""
        # Your custom logic here
        return self.instruct(prompt)

# Use custom session
m = MyCustomSession()
result = m.custom_method("Hello!")
```

## Session Context Managers

```python
# Automatic cleanup with context manager
with start_session() as m:
    result = m.instruct("Generate text")
    # Session automatically cleaned up
```

## Backend Options

```python
# Ollama (default)
from mellea.backends.ollama import OllamaModelBackend
backend = OllamaModelBackend(model_id="llama2")

# OpenAI
from mellea.backends.openai import OpenAIBackend
backend = OpenAIBackend(model_id="gpt-4")

# HuggingFace
from mellea.backends.huggingface import LocalHFBackend
backend = LocalHFBackend(model_id="ibm-granite/granite-3.2-8b-instruct")

# LiteLLM
from mellea.backends.litellm import LiteLLMBackend
backend = LiteLLMBackend(model_id="claude-3-opus")

# Create session with custom backend
m = MelleaSession(backend=backend, ctx=ChatContext())
```

## Context Types

- **SimpleContext**: Basic linear context
- **ChatContext**: Multi-turn conversation context
- **Custom**: Create your own context types

## Related Documentation

- See `mellea/stdlib/session.py` for session implementation
- See `mellea/stdlib/context.py` for context types
- See `mellea/backends/` for backend options