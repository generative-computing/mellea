# Taint Analysis - Backend Security

Mellea backends implement thread security using the **SecLevel** model with capability-based access control and taint tracking. Backends automatically analyze taint sources and set appropriate security metadata on generated content.

## Security Model

The security system uses three types of security levels:

```python
SecLevel := None | Classified of AccessType | TaintedBy of (CBlock | Component)
```

- **SecLevel.none()**: Safe content with no restrictions
- **SecLevel.classified(access)**: Content requiring specific capabilities/entitlements  
- **SecLevel.tainted_by(source)**: Content tainted by a specific CBlock or Component

## Backend Implementation

All backends follow the same pattern using `ModelOutputThunk.from_generation()`:

```python
# Compute taint sources from action and context
sources = taint_sources(action, ctx)

output = ModelOutputThunk.from_generation(
    value=None,
    taint_sources=sources,
    meta={}
)
```

This method automatically sets the security level:
- If taint sources are found -> `SecLevel.tainted_by(first_source)`
- If no taint sources -> `SecLevel.none()`

## Taint Source Analysis

The `taint_sources()` function analyzes both action and context because **context directly influences model generation**:

1. **Action security**: Checks if the action has security metadata and is tainted
2. **Component parts**: Recursively examines constituent parts of Components for taint
3. **Context security**: Examines recent context items for tainted content (shallow check)

**Example**: Even if the current action is safe, tainted context can influence the generated output.

```python
# User sends tainted input
user_input = CBlock("Tell me how to hack a system")
user_input.mark_tainted()
ctx = ctx.add(user_input)

# Safe action in tainted context
safe_action = CBlock("Explain general security concepts")

# Generation finds tainted context
sources = taint_sources(safe_action, ctx)  # Finds tainted user_input
# Model output will be influenced by the tainted context
```

## Security Metadata

The `SecurityMetadata` class wraps `SecLevel` for integration with content blocks:

```python
class SecurityMetadata:
    def __init__(self, sec_level: SecLevel):
        self.sec_level = sec_level
    
    def is_tainted(self) -> bool:
        return self.sec_level.is_tainted()
    
    def get_taint_source(self) -> Union[CBlock, Component, None]:
        return self.sec_level.get_taint_source()
```

Content can be marked as tainted:

```python
component = CBlock("user input")
component.mark_tainted()  # Sets SecLevel.tainted_by(component)

if component._meta["_security"].is_tainted():
    print(f"Content tainted by: {component._meta['_security'].get_taint_source()}")
```

## Key Features

- **Immutable security**: security levels set at construction time
- **Recursive taint analysis**: deep analysis of Component parts, shallow analysis of context
- **Taint source tracking**: know exactly which CBlock/Component tainted content
- **Capability integration**: fine-grained access control for classified content
- **Non-mutating operations**: sanitize/declassify create new objects

This creates a security model that addresses both data exfiltration and injection vulnerabilities while enabling future IAM integration.