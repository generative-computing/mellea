# Taint Tracking - Backend Security

Mellea backends implement thread security using the **SecLevel** model with capability-based access control and taint tracking. Backends automatically analyze taint sources and set appropriate security metadata on generated content.

## Security Model

The security system uses three types of security levels:

```python
SecLevel := None | Classified of AccessType | TaintedBy of (list[CBlock | Component] | None)
```

- **SecLevel.none()**: Safe content with no restrictions
- **SecLevel.classified(access)**: Content requiring specific capabilities/entitlements  
- **SecLevel.tainted_by(sources)**: Content tainted by one or more CBlocks/Components (list), or None for root tainted nodes

## Backend Implementation

All backends follow the same pattern when creating `ModelOutputThunk`:

```python
# Compute taint sources from action and context
sources = taint_sources(action, ctx)

# Set security level based on taint sources
from mellea.security import SecLevel
sec_level = SecLevel.tainted_by(sources) if sources else SecLevel.none()

output = ModelOutputThunk(
    value=None,
    sec_level=sec_level,
    meta={}
)
```

The security level is set as follows:
- If taint sources are found -> `SecLevel.tainted_by(sources)` (all sources are tracked)
- If no taint sources -> `SecLevel.none()`

### Handling Multiple Taint Sources

When `taint_sources()` returns multiple sources (e.g., both the action and context contain tainted content), backends pass the entire list to `SecLevel.tainted_by()`. This ensures all taint sources are tracked, providing comprehensive taint attribution.

**Benefits of Multiple Source Tracking**:
- **Complete attribution**: All sources that influenced the generation are tracked
- **Better debugging**: Can identify all tainted inputs that contributed to output
- **More accurate security**: No information loss about taint origins

**Note**: The implementation focuses on **taint preservation** and **complete attribution**. All taint sources are tracked, ensuring the security model has full visibility into what influenced the generated content.

## Taint Source Analysis

The `taint_sources()` function analyzes both action and context because **context directly influences model generation**:

1. **Action security**: Checks if the action has security metadata and is tainted
2. **Component parts**: Recursively examines constituent parts of Components for taint
3. **Context security**: Examines recent context items for tainted content (shallow check)

**Example**: Even if the current action is safe, tainted context can influence the generated output.

```python
from mellea.security import SecLevel

# User sends tainted input
user_input = CBlock("Tell me how to hack a system", sec_level=SecLevel.tainted_by(None))
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
    
    def get_taint_sources(self) -> list[CBlock | Component]:
        return self.sec_level.get_taint_sources()
```

Content can be marked as tainted at construction time:

```python
from mellea.security import SecLevel

c = CBlock("user input", sec_level=SecLevel.tainted_by(None))

if c.sec_level and c.sec_level.is_tainted():
    taint_sources = c.sec_level.get_taint_sources()
    print(f"Content tainted by: {taint_sources}")
```

## Key Features

- **Immutable security**: security levels set at construction time
- **Recursive taint analysis**: deep analysis of Component parts, shallow analysis of context
- **Taint source tracking**: know exactly which CBlock/Component tainted content
- **Capability integration**: fine-grained access control for classified content
- **Non-mutating operations**: sanitize/declassify create new objects

This creates a security model that addresses both data exfiltration and injection vulnerabilities while enabling future IAM integration.