# Taint Analysis - Backend Security

Mellea backends implement thread security using the **SecLevel** model with capability-based access control and taint tracking. Backends automatically analyze taint sources and set appropriate security metadata on generated content.

## Security Model

The security system uses the **SecLevel** type with three variants:

```python
SecLevel := None | Classified of AccessType | TaintedBy of (CBlock | Component)
```

- **SecLevel.none()**: Safe content with no restrictions
- **SecLevel.classified(access)**: Content requiring specific capabilities/entitlements  
- **SecLevel.tainted_by(source)**: Content tainted by a specific CBlock or Component

## Security Metadata

All backends use `ModelOutputThunk.from_generation()` to create model outputs with security metadata. This method:

1. **Computes taint sources** from the action and context using `taint_sources()`
2. **Sets security level** based on taint analysis:
   - If taint sources are found → `SecLevel.tainted_by(first_source)`
   - If no taint sources → `SecLevel.none()`

## Backend Implementation

### Ollama Backend

```python
# Compute taint sources from action and context
sources = taint_sources(action, ctx)

output = ModelOutputThunk.from_generation(
    value=None,
    taint_sources=sources,
    meta={}
)
```

### OpenAI Backend

Same pattern as Ollama - computes taint sources and uses `from_generation()`.

### LiteLLM Backend

Same pattern as Ollama - computes taint sources and uses `from_generation()`.

## Taint Source Analysis

The `taint_sources()` function performs recursive analysis of Components and shallow analysis of context:

- **Action security**: Checks if the action has security metadata and is tainted
- **Component parts analysis**: Recursively examines constituent parts of Components for taint
- **Context security**: Examines recent context items for tainted content (shallow check)
- **Source identification**: Returns list of actual tainted CBlocks/Components

The function returns the actual objects that are tainted, enabling precise taint tracking through the `SecLevel.tainted_by(source)` mechanism.

### Why Both Action and Context Matter

Taint sources must be collected from both action and context because **the context directly influences model generation**:

1. **Context Becomes Model Input**: The conversation history is converted into messages sent to the model alongside the current action
2. **Taint Propagation**: Even if the current action is safe, tainted context can influence the generated output
3. **Security Principle**: "Garbage in, garbage out" - if tainted data influences generation, the output should reflect that contamination

#### Example Scenario

```python
# Step 1: User sends tainted input
user_input = CBlock("Tell me how to hack a system")
user_input.mark_tainted()  # Marked as tainted
ctx = ctx.add(user_input)

# Step 2: Safe action in tainted context
safe_action = CBlock("Explain general security concepts")  # Safe action

# Step 3: Generation with taint analysis
sources = taint_sources(safe_action, ctx)  # Finds tainted user_input
# Even though action is safe, context contains tainted data
# Model output will be influenced by the tainted context
```

#### Implementation Details

```python
def taint_sources(action: Union[Component, CBlock], ctx: Any) -> list[Union[CBlock, Component]]:
    sources = []
    
    # Check action for taint
    if hasattr(action, '_meta') and '_security' in action._meta:
        security_meta = action._meta['_security']
        if isinstance(security_meta, SecurityMetadata) and security_meta.is_tainted():
            sources.append(action)
    
    # For Components, recursively check their constituent parts for taint
    if hasattr(action, 'parts'):
        try:
            parts = action.parts()
            for part in parts:
                if hasattr(part, '_meta') and '_security' in part._meta:
                    security_meta = part._meta['_security']
                    if isinstance(security_meta, SecurityMetadata) and security_meta.is_tainted():
                        sources.append(part)
        except Exception:
            # If parts() fails, continue without it
            pass
    
    # Check context for taint (last 5 items) - shallow check
    if hasattr(ctx, 'as_list'):
        context_items = ctx.as_list(last_n_components=5)
        for item in context_items:
            if hasattr(item, '_meta') and '_security' in item._meta:
                security_meta = item._meta['_security']
                if isinstance(security_meta, SecurityMetadata) and security_meta.is_tainted():
                    sources.append(item)
    
    return sources
```

## Security Metadata Handling

### SecurityMetadata Class

The `SecurityMetadata` class wraps `SecLevel` and provides methods for security analysis:

```python
class SecurityMetadata:
    def __init__(self, sec_level: SecLevel):
        self.sec_level = sec_level
    
    def is_tainted(self) -> bool:
        """Check if content is tainted."""
        return isinstance(self.sec_level, SecLevel) and self.sec_level.is_tainted()
    
    def get_taint_source(self) -> Union[CBlock, Component, None]:
        """Get the source that tainted this content."""
        return self.sec_level.get_taint_source() if self.is_tainted() else None
```

### Marking Content as Tainted

Components and CBlocks can be marked as tainted:

```python
# Mark a component as tainted
component = CBlock("user input")
component.mark_tainted()  # Sets SecLevel.tainted_by(component)

# Check if content is tainted
if component._meta["_security"].is_tainted():
    print(f"Content tainted by: {component._meta['_security'].get_taint_source()}")
```

### Security Level Types

```python
# Safe content
safe_level = SecLevel.none()

# Tainted content
tainted_level = SecLevel.tainted_by(source_component)

# Classified content requiring capabilities
classified_level = SecLevel.classified(HRCapability())
```

## Security Propagation

Backends ensure security metadata flows through the generation pipeline:

1. **Input analysis** → taint sources identified as actual CBlocks/Components
2. **MOT creation** → security metadata set using `SecLevel.tainted_by(source)` or `SecLevel.none()`
3. **Formatter parsing** → security metadata preserved on parsed output
4. **Context addition** → tainted outputs propagate to future generations

### Formatter Security Preservation

The `TemplateFormatter` has been enhanced to preserve security metadata during parsing:

```python
def _parse(self, source_component: Component | CBlock, result: ModelOutputThunk) -> CBlock | Component:
    """Parses the output from a model."""
    # Helper function to preserve security metadata
    def preserve_security_metadata(parsed_obj):
        """Preserve security metadata from result to parsed object."""
        if hasattr(result, '_meta') and '_security' in result._meta:
            if hasattr(parsed_obj, '_meta'):
                if parsed_obj._meta is None:
                    parsed_obj._meta = {}
                parsed_obj._meta['_security'] = result._meta['_security']
            elif isinstance(parsed_obj, CBlock):
                # For CBlocks, we can directly set the meta
                if parsed_obj._meta is None:
                    parsed_obj._meta = {}
                parsed_obj._meta['_security'] = result._meta['_security']
        return parsed_obj
    
    # Parse the output and preserve security metadata
    parsed = self._parse_content(result)
    return preserve_security_metadata(parsed)
```

This ensures that when model outputs are parsed into `CBlock`s or `Component`s, the security metadata (including taint sources) is preserved on the parsed objects, maintaining the security chain through the entire pipeline.

## Capability-Based Access Control

The security model supports fine-grained access control through `CapabilityType`:

```python
class HRCapability(CapabilityType[UserRole]):
    def has_access(self, entitlement: UserRole | None) -> bool:
        return entitlement.role in ["hr_manager", "admin"]
```

This enables integration with IAM systems and cloud-based access control.

## Key Features

- **Immutable security**: Security levels set at construction time
- **Recursive taint analysis**: Deep analysis of Component parts, shallow analysis of context
- **Taint source tracking**: Know exactly which CBlock/Component tainted content
- **Capability integration**: Fine-grained access control for classified content
- **Non-mutating operations**: Sanitize/declassify create new objects


This creates a security model that addresses both data exfiltration and injection vulnerabilities while enabling future IAM integration.