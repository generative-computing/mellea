# Metrics Refactor Plan: Migrate to Hooks/Plugins System

## Overview

This plan addresses issues [#607](https://github.com/generative-computing/mellea/issues/607) and [#608](https://github.com/generative-computing/mellea/issues/608) to refactor the token metrics implementation added in commit `0e71558` to use the hooks/plugins system introduced in commit `cbd63bd`.

**Current State**: Token metrics are recorded via direct `record_token_usage_metrics()` calls in each backend's `post_processing()` method (5 backends: OpenAI, Ollama, LiteLLM, HuggingFace, WatsonX).

**Target State**: Token metrics recorded via a plugin that hooks into `generation_post_call`, with token usage data standardized on `ModelOutputThunk` using a `usage` field that matches the OpenAI API standard.

## Key Commits Analysis

### Commit 0e71558 (Metrics Implementation)
- Added `mellea/telemetry/metrics.py` with `record_token_usage_metrics()`
- Modified 5 backends to call `record_token_usage_metrics()` in their `post_processing()` methods
- Token extraction logic varies per backend:
  - **OpenAI/LiteLLM/WatsonX**: Extract from `usage` dict via `get_value(usage, "prompt_tokens")`
  - **Ollama**: Extract from response attributes (`prompt_eval_count`, `eval_count`)
  - **HuggingFace**: Calculate from `input_ids` and output sequences
- Each backend stores usage info in different `_meta` locations

### Commit cbd63bd (Hooks/Plugins System)
- Added complete hooks/plugins infrastructure in `mellea/plugins/`
- Implemented `generation_post_call` hook that fires after `post_process()` completes
- Hook fires in `ModelOutputThunk.astream()` at line 384-398 in `mellea/core/base.py`
- Payload: `GenerationPostCallPayload(prompt, model_output, latency_ms)`
- Policy: `generation_post_call` is **observe-only** (no writable fields)

## Problem Statement

### Issue #607: Standardize Token Storage
**Problem**: Token usage is stored inconsistently across backends in various `_meta` locations, making programmatic access difficult.

**Solution**: Add a standard `usage` field to `ModelOutputThunk` that all backends populate, matching the OpenAI API standard.

### Issue #608: Use Hooks for Metrics
**Problem**: Metrics recording is duplicated across 5 backends, tightly coupled to backend implementation.

**Solution**: Create a metrics plugin that hooks `generation_post_call` to record token usage from the standardized field.

## Architecture Decision

The refactor follows this sequence:

1. **First** (Issue #607): Standardize token storage on `ModelOutputThunk`
2. **Then** (Issue #608): Create metrics plugin to consume standardized data

This ordering is critical because:
- The plugin needs a consistent data source to read from
- Backends must populate the standard field before the hook fires
- The hook fires **after** `post_processing()` completes, so backends have already extracted tokens

## Detailed Implementation Plan

### Phase 1: Standardize Token Storage (Issue #607)

#### 1.1 Add `usage` Field to ModelOutputThunk

**File**: `mellea/core/base.py`

**Changes**:
```python
class ModelOutputThunk(CBlock, Generic[S]):
    def __init__(
        self,
        value: str | None,
        meta: dict[str, Any] | None = None,
        parsed_repr: S | None = None,
        tool_calls: dict[str, ModelToolCall] | None = None,
    ):
        # ... existing code ...
        
        # Add new field for standardized usage information
        self.usage: dict[str, int] | None = None
        """Usage information following OpenAI API standard.
        
        Core fields: 'prompt_tokens', 'completion_tokens', 'total_tokens'.
        Populated by backends during post_processing. None if unavailable.
        
        Future: May include optional breakdown fields like 'completion_tokens_details'
        and 'prompt_tokens_details' for advanced features (reasoning, audio, caching).
        """
```

**Rationale**:
- Matches OpenAI API standard (industry convention)
- Consistent with existing `tool_calls` field pattern
- Extensible for future usage fields beyond tokens (cost, reasoning tokens, cached tokens)
- Simple dict format compatible with all backends
- Optional (None) when backend doesn't provide usage info
- Accessible via `model_output.usage` in hooks and user code

#### 1.2 Update Backend `post_processing()` Methods

Each backend's `post_processing()` method must populate `mot.token_usage` **before** the method returns (since `generation_post_call` hook fires after `post_processing` completes).

**Pattern for all backends**:
```python
async def post_processing(self, mot: ModelOutputThunk, ...):
    # ... existing processing logic ...
    
    # Extract token usage (backend-specific logic)
    prompt_tokens = <extract_prompt_tokens>
    completion_tokens = <extract_completion_tokens>
    
    # Populate standardized field (matches OpenAI API format)
    if prompt_tokens is not None or completion_tokens is not None:
        mot.usage = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        }
    
    # REMOVE: Direct metrics recording
    # from ..telemetry.metrics import record_token_usage_metrics
    # record_token_usage_metrics(...)
```

**Backend-Specific Extraction Logic**:

1. **OpenAI** (`mellea/backends/openai.py:562-646`):
   ```python
   # Extract from response or streaming usage (lines 620-626)
   response = mot._meta["oai_chat_response"]
   usage = response.get("usage") if isinstance(response, dict) else None
   if usage is None:
       usage = mot._meta.get("oai_streaming_usage")
   
   # Populate standardized field (already matches OpenAI format)
   if usage:
       mot.usage = {
           "prompt_tokens": get_value(usage, "prompt_tokens") or 0,
           "completion_tokens": get_value(usage, "completion_tokens") or 0,
           "total_tokens": get_value(usage, "total_tokens") or 0,
       }
   ```

2. **Ollama** (`mellea/backends/ollama.py:583-642`):
   ```python
   # Extract from response attributes (lines 620-623)
   response = mot._meta.get("ollama_response")
   prompt_tokens = getattr(response, "prompt_eval_count", None) if response else None
   completion_tokens = getattr(response, "eval_count", None) if response else None
   
   # Convert to OpenAI-compatible format
   if prompt_tokens is not None or completion_tokens is not None:
       mot.usage = {
           "prompt_tokens": prompt_tokens or 0,
           "completion_tokens": completion_tokens or 0,
           "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
       }
   ```

3. **LiteLLM** (`mellea/backends/litellm.py:425-508`):
   ```python
   # Extract from full response or streaming usage (lines 483-489)
   full_response = mot._meta.get("litellm_full_response")
   usage = full_response.get("usage") if isinstance(full_response, dict) else None
   if usage is None:
       usage = mot._meta.get("litellm_streaming_usage")
   
   # Populate standardized field (already matches OpenAI format)
   if usage:
       mot.usage = {
           "prompt_tokens": get_value(usage, "prompt_tokens") or 0,
           "completion_tokens": get_value(usage, "completion_tokens") or 0,
           "total_tokens": get_value(usage, "total_tokens") or 0,
       }
   ```

4. **HuggingFace** (`mellea/backends/huggingface.py:1001-1092`):
   ```python
   # Calculate from sequences (lines 1063-1076)
   hf_output = mot._meta.get("hf_output")
   if isinstance(hf_output, GenerateDecoderOnlyOutput):
       if input_ids is not None and hf_output.sequences is not None:
           try:
               n_prompt = input_ids.shape[1]
               n_completion = hf_output.sequences[0].shape[0] - n_prompt
               # Convert to OpenAI-compatible format
               mot.usage = {
                   "prompt_tokens": n_prompt,
                   "completion_tokens": n_completion,
                   "total_tokens": n_prompt + n_completion,
               }
           except Exception:
               pass  # Leave as None if calculation fails
   ```

5. **WatsonX** (`mellea/backends/watsonx.py:450-508`):
   ```python
   # Extract from response usage (similar to OpenAI)
   response = mot._meta.get("watsonx_response")
   usage = response.get("usage") if isinstance(response, dict) else None
   
   # Populate standardized field (already matches OpenAI format)
   if usage:
       mot.usage = {
           "prompt_tokens": get_value(usage, "prompt_tokens") or 0,
           "completion_tokens": get_value(usage, "completion_tokens") or 0,
           "total_tokens": get_value(usage, "total_tokens") or 0,
       }
   ```

**Files to Modify**:
- `mellea/core/base.py` (add field)
- `mellea/backends/openai.py` (populate field, remove direct metrics call)
- `mellea/backends/ollama.py` (populate field, remove direct metrics call)
- `mellea/backends/litellm.py` (populate field, remove direct metrics call)
- `mellea/backends/huggingface.py` (populate field, remove direct metrics call)
- `mellea/backends/watsonx.py` (populate field, remove direct metrics call)

### Phase 2: Create Metrics Plugin (Issue #608)

#### 2.1 Create Token Metrics Plugin

**New File**: `mellea/plugins/builtin/token_metrics.py`

**Implementation**:
```python
"""Built-in plugin for recording token usage metrics via OpenTelemetry."""

from mellea.plugins import Plugin, hook, HookType, PluginMode
from mellea.plugins.hooks.generation import GenerationPostCallPayload


class TokenMetricsPlugin(Plugin, name="token-metrics", priority=100):
    """Records token usage metrics from generation_post_call hook.
    
    This plugin automatically records input/output token counts to OpenTelemetry
    metrics when MELLEA_METRICS_ENABLED=true. It reads from the standardized
    ModelOutputThunk.usage field populated by backends.
    
    Execution mode: FIRE_AND_FORGET (async, non-blocking, observe-only)
    Priority: 100 (runs after other plugins)
    """
    
    @hook(HookType.GENERATION_POST_CALL, mode=PluginMode.FIRE_AND_FORGET)
    async def record_tokens(
        self,
        payload: GenerationPostCallPayload,
        ctx
    ):
        """Record token usage metrics from the model output."""
        from mellea.telemetry.metrics import is_metrics_enabled
        
        # Early return if metrics disabled (zero overhead)
        if not is_metrics_enabled():
            return
        
        mot = payload.model_output
        if mot is None or mot.usage is None:
            return
        
        # Extract backend info from context
        backend = ctx.global_context.state.get("backend")
        if backend is None:
            return
        
        from mellea.telemetry.backend_instrumentation import (
            get_model_id_str,
            get_system_name,
        )
        from mellea.telemetry.metrics import record_token_usage_metrics
        
        # Record using standardized usage dict
        record_token_usage_metrics(
            input_tokens=mot.usage.get("prompt_tokens"),
            output_tokens=mot.usage.get("completion_tokens"),
            model=get_model_id_str(backend),
            backend=backend.__class__.__name__,
            system=get_system_name(backend),
        )
```

**Key Design Decisions**:
- **FIRE_AND_FORGET mode**: Metrics recording is async, non-blocking, and cannot fail the generation
- **Priority 100**: Runs after other plugins (lower priority = earlier execution)
- **Observe-only**: Reads from `model_output.usage`, doesn't modify payload
- **Zero overhead**: Early return when metrics disabled
- **Backend-agnostic**: Works with any backend that populates `usage`
- **OpenAI-compatible**: Uses standard field name for familiarity and future extensibility

#### 2.2 Auto-Register Plugin When Metrics Enabled

**File**: `mellea/telemetry/metrics.py`

**Changes**:
```python
# At module initialization (after _meter setup, around line 247)
if _OTEL_AVAILABLE and _METRICS_ENABLED:
    _meter_provider = _setup_meter_provider()
    if _meter_provider is not None:
        _meter = metrics.get_meter("mellea.metrics", version("mellea"))
        
        # Auto-register token metrics plugin
        from mellea.plugins.builtin.token_metrics import TokenMetricsPlugin
        from mellea.plugins import register
        
        _token_metrics_plugin = TokenMetricsPlugin()
        register(_token_metrics_plugin)
```

**Rationale**:
- Automatic activation when `MELLEA_METRICS_ENABLED=true`
- No user code changes required
- Consistent with existing metrics module behavior
- Plugin is globally registered (works with both session and functional API)

#### 2.3 Update Plugin Module Structure

**New Directory**: `mellea/plugins/builtin/`

**Files**:
- `mellea/plugins/builtin/__init__.py` - Export built-in plugins
- `mellea/plugins/builtin/token_metrics.py` - Token metrics plugin

**Update**: `mellea/plugins/__init__.py`
```python
# Add to exports
from mellea.plugins.builtin import TokenMetricsPlugin

__all__ = [
    # ... existing exports ...
    "TokenMetricsPlugin",
]
```

### Phase 3: Update Tests

#### 3.1 Update Backend Integration Tests

**Files**: `test/telemetry/test_metrics_backend.py`

**Changes**:
- Tests should verify `mot.token_usage` is populated (not just metrics recorded)
- Add assertions: `assert mot.token_usage is not None`
- Add assertions: `assert mot.token_usage["prompt_tokens"] > 0`
- Keep existing metrics verification (plugin should still record them)

**Example**:
```python
async def test_openai_token_metrics(ollama_backend):
    """Test that OpenAI backend populates usage and metrics are recorded."""
    # ... existing test setup ...
    
    # Verify usage field is populated
    assert mot.usage is not None
    assert mot.usage["prompt_tokens"] > 0
    assert mot.usage["completion_tokens"] > 0
    assert mot.usage["total_tokens"] > 0
    
    # Verify metrics were recorded (via plugin)
    # ... existing metrics verification ...
```

#### 3.2 Add Plugin-Specific Tests

**New File**: `test/plugins/test_token_metrics_plugin.py`

**Tests**:
1. Test plugin registers correctly when metrics enabled
2. Test plugin reads from `mot.usage` correctly
3. Test plugin handles missing `usage` gracefully
4. Test plugin is no-op when metrics disabled
5. Test plugin works with all backends
6. Test plugin doesn't block on errors (FIRE_AND_FORGET mode)

#### 3.3 Update Unit Tests

**File**: `test/telemetry/test_metrics_token.py`

**Changes**:
- Tests for `record_token_usage_metrics()` remain unchanged (function still exists)
- Add note that function is now called by plugin, not backends directly

### Phase 4: Update Documentation

#### 4.1 Update Telemetry Documentation

**File**: `docs/dev/telemetry.md`

**Changes**:
- Update "Token Usage Metrics" section to explain plugin-based architecture
- Add note about `ModelOutputThunk.usage` field (OpenAI-compatible)
- Update backend support table to note standardized field
- Add example of accessing token usage programmatically

**Example Addition**:
```markdown
### Programmatic Access to Token Usage

Token usage information is available on the `ModelOutputThunk` after generation:

```python
with start_session() as m:
    result = m.instruct("What is the capital of France?")
    
    # Access token usage (OpenAI-compatible format)
    if result.usage:
        print(f"Prompt tokens: {result.usage['prompt_tokens']}")
        print(f"Completion tokens: {result.usage['completion_tokens']}")
        print(f"Total tokens: {result.usage['total_tokens']}")
```

Token metrics are automatically recorded to OpenTelemetry when 
`MELLEA_METRICS_ENABLED=true` via the built-in `TokenMetricsPlugin`.
```

#### 4.2 Update Plugin Documentation

**File**: `docs/docs/core-concept/plugins.mdx`

**Changes**:
- Add section on built-in plugins
- Document `TokenMetricsPlugin` as an example of FIRE_AND_FORGET mode
- Show how built-in plugins auto-register

#### 4.3 Update Examples

**File**: `docs/examples/telemetry/metrics_example.py`

**Changes**:
- Add example showing programmatic access to `usage`
- Add comment explaining plugin-based architecture
- No functional changes (metrics still work the same way)

**Example Addition**:
```python
# Example 5: Programmatic token access
print("\n5. Accessing token usage programmatically...")
result = m.instruct("Count to five")
if result.usage:
    print(f"  Prompt tokens: {result.usage['prompt_tokens']}")
    print(f"  Completion tokens: {result.usage['completion_tokens']}")
    print(f"  Total tokens: {result.usage['total_tokens']}")
```

### Phase 5: Cleanup

#### 5.1 Remove Backend-Specific Metrics Code

**All Backend Files**:
- Remove `from ..telemetry.metrics import is_metrics_enabled` imports
- Remove `from ..telemetry.metrics import record_token_usage_metrics` imports
- Remove `from ..telemetry.backend_instrumentation import get_model_id_str, get_system_name` imports (if only used for metrics)
- Remove `if is_metrics_enabled():` blocks and `record_token_usage_metrics()` calls

**Estimated Lines Removed**: ~15-20 lines per backend × 5 backends = ~75-100 lines

#### 5.2 Keep Core Metrics Infrastructure

**Files to Keep Unchanged**:
- `mellea/telemetry/metrics.py` - Core metrics functions still needed by plugin
- `mellea/telemetry/backend_instrumentation.py` - Helper functions still needed

## Implementation Order

### Step 1: Add `token_usage` Field (Issue #607)
1. Modify `mellea/core/base.py` to add `token_usage` field
2. Update all 5 backends to populate the field
3. Keep existing `record_token_usage_metrics()` calls temporarily (dual-write)
4. Run tests to verify field is populated correctly

### Step 2: Create Metrics Plugin (Issue #608)
5. Create `mellea/plugins/builtin/` directory
6. Implement `TokenMetricsPlugin` in `token_metrics.py`
7. Auto-register plugin in `mellea/telemetry/metrics.py`
8. Run tests to verify plugin records metrics correctly

### Step 3: Remove Duplicate Code
9. Remove direct `record_token_usage_metrics()` calls from all backends
10. Remove unused imports from backends
11. Run full test suite to verify no regressions

### Step 4: Update Documentation and Tests
12. Update backend integration tests to verify `token_usage` field
13. Add plugin-specific tests
14. Update documentation (telemetry.md, plugins.mdx)
15. Update examples to show programmatic access

## Testing Strategy

### Test Coverage Required

1. **Unit Tests** (existing, should pass unchanged):
   - `test/telemetry/test_metrics_token.py` - Tests for `record_token_usage_metrics()`

2. **Backend Integration Tests** (need updates):
   - `test/telemetry/test_metrics_backend.py` - Add `token_usage` field assertions
   - All 5 backend tests should verify field population

3. **Plugin Tests** (new):
   - `test/plugins/test_token_metrics_plugin.py` - Plugin-specific tests
   - Test plugin registration, execution, error handling

4. **End-to-End Tests** (existing, should pass):
   - `docs/examples/telemetry/metrics_example.py` - Should work unchanged

### Test Execution

```bash
# Run metrics tests
uv run pytest test/telemetry/test_metrics*.py -v

# Run plugin tests
uv run pytest test/plugins/test_token_metrics_plugin.py -v

# Run backend tests
uv run pytest test/backends/ -k "metrics" -v

# Run example as test
uv run pytest docs/examples/telemetry/metrics_example.py -v
```

## Migration Impact

### Breaking Changes
**None** - This is a pure refactor with no API changes:
- Metrics still recorded automatically when `MELLEA_METRICS_ENABLED=true`
- Same metric names, attributes, and exporters
- User code unchanged

### New Features
- **Programmatic access**: Users can now access `mot.token_usage` directly
- **Extensibility**: Users can create custom metrics plugins following this pattern
- **Consistency**: Token data in standard location across all backends

### Performance Impact
- **Negligible**: Plugin uses FIRE_AND_FORGET mode (async, non-blocking)
- **Same overhead**: Metrics recording logic unchanged, just moved to plugin
- **Zero overhead when disabled**: Early return in plugin when metrics disabled

## Risks and Mitigations

### Risk 1: Hook Timing
**Risk**: `generation_post_call` fires after `post_processing()`, but what if backends don't populate `token_usage` in time?

**Mitigation**: Backends populate `token_usage` **during** `post_processing()`, which completes **before** the hook fires. The hook call is at line 384-398 in `ModelOutputThunk.astream()`, after line 366 (`await self._post_process(self)`).

### Risk 2: Missing Token Data
**Risk**: Some backends might not have token usage available.

**Mitigation**: 
- `token_usage` field is optional (None when unavailable)
- Plugin checks `if mot.token_usage is None: return`
- Existing behavior preserved (metrics not recorded when data unavailable)

### Risk 3: Plugin Framework Dependency
**Risk**: Plugin requires `mellea[hooks]` extra dependency.

**Mitigation**:
- Plugin only imported when metrics enabled
- Graceful fallback if hooks not installed (warning message)
- Most users enabling metrics will have full installation

### Risk 4: Test Failures
**Risk**: Existing tests might fail if they expect metrics without the plugin.

**Mitigation**:
- Plugin auto-registers when metrics enabled (same as before)
- Tests run with `MELLEA_METRICS_ENABLED=true` will get plugin automatically
- Dual-write during Step 1 ensures no test breakage during transition

## Success Criteria

### Phase 1 Complete When:
- [ ] `ModelOutputThunk.token_usage` field added
- [ ] All 5 backends populate the field correctly
- [ ] Backend integration tests verify field population
- [ ] Existing metrics tests still pass (dual-write active)

### Phase 2 Complete When:
- [ ] `TokenMetricsPlugin` implemented and tested
- [ ] Plugin auto-registers when metrics enabled
- [ ] Plugin records metrics correctly (verified by existing tests)
- [ ] Plugin-specific tests added and passing

### Phase 3 Complete When:
- [ ] Direct `record_token_usage_metrics()` calls removed from backends
- [ ] Unused imports cleaned up
- [ ] All tests pass (metrics now recorded via plugin only)

### Phase 4 Complete When:
- [ ] Documentation updated (telemetry.md, plugins.mdx)
- [ ] Examples updated to show programmatic access
- [ ] All documentation builds without errors

### Overall Success:
- [ ] All tests pass (`uv run pytest test/`)
- [ ] No breaking changes to user API
- [ ] Metrics still recorded correctly
- [ ] Code is cleaner (no duplication across backends)
- [ ] Token usage accessible programmatically via `mot.token_usage`

## File Checklist

### Files to Create
- [ ] `mellea/plugins/builtin/__init__.py`
- [ ] `mellea/plugins/builtin/token_metrics.py`
- [ ] `test/plugins/test_token_metrics_plugin.py`

### Files to Modify
- [ ] `mellea/core/base.py` - Add `token_usage` field
- [ ] `mellea/backends/openai.py` - Populate field, remove metrics call
- [ ] `mellea/backends/ollama.py` - Populate field, remove metrics call
- [ ] `mellea/backends/litellm.py` - Populate field, remove metrics call
- [ ] `mellea/backends/huggingface.py` - Populate field, remove metrics call
- [ ] `mellea/backends/watsonx.py` - Populate field, remove metrics call
- [ ] `mellea/telemetry/metrics.py` - Auto-register plugin
- [ ] `mellea/plugins/__init__.py` - Export TokenMetricsPlugin
- [ ] `test/telemetry/test_metrics_backend.py` - Add token_usage assertions
- [ ] `docs/dev/telemetry.md` - Document new architecture
- [ ] `docs/docs/core-concept/plugins.mdx` - Document built-in plugin
- [ ] `docs/examples/telemetry/metrics_example.py` - Add programmatic access example

### Files to Review (No Changes Expected)
- `mellea/telemetry/backend_instrumentation.py` - Helper functions still used
- `test/telemetry/test_metrics_token.py` - Unit tests for core function
- `mellea/plugins/hooks/generation.py` - Payload definition unchanged

## Estimated Effort

- **Phase 1** (Standardize storage): 2-3 hours
  - Add field: 15 min
  - Update 5 backends: 1.5 hours (30 min each)
  - Update tests: 1 hour

- **Phase 2** (Create plugin): 1-2 hours
  - Implement plugin: 45 min
  - Auto-registration: 15 min
  - Plugin tests: 1 hour

- **Phase 3** (Cleanup): 30 min
  - Remove duplicate code: 20 min
  - Verify tests: 10 min

- **Phase 4** (Documentation): 1 hour
  - Update docs: 30 min
  - Update examples: 30 min

**Total**: 4.5-6.5 hours

## Open Questions

1. **Should `token_usage` be added to `GenerationPostCallPayload`?**
   - Pro: Makes it explicit in the hook payload
   - Con: Redundant (already on `model_output`)
   - **Recommendation**: No, keep it on MOT only (single source of truth)

2. **Should the plugin be optional or always registered?**
   - Current plan: Auto-register when `MELLEA_METRICS_ENABLED=true`
   - Alternative: Always register, let plugin check `is_metrics_enabled()` internally
   - **Recommendation**: Auto-register (cleaner, no overhead when disabled)

3. **Should we support custom metrics plugins?**
   - Users could create their own plugins reading from `token_usage`
   - **Recommendation**: Yes, document this pattern in plugins.mdx

4. **What about vLLM backend?**
   - Not modified in commit 0e71558 (no metrics support)
   - **Recommendation**: Out of scope for this refactor, can be added later

## References

- Issue #607: https://github.com/generative-computing/mellea/issues/607
- Issue #608: https://github.com/generative-computing/mellea/issues/608
- PR #563 (metrics): https://github.com/generative-computing/mellea/pull/563
- PR #582 (hooks): https://github.com/generative-computing/mellea/pull/582
- Commit 0e71558: Metrics implementation
- Commit cbd63bd: Hooks/plugins system
- Hook system spec: `docs/dev/hook_system.md`
- Plugin examples: `docs/examples/plugins/`