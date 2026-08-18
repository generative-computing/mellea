# Migration Guide: `_call_tools` → `call_tools` (Public API)

As of this release, `_call_tools()` and `_acall_tools()` have been promoted from
private functions to a stable public API: `call_tools()` and `acall_tools()`.

## Overview

The function signatures and behavior remain **unchanged** — this is a pure
naming/visibility upgrade. You can migrate at your own pace; the old names
continue to work with no deprecation warnings (yet).

## Migration Steps

### Step 1: Update imports

Replace underscore-prefixed imports with the public names:

```python
# ❌ Old (private API)
from mellea.stdlib.functional import _call_tools, _acall_tools

# ✓ New (public API)
from mellea.stdlib.functional import call_tools, acall_tools
```

### Step 2: Update function calls

Replace function calls throughout your code:

```python
# ❌ Old
tool_messages = _call_tools(result, backend)

# ✓ New
tool_messages = call_tools(result, backend)
```

```python
# ❌ Old
tool_messages = await _acall_tools(result, backend)

# ✓ New
tool_messages = await acall_tools(result, backend)
```

## Timeline

| Phase | Date | Action |
| --- | --- | --- |
| Current | Now | New public API; old names work as aliases |
| Deprecation (2-3 releases) | TBD | Add `DeprecationWarning` to old names |
| Removal (next major) | TBD | Remove `_call_tools` and `_acall_tools` |

## Why the Change?

- **Stability**: These functions are fundamental to Mellea's extensibility
- **Discoverability**: Public functions appear in generated API docs and IDE
  autocompletion
- **Commitment**: Public APIs get stability guarantees; private ones can change
  anytime

## Backward Compatibility

For now, `_call_tools` and `_acall_tools` continue to work:

```python
# Still works (but migrate when convenient)
from mellea.stdlib.functional import _call_tools
result = _call_tools(mot, backend)
```

However, we recommend migrating during your next code review cycle to ensure
your documentation and examples are up-to-date.

## Documentation

After migration, see:

- [How-To: Execute Tool Calls](docs/how-to/execute-tool-calls.md) — Reference
  and patterns
- [How-To: Choosing Primitives vs High-Level APIs](docs/how-to/primitives-vs-high-level.md)
  — When to use `call_tools()`
- [GitHub Discussion #1460](https://github.com/generative-computing/mellea/discussions/1460)
  — Design discussion and rationale

## Examples

### Basic Migration

**Before:**

```python
from mellea.stdlib.functional import _call_tools

tool_messages = _call_tools(result, backend)
```

**After:**

```python
from mellea.stdlib.functional import call_tools

tool_messages = call_tools(result, backend)
```

### Async Migration

**Before:**

```python
from mellea.stdlib.functional import _acall_tools

tool_messages = await _acall_tools(result, backend)
```

**After:**

```python
from mellea.stdlib.functional import acall_tools

tool_messages = await acall_tools(result, backend)
```

## Questions?

- Check the new documentation in `docs/docs/how-to/execute-tool-calls.md`
- See examples in `docs/examples/primitives/`
- Open an issue on GitHub
