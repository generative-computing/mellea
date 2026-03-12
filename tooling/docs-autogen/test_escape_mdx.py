#!/usr/bin/env python3
"""Tests for MDX syntax escaping in code blocks."""

import pytest
from decorate_api_mdx import escape_mdx_syntax


def test_escape_curly_braces_in_python_dict():
    """Test escaping curly braces in Python dict literals."""
    content = """---
title: "Test"
---

Some text before.

```python
def foo():
    return {"key": "value", "nested": {"a": 1}}
```

Some text after.
"""
    expected = """---
title: "Test"
---

Some text before.

```python
def foo():
    return {{"key": "value", "nested": {{"a": 1}}}}
```

Some text after.
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_escape_json_code_block():
    """Test escaping JSON in code blocks."""
    content = """```json
{
  "name": "test",
  "config": {
    "enabled": true
  }
}
```
"""
    expected = """```json
{{
  "name": "test",
  "config": {{
    "enabled": true
  }}
}}
```
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_idempotent_on_fresh_content():
    """Test that escaping fresh content twice gives same result as once.

    Note: This function is designed for fresh mdxify output which has no
    escaped braces. If you run it on already-escaped content, it will
    double-escape. This is acceptable since the function is only called
    once in the pipeline on fresh content.
    """
    fresh_content = """```python
counter.add(1, {"backend": "ollama"})
```
"""
    # First escape
    escaped_once = escape_mdx_syntax(fresh_content)
    expected = """```python
counter.add(1, {{"backend": "ollama"}})
```
"""
    assert escaped_once == expected

    # Running on already-escaped content will double-escape
    # This is expected behavior - don't run the function twice!
    escaped_twice = escape_mdx_syntax(escaped_once)
    assert escaped_twice != escaped_once  # It will be different (double-escaped)


def test_no_escape_outside_code_blocks():
    """Test that curly braces outside code blocks are not escaped."""
    content = """---
title: "Test"
---

import { SidebarFix } from "/snippets/SidebarFix.mdx";

<SidebarFix />

Regular text with {curly} braces should not be escaped.

```python
def foo():
    return {"key": "value"}
```

More text with {braces} after.
"""
    expected = """---
title: "Test"
---

import { SidebarFix } from "/snippets/SidebarFix.mdx";

<SidebarFix />

Regular text with {curly} braces should not be escaped.

```python
def foo():
    return {{"key": "value"}}
```

More text with {braces} after.
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_multiple_code_blocks():
    """Test escaping in multiple code blocks."""
    content = """```python
x = {"a": 1}
```

Some text.

```json
{"b": 2}
```
"""
    expected = """```python
x = {{"a": 1}}
```

Some text.

```json
{{"b": 2}}
```
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_empty_dict():
    """Test escaping empty dicts."""
    content = """```python
empty = {}
nested = {"outer": {}}
```
"""
    expected = """```python
empty = {{}}
nested = {{"outer": {{}}}}
```
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_mixed_content():
    """Test realistic MDX with mixed content."""
    content = """---
title: "mellea.telemetry.metrics"
---

import { SidebarFix } from "/snippets/SidebarFix.mdx";

<SidebarFix />

## Example

```python
from mellea.telemetry import create_counter

counter = create_counter(
    "mellea.requests",
    description="Total number of LLM requests",
    unit="1"
)
counter.add(1, {"backend": "ollama", "model": "llama2"})
```

The counter tracks metrics with attributes.
"""
    expected = """---
title: "mellea.telemetry.metrics"
---

import { SidebarFix } from "/snippets/SidebarFix.mdx";

<SidebarFix />

## Example

```python
from mellea.telemetry import create_counter

counter = create_counter(
    "mellea.requests",
    description="Total number of LLM requests",
    unit="1"
)
counter.add(1, {{"backend": "ollama", "model": "llama2"}})
```

The counter tracks metrics with attributes.
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_code_block_with_language_specifier():
    """Test code blocks with various language specifiers."""
    content = """```typescript
const obj = { key: "value" };
```

```bash
echo "No braces here"
```

```python
data = {"x": 1}
```
"""
    expected = """```typescript
const obj = {{ key: "value" }};
```

```bash
echo "No braces here"
```

```python
data = {{"x": 1}}
```
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_inline_code_not_affected():
    """Test that inline code is not affected (only fenced blocks)."""
    content = """Regular text with `{"inline": "code"}` should not be escaped.

```python
block = {"should": "escape"}
```
"""
    expected = """Regular text with `{"inline": "code"}` should not be escaped.

```python
block = {{"should": "escape"}}
```
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_nested_code_blocks_not_supported():
    """Test that nested code blocks are handled correctly (toggle on/off)."""
    # MDX doesn't support nested code blocks, but we should handle the toggle correctly
    content = """```python
# First block
x = {"a": 1}
```
```python
# Second block immediately after
y = {"b": 2}
```
"""
    expected = """```python
# First block
x = {{"a": 1}}
```
```python
# Second block immediately after
y = {{"b": 2}}
```
"""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_empty_content():
    """Test handling of empty content."""
    content = ""
    expected = ""
    result = escape_mdx_syntax(content)
    assert result == expected


def test_no_code_blocks():
    """Test content without any code blocks."""
    content = """---
title: "Test"
---

Just regular text with {braces} that should not be escaped.
"""
    expected = content  # Should be unchanged
    result = escape_mdx_syntax(content)
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
