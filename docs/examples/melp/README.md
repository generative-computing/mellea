# MELP (Mellea Language Programming) Examples

This directory contains examples of MELP - an experimental lazy evaluation system for Mellea programs.

## Files

### simple_example.py
Basic introduction to MELP's lazy evaluation concepts.

### lazy.py
Core lazy evaluation patterns and primitives.

### lazy_fib.py
Demonstrates lazy evaluation with Fibonacci sequence generation.

### lazy_fib_sample.py
Sampling and evaluation strategies with lazy computations.

### states.py
State management in lazy evaluation contexts.

## Concepts Demonstrated

- **Lazy Evaluation**: Deferring computation until results are needed
- **Thunks**: Suspended computations that can be evaluated later
- **State Management**: Handling state in lazy evaluation contexts
- **Sampling Strategies**: Combining lazy evaluation with sampling
- **Composability**: Building complex lazy programs from simple parts

## What is MELP?

MELP is an experimental feature that brings lazy evaluation to Mellea programs. It allows you to:
- Define computations without immediately executing them
- Compose complex workflows declaratively
- Optimize execution by avoiding unnecessary computations
- Implement advanced control flow patterns

## Basic Pattern

```python
# Define lazy computations
lazy_result = lazy_function(args)

# Compose lazy computations
composed = lazy_map(transform, lazy_result)

# Force evaluation when needed
actual_result = force(composed)
```

## Status

⚠️ **Experimental**: MELP is an experimental feature and APIs may change. Use with caution in production code.

## Related Documentation

- See `mellea/stdlib/functional.py` for functional programming primitives
- See `docs/dev/mellea_library.md` for design philosophy