# Context Examples

This directory contains examples demonstrating how to work with Mellea's context system, particularly when using sampling strategies and validation.

## Files

### contexts_with_sampling.py
Shows how to retrieve and inspect context information when using sampling strategies and validation.

**Key Features:**
- Using `RejectionSamplingStrategy` with requirements
- Accessing `SamplingResult` objects to inspect generation attempts
- Retrieving context for different generation attempts
- Examining validation contexts for each requirement
- Understanding the context tree structure

**Usage:**
```bash
python docs/examples/context/contexts_with_sampling.py
```

## Concepts Demonstrated

- **Sampling Results**: Working with `SamplingResult` objects
- **Context Inspection**: Accessing generation and validation contexts
- **Multiple Attempts**: Examining different generation attempts
- **Context Trees**: Understanding how contexts link together
- **Validation Context**: Inspecting how requirements were evaluated

## Key APIs

```python
# Get sampling result with full context information
res = m.instruct(
    "Write a sentence.",
    requirements=[...],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True
)

# Access different generation attempts
res.sample_generations[index]
res.sample_contexts[index]
res.sample_validations[index]

# Navigate context tree
gen_ctx.previous_node.node_data
val_ctx.node_data
```

## Related Documentation

- See `mellea/stdlib/context.py` for context implementation
- See `mellea/stdlib/sampling/` for sampling strategies
- See `docs/dev/spans.md` for context architecture details