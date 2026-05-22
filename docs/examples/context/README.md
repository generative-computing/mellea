# Context Examples

This directory contains examples demonstrating how to work with Mellea's context system: inspecting per-attempt contexts produced by sampling strategies, and shrinking contexts with the `Compactor` protocol.

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

```
python docs/examples/context/contexts_with_sampling.py
```

### window_compactor.py

`WindowCompactor` — opt-in by passing `compactor=` (or the `window_size=` sugar). Demonstrates system-prefix pinning, `pin_system_and_initial_user`, `pin_nothing` (pure last-N), and `size=0` to clear the body.

### threshold_compactor.py

`ThresholdCompactor` — gate an inner compactor on the conversation's cumulative token size. The reading is taken from the most recent `ModelOutputThunk`'s `total_tokens`, which for a chat backend equals `prompt_tokens` (full conversation history sent to the model) + `completion_tokens` (reply). The gate fires once the running conversation size crosses the threshold; once compaction shrinks the context, the next call produces a smaller reading and the gate closes again.

### custom_compactor.py

Implement the `Compactor` protocol with a plain class (no inheritance). Shows Pattern 1 (wired into `ChatContext`) and Pattern 2 (manual `compact()` call).

### react_compaction.py

Compose the ReACT loop with a sync `Compactor`. Two integration points:

- **Per-add** — wire a `Compactor` onto the `ChatContext` so it runs every time `react()` appends a Message, ToolMessage, or thunk.
- **Per-turn** — pass `compactor=` to `react()`; it fires once per ReACT iteration after the tool observation.

`LLMSummarizeCompactor` is also a sync `Compactor` — it hides the async backend call internally (worker thread when called from an already-running event loop) so callers don't have to think about sync vs async.

Use `pin_react_initiator` (from `mellea.stdlib.components.react`) as the predicate so the goal and tool registration survive compaction.

## Concepts Demonstrated

- **Sampling Results**: Working with `SamplingResult` objects
- **Context Inspection**: Accessing generation and validation contexts
- **Multiple Attempts**: Examining different generation attempts
- **Context Trees**: Understanding how contexts link together
- **Validation Context**: Inspecting how requirements were evaluated
- **Compaction Protocol**: Sync `Compactor` for per-`add()` shrinking
- **Pin Predicates**: Auto-protect leading system messages or the user's initial prompt during compaction

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

```python
# Wire a compactor into a ChatContext (Pattern 1 — runs on every add())
from mellea.stdlib.context import ChatContext, WindowCompactor, ThresholdCompactor

ctx = ChatContext(compactor=WindowCompactor(size=5))            # default: pin_system
ctx = ChatContext(window_size=5)                                # sugar for the line above
ctx = ChatContext(
    compactor=ThresholdCompactor(WindowCompactor(size=5), threshold=8000),
)

# Manual compaction (Pattern 2)
ctx = WindowCompactor(size=0).compact(ctx)                      # drop body, keep pinned prefix
```

## Related Documentation

- See `mellea/stdlib/context/` for context and compactor implementations
- See `mellea/stdlib/sampling/` for sampling strategies
- See `mellea/stdlib/frameworks/react.py` for the ReACT loop
- See `docs/dev/spans.md` for context architecture details
