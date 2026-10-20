# Context Examples

Examples demonstrating context tree inspection with sampling strategies and context shrinking using the `Compactor` protocol.

## Examples

| File | Description |
| :--- | :--- |
| [`contexts_with_sampling.py`](./contexts_with_sampling.py) | Inspecting per-attempt generation contexts, validation contexts, and context tree navigation in sampling results. |
| [`window_compactor.py`](./window_compactor.py) | Sliding window compaction with system-prefix pinning and custom pin predicates. |
| [`threshold_compactor.py`](./threshold_compactor.py) | Gating window compaction on cumulative conversation token thresholds. |
| [`custom_compactor.py`](./custom_compactor.py) | Implementing custom `Compactor` protocols via class implementations and manual invocation. |
| [`react_compaction.py`](./react_compaction.py) | Composing ReACT agent loops with per-add and per-turn context compaction strategies. |
