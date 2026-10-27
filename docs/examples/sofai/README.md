# SOFAI Examples

Examples demonstrating `SOFAISamplingStrategy` (Slow and Fast AI): a two-tier
sampling strategy where a fast S1 model iterates with validation feedback and
escalates to a slower S2 model only when S1 exhausts its budget. For the
strategy options (`s2_solver_mode`, `feedback_strategy`, `judge_backend`) and
usage guidance, see
[Inference-time scaling](../../docs/advanced/inference-time-scaling.md#sofai--dual-model-escalation).

## Examples

| File | Description |
| :--- | :--- |
| [`sofai_graph_coloring.py`](./sofai_graph_coloring.py) | Graph coloring on an odd 5-cycle with a `granite4.1:3b` S1 solver, a `granite4:latest` S2 solver, and a custom validator that returns detailed per-constraint feedback. Runs on consumer-grade hardware with Ollama. |
