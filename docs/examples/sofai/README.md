# SOFAI (Slow and Fast AI) Examples

This directory contains examples of the SOFAI sampling strategy - a two-tier approach
using fast and slow models. For usage guidance and `s2_solver_mode`, see
[Inference-time scaling](../../docs/advanced/inference-time-scaling.md#sofai--dual-model-escalation).
The `SOFAISamplingStrategy` docstring in `mellea/stdlib/sampling/sofai.py` documents
the remaining options, including `judge_backend` and `feedback_strategy`.

## Examples

| File | Description |
| :--- | :--- |
| [`sofai_graph_coloring.py`](./sofai_graph_coloring.py) | Complete example using SOFAI for constraint satisfaction problems (graph coloring), with a `granite4.1:3b` S1 solver, a `granite4:latest` S2 solver, and a custom validator that returns detailed feedback. Consumer-grade hardware friendly. |

## SOFAI Architecture

```
User Request
    ↓
S1 (Fast Model) ←─────┐
    ↓                  │
Validation             │
    ↓                  │
Pass? ──No→ Feedback ──┘
    │
   Yes
    ↓
Success? ──No→ Escalate to S2 (Slow Model)
    │              ↓
   Yes         Validation
    ↓              ↓
  Result        Result
```
