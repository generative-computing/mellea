# SOFAI (Slow and Fast AI) Examples

This directory contains examples of the SOFAI sampling strategy - a two-tier approach using fast and slow models.

## Files

### sofai_graph_coloring.py
Complete example using SOFAI for constraint satisfaction problems (graph coloring).

**Key Features:**
- Two-tier model architecture (fast S1, slow S2)
- Iterative feedback loop with fast model
- Escalation to slow model when needed
- Custom validation with detailed feedback
- Consumer-grade hardware friendly

## Concepts Demonstrated

- **Two-Tier Architecture**: Fast model for iteration, slow model for hard cases
- **Escalation Strategy**: When to switch from fast to slow model
- **Feedback Loops**: Providing validation feedback to guide generation
- **Constraint Satisfaction**: Solving CSP problems with LLMs
- **Cost Optimization**: Using expensive models only when necessary

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

## Basic Usage

```python
from mellea import start_session
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.sampling import SOFAISamplingStrategy
from mellea.stdlib.requirements import req

# Create fast and slow backends
s1_backend = OllamaModelBackend(model_id="phi:2.7b")
s2_backend = OllamaModelBackend(model_id="qwen3-4b-thinking")

# Create SOFAI strategy
strategy = SOFAISamplingStrategy(
    s1_backend=s1_backend,
    s2_backend=s2_backend,
    s1_loop_budget=3,  # Try fast model 3 times
    escalate_on_failure=True
)

# Use with requirements
m = start_session()
result = m.instruct(
    "Solve the problem...",
    requirements=[req("Must satisfy constraint X")],
    strategy=strategy
)
```

## Configuration Options

```python
SOFAISamplingStrategy(
    s1_backend=fast_backend,      # Fast model for iteration
    s2_backend=slow_backend,       # Slow model for escalation
    s1_loop_budget=3,              # Max attempts with fast model
    escalate_on_failure=True,      # Use slow model if fast fails
    feedback_strategy="all_errors" # How to provide feedback
)
```

## Feedback Strategies

- `"simple"`: Basic pass/fail feedback
- `"first_error"`: Feedback on first failing requirement
- `"all_errors"`: Detailed feedback on all failures

## Use Cases

- **Constraint Satisfaction**: Graph coloring, scheduling, planning
- **Complex Reasoning**: Multi-step problems requiring iteration
- **Cost Optimization**: Use expensive models only when needed
- **Quality Assurance**: Fast iteration with slow model fallback

## Model Selection

### Fast Models (S1)
- phi:2.7b
- llama2:7b
- mistral:7b
- granite-3.2-8b-instruct

### Slow Models (S2)
- qwen3-4b-thinking
- llama3:70b
- mixtral:8x7b
- granite-3.3-8b-instruct

## Performance Tips

1. **Choose appropriate S1 budget**: Balance speed vs. success rate
2. **Use specific requirements**: Clear constraints help both models
3. **Provide good feedback**: Detailed validation helps iteration
4. **Profile your use case**: Measure when escalation is needed

## Related Documentation

- See `mellea/stdlib/sampling/sofai.py` for implementation
- See `test/stdlib/sampling/test_sofai_*.py` for more examples
- See `mellea/stdlib/sampling/` for other sampling strategies