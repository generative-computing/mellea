# m_decompose

This module with the example demonstrates **task decomposition pipelines** built with *Mellea generative programs*.

Instead of solving a complex task with a single prompt, the system first **decomposes the task into subtasks**, then executes them sequentially through a assembled pipeline.

This pattern improves reasoning quality, interpretability, and modularity in LLM-powered systems.

---

# Overview

Many complex tasks contain multiple reasoning steps.  
The `m_decompose` pipeline handles this by splitting the task into smaller units.

```
User Request
     ↓
Task Decomposition
     ↓
Subtasks
     ↓
Task Execution
     ↓
Final Result
```

Rather than writing a large prompt, the workflow uses **generative modules and reusable prompts**.

---

# Directory

```
m_decompose/
├── decompose.py
├── pipeline.py
├── prompt_modules
└── README.md
```

**decompose.py**

Generates the refined subtasks from the user request.

**pipeline.py**

Runs the full workflow:

1. decompose the task  
2. execute subtasks  
3. aggregate results

**prompt_modules**

Reusable prompt components used by the pipeline.

**m_decomp_result_v1.py.jinja2**

Template used to format the final output.

---

# Quick Start

Example usage:

```python
from mellea.cli.decompose.pipeline import decompose, DecompBackend
import json

query = """I will visit Grand Canyon National Park for 3 days in early May. Please create a travel itinerary that includes major scenic viewpoints and short hiking trails. The daily walking distance should stay under 6 miles, and each day should include at least one sunset or sunrise viewpoint."""

result = decompose(
    task_prompt=query,
    model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    backend=DecompBackend.openai,
    backend_endpoint="http://localhost:8000/v1",
    backend_api_key="EMPTY",
)

print(json.dumps(result, indent=2, ensure_ascii=False))
```


The pipeline then executes each step and produces the final answer.

---

# What This Example Shows

This example highlights three key ideas:

- **Task Decomposition and Execution** — analyze complex problems into smaller planning and execution steps.  
- **Generative Mellea Program** — conduct LLM workflows as an programmatic pipeline instead of single call.  
- **Instruction Modular** — separate instruction design from execution logic using reusable modules.

---

# Summary

`m_decompose` shows how to build **LLM pipelines** using task decomposition.
