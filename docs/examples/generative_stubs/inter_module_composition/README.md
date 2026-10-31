# Inter-Module Composition

Composing modular generative functions across separate Python files to build multi-stage pipelines.

## Examples

| File | Description |
| :--- | :--- |
| [`summarize_and_decide.py`](./summarize_and_decide.py) | Main executable pipeline: composes summarizers and decision aides into an end-to-end workflow. |
| [`summarizers.py`](./summarizers.py) | Reusable module: text summarization generative stubs. |
| [`decision_aides.py`](./decision_aides.py) | Reusable module: decision-making and recommendation generative stubs. |