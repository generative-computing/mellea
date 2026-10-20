# Qiskit Code Validation

Generating quantum computing code that automatically passes `flake8-qiskit-migration` validation rules using instruct-validate-repair.

## Overview

Takes a prompt containing deprecated Qiskit code and:

1. Generates corrected code using the LLM.
2. Validates the output against QKT migration rules.
3. Automatically repairs the code if validation fails (up to 10 attempts).

## Files

| File | Description |
| :--- | :--- |
| [`qiskit_code_validation.py`](./qiskit_code_validation.py) | Main executable example: generates and repairs Qiskit code against migration linter rules. |
| [`validation_helpers.py`](./validation_helpers.py) | Supporting helper module: linter execution and AST validation utilities. |

## Quick Start

```bash
uv run docs/examples/instruct_validate_repair/qiskit_code_validation/qiskit_code_validation.py
```

Dependencies (`mellea`, `flake8-qiskit-migration`) are resolved automatically via PEP 723 inline script metadata.

## Sampling Strategies & Benchmarks

The example supports two repair strategies:

- `RepairTemplateStrategy`: Adds validation failure reasons directly to the instruction and retries generation.
- `MultiTurnStrategy`: Builds conversation history by adding validation failures as new user messages.

Benchmarks on `mistral-small-3.2-24b-qiskit` model:

| Dataset | Strategy | First Pass (QKT) | Post-Repair (QKT) |
| :--- | :--- | :--- | :--- |
| **QHE** (QiskitHumanEval) | RepairTemplate | 97.4% | **100%** |
| | MultiTurn | 95.4% | **100%** |
| **QKT** (Migration Tasks) | RepairTemplate | 88.9% | **100%** |
| | MultiTurn | **97.8%** | **100%** |

*Note: Pass rates measure whether generated code passes QKT validation rules.*

## Model Selection

- Recommended: `hf.co/Qiskit/mistral-small-3.2-24b-qiskit-GGUF:latest` (domain-specialized fine-tune).
- General-purpose alternative: `granite4:micro-h` (set `system_prompt = QISKIT_SYSTEM_PROMPT` in `qiskit_code_validation.py` to guide modern Qiskit 1.0+ APIs).
