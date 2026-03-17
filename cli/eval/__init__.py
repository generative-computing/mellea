"""CLI package for test-based LLM evaluation.

Provides the ``m eval`` command group, which orchestrates running a generator model
against structured test files and scoring each response with a judge model. Each test
file specifies a set of instructions and input examples; results — including per-input
pass/fail judgements and cumulative pass rates — are written to JSON or JSONL for
downstream analysis.
"""
