# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the nightly test harness."""

from pathlib import Path

SCRIPT = Path(__file__).parent / "scripts" / "run_tests_with_ollama_and_vllm.sh"


def test_nightly_script_restarts_ollama_before_notebooks():
    """The notebook pass must have Ollama after phased cleanup."""
    source = SCRIPT.read_text(encoding="utf-8")
    notebook_section = source.split("# --- Run notebooks (nbmake) ---", maxsplit=1)[1]

    assert 'WITH_NOTEBOOKS="${WITH_NOTEBOOKS:-${WITH_EXAMPLES:-0}}"' in source
    assert 'if [[ "$WITH_NOTEBOOKS" == "1" ]]; then' in notebook_section
    assert 'log "Starting Ollama for notebook tests..."' in notebook_section
    assert "    start_ollama" in notebook_section
