# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""File input expands literal backslash-n inside one task job."""

from pathlib import Path
from typing import Any

import pytest

from cli.decompose.decompose import run
from cli.decompose.pipeline import DecompPipelineResult


def test_input_file_expands_literal_backslash_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One physical line with a literal backslash-n is one multiline task."""
    input_path = tmp_path / "tasks.txt"
    input_path.write_text("Alpha\\nline\n", encoding="utf-8")
    prompts: list[str] = []

    def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
        prompts.append(kwargs["task_prompt"])
        return {
            "original_task_prompt": kwargs["task_prompt"],
            "subtask_list": [],
            "identified_constraints": [],
            "subtasks": [],
        }

    monkeypatch.setattr("cli.decompose.pipeline.decompose", fake_decompose)
    monkeypatch.setattr("cli.decompose.decompose.configure_logging", lambda _: None)

    run(out_dir=tmp_path, out_name="case", input_file=str(input_path))

    assert prompts == ["Alpha\nline"]
