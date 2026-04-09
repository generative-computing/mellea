"""Tests for ``cli/decompose/decompose.py`` run flow.

This module validates prompt loading, argument forwarding, template selection,
output writing, multi-job behavior, and cleanup behavior for the ``run`` command.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from cli.decompose.decompose import DecompVersion, reorder_subtasks, run
from cli.decompose.logging import LogMode
from cli.decompose.pipeline import DecompBackend, DecompPipelineResult


class DummyTemplate:
    """Minimal Jinja template stub used by tests."""

    def render(self, **kwargs: Any) -> str:
        return "# generated test program"


class DummyEnvironment:
    """Minimal Jinja environment stub used by tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_template(self, template_name: str) -> DummyTemplate:
        return DummyTemplate()


def make_decomp_result(with_code_validation: bool = False) -> DecompPipelineResult:
    """Create a valid decomposition result for CLI tests."""
    identified_constraints = []
    subtask_constraints = []

    if with_code_validation:
        identified_constraints = [
            {
                "constraint": "Return valid JSON.",
                "val_strategy": "code",
                "val_fn": "def validate_input(text: str) -> bool:\n    return text.startswith('{')",
                "val_fn_name": "val_fn_1",
            }
        ]
        subtask_constraints = [
            {
                "constraint": "Return valid JSON.",
                "val_strategy": "code",
                "val_fn": "def validate_input(text: str) -> bool:\n    return text.startswith('{')",
                "val_fn_name": "val_fn_1",
            }
        ]

    return {
        "original_task_prompt": "Test task prompt",
        "subtask_list": ["Task A"],
        "identified_constraints": identified_constraints,
        "subtasks": [
            {
                "subtask": "1. Task A",
                "tag": "TASK_A",
                "general_instructions": "Keep the answer concise.",
                "constraints": subtask_constraints,
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            }
        ],
    }


def write_input_file(tmp_path: Path, content: str, name: str = "input.txt") -> str:
    """Write task prompt content and return file path as string."""
    input_path = tmp_path / name
    input_path.write_text(content, encoding="utf-8")
    return str(input_path)


@pytest.fixture
def patch_jinja(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Jinja objects used by ``run``."""
    monkeypatch.setattr("cli.decompose.decompose.Environment", DummyEnvironment)
    monkeypatch.setattr(
        "cli.decompose.decompose.FileSystemLoader", lambda *args, **kwargs: None
    )


@pytest.fixture
def patch_validate_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch filename validation."""
    monkeypatch.setattr("cli.decompose.decompose.validate_filename", lambda _: True)


@pytest.fixture
def patch_logging(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Patch logger setup and logger access."""
    logger = Mock()
    monkeypatch.setattr("cli.decompose.decompose.configure_logging", lambda _: None)
    monkeypatch.setattr("cli.decompose.decompose.get_logger", lambda _: logger)
    monkeypatch.setattr(
        "cli.decompose.decompose.log_section", lambda *args, **kwargs: None
    )
    return logger


class TestRunSuccess:
    """Tests for successful run scenarios."""

    def test_default_backend_and_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Default backend/model are forwarded to pipeline."""
        input_file = write_input_file(tmp_path, "Test prompt")
        captured: dict[str, Any] = {}

        def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
            captured.update(kwargs)
            return make_decomp_result()

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose", fake_decompose
        )

        run(out_dir=tmp_path, out_name="default_case", input_file=input_file)

        assert captured["backend"] == DecompBackend.ollama
        assert captured["model_id"] == "mistral-small3.2:latest"

    def test_input_file_mode_forwards_inference_args(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Input file mode forwards args to ``pipeline.decompose``."""
        input_file = write_input_file(tmp_path, "Summarize document.")
        captured: dict[str, Any] = {}

        def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
            captured.update(kwargs)
            return make_decomp_result()

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose", fake_decompose
        )

        run(
            out_dir=tmp_path,
            out_name="case_forward",
            input_file=input_file,
            model_id="llama3:8b",
            backend=DecompBackend.ollama,
            backend_req_timeout=111,
            input_var=["DOC"],
            log_mode=LogMode.debug,
        )

        assert captured["task_prompt"] == "Summarize document."
        assert captured["backend"] == DecompBackend.ollama
        assert captured["model_id"] == "llama3:8b"
        assert captured["backend_req_timeout"] == 111
        assert captured["user_input_variable"] == ["DOC"]
        assert captured["log_mode"] == LogMode.debug

    def test_interactive_mode_reads_prompt_and_ignores_input_vars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Interactive mode reads prompt and sends ``None`` input vars."""
        captured: dict[str, Any] = {}

        def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
            captured.update(kwargs)
            return make_decomp_result()

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose", fake_decompose
        )
        monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "A\\nB")

        run(
            out_dir=tmp_path,
            out_name="interactive_case",
            input_file=None,
            input_var=["IGNORED_IN_INTERACTIVE_MODE"],
        )

        assert captured["task_prompt"] == "A\nB"
        assert captured["user_input_variable"] is None

    def test_latest_version_resolves_to_last_declared_version(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """``latest`` resolves to the last declared enum version."""
        input_file = write_input_file(tmp_path, "Test")
        requested_templates: list[str] = []
        environment = Mock()
        environment.get_template.side_effect = lambda template_name: (
            requested_templates.append(template_name) or DummyTemplate()
        )

        monkeypatch.setattr("cli.decompose.decompose.Environment", lambda *args, **kwargs: environment)
        monkeypatch.setattr(
            "cli.decompose.decompose.FileSystemLoader", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose",
            lambda **kwargs: make_decomp_result(),
        )

        run(
            out_dir=tmp_path,
            out_name="version_case",
            input_file=input_file,
            version=DecompVersion.latest,
        )

        assert requested_templates == ["m_decomp_result_v3.py.jinja2"]

    def test_successful_run_writes_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Successful run writes expected directory structure and files."""
        input_file = write_input_file(tmp_path, "Generate subtasks.")

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose",
            lambda **kwargs: make_decomp_result(),
        )

        run(out_dir=tmp_path, out_name="ok_case", input_file=input_file)

        out_path = tmp_path / "ok_case"
        assert out_path.exists()
        assert out_path.is_dir()
        assert (out_path / "ok_case.json").exists()
        assert (out_path / "ok_case.py").exists()
        assert (out_path / "validations").exists()
        assert (out_path / "validations" / "__init__.py").exists()

    def test_latest_template_writes_validation_modules_for_code_constraints(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Latest template includes code-validation imports and writes modules."""
        input_file = write_input_file(tmp_path, "Generate JSON.")

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose",
            lambda **kwargs: make_decomp_result(with_code_validation=True),
        )

        run(out_dir=tmp_path, out_name="validated_case", input_file=input_file)

        validation_path = tmp_path / "validated_case" / "validations" / "val_fn_1.py"
        program_path = tmp_path / "validated_case" / "validated_case.py"

        assert validation_path.exists()
        assert "def validate_input" in validation_path.read_text(encoding="utf-8")

        program = program_path.read_text(encoding="utf-8")
        assert "from mellea.stdlib.requirements import req" in program
        assert (
            "from validations.val_fn_1 import validate_input as val_fn_1" in program
        )
        assert "Keep the answer concise." in program

    def test_multi_line_input_file_creates_numbered_jobs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Each non-empty input line becomes one numbered output job."""
        input_file = write_input_file(tmp_path, "Task 1\n\nTask 2\n")
        calls: list[str] = []

        def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
            calls.append(kwargs["task_prompt"])
            return make_decomp_result()

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose", fake_decompose
        )

        run(out_dir=tmp_path, out_name="batch", input_file=input_file)

        assert calls == ["Task 1", "Task 2"]
        assert (tmp_path / "batch_1" / "batch_1.json").exists()
        assert (tmp_path / "batch_2" / "batch_2.json").exists()


class TestRunFailures:
    """Tests for failure scenarios during run."""

    def test_pipeline_exception_after_output_dir_creation_cleans_up(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """A failure after output-dir registration removes partial output."""
        input_file = write_input_file(tmp_path, "fail")

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose",
            lambda **kwargs: make_decomp_result(),
        )

        original_mkdir = Path.mkdir

        def fail_after_create(self: Path, *args: Any, **kwargs: Any) -> None:
            original_mkdir(self, *args, **kwargs)
            if self.name == "validations" and self.parent.name == "fail_case":
                raise RuntimeError("fail")

        monkeypatch.setattr(Path, "mkdir", fail_after_create)

        with pytest.raises(RuntimeError, match="fail"):
            run(out_dir=tmp_path, out_name="fail_case", input_file=input_file)

        assert not (tmp_path / "fail_case").exists()

    def test_pipeline_decompose_exception_is_reraised(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Exceptions from ``pipeline.decompose`` are propagated."""
        input_file = write_input_file(tmp_path, "fail")

        def raise_inference_error(**kwargs: Any) -> DecompPipelineResult:
            raise RuntimeError("inference error")

        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose", raise_inference_error
        )

        with pytest.raises(RuntimeError, match="inference error"):
            run(out_dir=tmp_path, out_name="err_case", input_file=input_file)

        assert not (tmp_path / "err_case").exists()

    def test_invalid_output_dir_fails_before_inference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Invalid ``out_dir`` fails before pipeline call."""
        input_file = write_input_file(tmp_path, "Test prompt")

        decompose_mock = Mock(return_value=make_decomp_result())
        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose", decompose_mock
        )

        missing_dir = tmp_path / "does_not_exist"

        with pytest.raises(
            AssertionError, match='Path passed in the "out-dir" is not a directory'
        ):
            run(out_dir=missing_dir, out_name="m_decomp_result", input_file=input_file)

        decompose_mock.assert_not_called()

    def test_empty_input_file_raises_value_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Input files with only blank lines are rejected."""
        input_file = write_input_file(tmp_path, "\n \n\t\n")

        decompose_mock = Mock(return_value=make_decomp_result())
        monkeypatch.setattr(
            "cli.decompose.decompose.pipeline.decompose", decompose_mock
        )

        with pytest.raises(
            ValueError, match="Input file contains no non-empty task lines"
        ):
            run(out_dir=tmp_path, out_name="empty_case", input_file=input_file)

        decompose_mock.assert_not_called()
