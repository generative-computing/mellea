"""Tests for cli/decompose/decompose.py inference flow.

This module tests the run function's inference-related behavior, focusing on
local Ollama backend execution, prompt handling, argument forwarding, version
resolution, and failure cleanup.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from cli.decompose.decompose import DecompVersion, run
from cli.decompose.logging import LogMode
from cli.decompose.pipeline import DecompBackend, DecompPipelineResult

# ============================================================================
# Helpers
# ============================================================================


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


def make_decomp_result() -> DecompPipelineResult:
    """Create a minimal valid decomposition result for CLI tests."""
    return {
        "original_task_prompt": "Test task prompt",
        "subtask_list": ["Task A"],
        "identified_constraints": [],
        "subtasks": [
            {
                "subtask": "Task A",
                "tag": "TASK_A",
                "general_instructions": "",
                "constraints": [],
                "prompt_template": "Do A",
                "input_vars_required": [],
                "depends_on": [],
            }
        ],
    }


@pytest.fixture
def patch_jinja(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Jinja imports used inside run()."""
    monkeypatch.setattr("jinja2.Environment", DummyEnvironment)
    monkeypatch.setattr("jinja2.FileSystemLoader", lambda *args, **kwargs: None)


@pytest.fixture
def patch_validate_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch filename validation."""
    monkeypatch.setattr("cli.decompose.utils.validate_filename", lambda _: True)


@pytest.fixture
def patch_logging(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Patch logger setup."""
    logger = Mock()
    monkeypatch.setattr("cli.decompose.decompose.configure_logging", lambda _: None)
    monkeypatch.setattr("cli.decompose.decompose.get_logger", lambda _: logger)
    monkeypatch.setattr(
        "cli.decompose.decompose.log_section", lambda *args, **kwargs: None
    )
    return logger


# ============================================================================
# Tests for run inference success cases
# ============================================================================


class TestRunInferenceSuccess:
    """Tests for successful inference using local Ollama backend."""

    def test_default_ollama_backend_and_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Test default backend is ollama with 8b model."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Test prompt")

        captured: dict[str, Any] = {}

        def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
            captured.update(kwargs)
            return make_decomp_result()

        monkeypatch.setattr("cli.decompose.pipeline.decompose", fake_decompose)

        with prompt_path.open("r") as prompt_file:
            run(
                out_dir=tmp_path,
                out_name="default_case",
                prompt_file=prompt_file,
            )

        assert captured["backend"] == DecompBackend.ollama
        assert captured["model_id"] == "llama3:8b"

    def test_prompt_file_mode_forwards_inference_args(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Test prompt_file mode forwards inference arguments correctly."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Summarize document.")

        captured: dict[str, Any] = {}

        def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
            captured.update(kwargs)
            return make_decomp_result()

        monkeypatch.setattr("cli.decompose.pipeline.decompose", fake_decompose)

        with prompt_path.open("r") as prompt_file:
            run(
                out_dir=tmp_path,
                out_name="case_forward",
                prompt_file=prompt_file,
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

    def test_interactive_mode_reads_prompt_and_clears_input_vars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Test interactive mode reads typer.prompt and clears input vars."""
        captured: dict[str, Any] = {}

        def fake_decompose(**kwargs: Any) -> DecompPipelineResult:
            captured.update(kwargs)
            return make_decomp_result()

        monkeypatch.setattr("cli.decompose.pipeline.decompose", fake_decompose)
        monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "A\\nB")

        run(
            out_dir=tmp_path,
            out_name="interactive_case",
            prompt_file=None,
            input_var=["SHOULD_BE_IGNORED"],
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
        """Test latest version resolves to the last declared version."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Test")

        requested_templates: list[str] = []

        class TrackingEnvironment:
            """Tracking Jinja environment for template resolution assertions."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def get_template(self, template_name: str) -> DummyTemplate:
                requested_templates.append(template_name)
                return DummyTemplate()

        monkeypatch.setattr("jinja2.Environment", TrackingEnvironment)
        monkeypatch.setattr("jinja2.FileSystemLoader", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "cli.decompose.pipeline.decompose",
            lambda **kwargs: make_decomp_result(),
        )

        with prompt_path.open("r") as prompt_file:
            run(
                out_dir=tmp_path,
                out_name="version_case",
                prompt_file=prompt_file,
                version=DecompVersion.latest,
            )

        assert requested_templates == ["m_decomp_result_v2.py.jinja2"]

    def test_successful_inference_writes_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Test successful inference writes expected output files."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Generate subtasks.")

        monkeypatch.setattr(
            "cli.decompose.pipeline.decompose",
            lambda **kwargs: make_decomp_result(),
        )

        with prompt_path.open("r") as prompt_file:
            run(
                out_dir=tmp_path,
                out_name="ok_case",
                prompt_file=prompt_file,
            )

        out_dir = tmp_path / "ok_case"
        assert out_dir.exists()
        assert out_dir.is_dir()
        assert (out_dir / "ok_case.json").exists()
        assert (out_dir / "ok_case.py").exists()
        assert (out_dir / "validations").exists()
        assert (out_dir / "validations" / "__init__.py").exists()


# ============================================================================
# Tests for run inference failure cases
# ============================================================================


class TestRunInferenceFailures:
    """Tests for failure scenarios during inference."""

    def test_pipeline_exception_after_output_dir_creation_cleans_up(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Test partial output directory is removed when later steps fail."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("fail")

        monkeypatch.setattr(
            "cli.decompose.pipeline.decompose",
            lambda **kwargs: make_decomp_result(),
        )

        original_mkdir = Path.mkdir

        def fail_after_create(self: Path, *args: Any, **kwargs: Any) -> None:
            original_mkdir(self, *args, **kwargs)
            if self.name == "fail_case":
                raise RuntimeError("fail")

        monkeypatch.setattr(Path, "mkdir", fail_after_create)

        with prompt_path.open("r") as prompt_file:
            with pytest.raises(RuntimeError, match="fail"):
                run(
                    out_dir=tmp_path,
                    out_name="fail_case",
                    prompt_file=prompt_file,
                )

        assert not (tmp_path / "fail_case").exists()

    def test_pipeline_decompose_exception_is_reraised(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Test pipeline inference exception is re-raised."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("fail")

        def raise_inference_error(**kwargs: Any) -> DecompPipelineResult:
            raise RuntimeError("inference error")

        monkeypatch.setattr(
            "cli.decompose.pipeline.decompose",
            raise_inference_error,
        )

        with prompt_path.open("r") as prompt_file:
            with pytest.raises(RuntimeError, match="inference error"):
                run(
                    out_dir=tmp_path,
                    out_name="err_case",
                    prompt_file=prompt_file,
                )

        assert not (tmp_path / "err_case").exists()

    def test_invalid_output_dir_fails_before_inference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_jinja: None,
        patch_validate_filename: None,
        patch_logging: Mock,
    ) -> None:
        """Test invalid out_dir prevents inference from running."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Test prompt")

        decompose_mock = Mock(return_value=make_decomp_result())
        monkeypatch.setattr("cli.decompose.pipeline.decompose", decompose_mock)

        missing_dir = tmp_path / "does_not_exist"

        with prompt_path.open("r") as prompt_file:
            with pytest.raises(
                AssertionError,
                match='Path passed in the "out-dir" is not a directory',
            ):
                run(
                    out_dir=missing_dir,
                    out_name="m_decomp_result",
                    prompt_file=prompt_file,
                )

        decompose_mock.assert_not_called()