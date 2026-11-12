# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for Hub token lookup in aLoRA upload commands."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("huggingface_hub")
pytest.importorskip("typer")

_MISSING_TOKEN_MESSAGE = (
    "Hugging Face token not found. Set `HF_TOKEN` or run `hf auth login` first."
)


def test_upload_modules_import_without_hf_folder():
    """Upload modules must import against Hub 1.x, without the removed HfFolder."""
    for name in ("cli.alora.upload", "cli.alora.intrinsic_uploader"):
        importlib.import_module(name)


@pytest.mark.parametrize("token", [None, "test-token-not-a-real-credential"])
def test_upload_model_token_lookup(monkeypatch, tmp_path, token):
    """Plain uploads use the public token API and reject missing credentials."""
    from cli.alora import upload

    get_token = Mock(return_value=token)
    create_repo = Mock()
    upload_folder = Mock()
    monkeypatch.setattr(upload, "get_token", get_token)
    monkeypatch.setattr(upload, "create_repo", create_repo)
    monkeypatch.setattr(upload, "upload_folder", upload_folder)

    if token is None:
        with pytest.raises(OSError) as exc_info:
            upload.upload_model(str(tmp_path), "example/adapter")
        assert str(exc_info.value) == _MISSING_TOKEN_MESSAGE
        create_repo.assert_not_called()
        upload_folder.assert_not_called()
    else:
        upload.upload_model(str(tmp_path), "example/adapter")
        create_repo.assert_called_once_with(
            repo_id="example/adapter", token=token, private=True, exist_ok=True
        )
        assert upload_folder.call_args.kwargs["token"] == token
    get_token.assert_called_once_with()


@pytest.mark.parametrize("token", [None, "test-token-not-a-real-credential"])
def test_upload_intrinsic_token_lookup(monkeypatch, tmp_path, token):
    """Intrinsic uploads preserve auth errors and clean up staged files."""
    from cli.alora import intrinsic_uploader

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "adapter_config.json").write_text("{}")
    io_yaml = tmp_path / "io.yaml"
    io_yaml.write_text("instruction: synthetic test\n")
    get_token = Mock(return_value=token)
    create_repo = Mock(return_value=SimpleNamespace(url="https://example.invalid"))
    staged_paths = []

    def capture_upload(**kwargs):
        staged = Path(kwargs["folder_path"])
        assert (staged / "io.yaml").read_text() == io_yaml.read_text()
        assert kwargs["token"] == token
        assert kwargs["path_in_repo"] == "adapter/base/alora"
        staged_paths.append(staged)

    upload_folder = Mock(side_effect=capture_upload)
    monkeypatch.setattr(intrinsic_uploader, "get_token", get_token)
    monkeypatch.setattr(intrinsic_uploader, "create_repo", create_repo)
    monkeypatch.setattr(intrinsic_uploader, "upload_folder", upload_folder)

    args = (str(weights), "example/adapter", "example/base", "alora", str(io_yaml))
    if token is None:
        with pytest.raises(OSError) as exc_info:
            intrinsic_uploader.upload_intrinsic(*args)
        assert str(exc_info.value) == _MISSING_TOKEN_MESSAGE
        create_repo.assert_not_called()
        upload_folder.assert_not_called()
    else:
        intrinsic_uploader.upload_intrinsic(*args)
        create_repo.assert_called_once_with(
            repo_id="example/adapter", token=token, private=True, exist_ok=True
        )
        upload_folder.assert_called_once()
        assert not staged_paths[0].exists()
    get_token.assert_called_once_with()


def test_upload_intrinsic_leaves_source_directory_untouched(monkeypatch, tmp_path):
    """Staging must not modify the adapter directory (issue #1812)."""
    from cli.alora import intrinsic_uploader

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "adapter_config.json").write_text("{}")
    (weights / "README.md").write_text("# my model card\n")
    io_yaml = tmp_path / "io.yaml"
    io_yaml.write_text("instruction: synthetic test\n")
    before = sorted(p.name for p in weights.iterdir())

    staged_names = []

    def capture_upload(**kwargs):
        staged = Path(kwargs["folder_path"])
        staged_names.append(sorted(p.name for p in staged.iterdir()))

    monkeypatch.setattr(intrinsic_uploader, "get_token", Mock(return_value="t"))
    monkeypatch.setattr(
        intrinsic_uploader,
        "create_repo",
        Mock(return_value=SimpleNamespace(url="https://example.invalid")),
    )
    monkeypatch.setattr(
        intrinsic_uploader, "upload_folder", Mock(side_effect=capture_upload)
    )

    intrinsic_uploader.upload_intrinsic(
        str(weights), "example/adapter", "example/base", "alora", str(io_yaml)
    )

    # The staged copy adds io.yaml and leaves out README.md...
    assert staged_names == [["adapter_config.json", "io.yaml"]]
    # ...while the source directory keeps every file it had.
    assert sorted(p.name for p in weights.iterdir()) == before
    assert (weights / "README.md").read_text() == "# my model card\n"


@pytest.mark.parametrize("token", [None, "test-token-not-a-real-credential"])
def test_add_readme_token_lookup(monkeypatch, token):
    """README orchestration uses Hub token lookup, with generation isolated."""
    import huggingface_hub

    from cli.alora.commands import alora_add_readme

    # Isolate LLM generation: this test exercises upload orchestration, not a model.
    generator = ModuleType("cli.alora.readme_generator")

    def generate_readme(**kwargs):
        Path(kwargs["output_path"]).write_text("# Synthetic adapter\n")

    monkeypatch.setattr(generator, "generate_readme", generate_readme, raising=False)
    monkeypatch.setitem(sys.modules, "cli.alora.readme_generator", generator)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    get_token = Mock(return_value=token)
    create_repo = Mock()
    upload_file = Mock()
    monkeypatch.setattr(huggingface_hub, "get_token", get_token)
    monkeypatch.setattr(huggingface_hub, "create_repo", create_repo)
    monkeypatch.setattr(huggingface_hub, "upload_file", upload_file)

    args = dict(
        datafile="synthetic.jsonl",
        basemodel="example/base",
        promptfile=None,
        name="example/adapter",
        hints=None,
        io_yaml=None,
        # Explicit layout: this test exercises token lookup, not layout detection.
        layout="intrinsic",
    )
    if token is None:
        with pytest.raises(OSError) as exc_info:
            alora_add_readme(**args)
        assert str(exc_info.value) == _MISSING_TOKEN_MESSAGE
        create_repo.assert_not_called()
        upload_file.assert_not_called()
    else:
        alora_add_readme(**args)
        create_repo.assert_called_once_with(
            repo_id="example/adapter", token=token, private=True, exist_ok=True
        )
        assert upload_file.call_args.kwargs["token"] == token
    get_token.assert_called_once_with()
