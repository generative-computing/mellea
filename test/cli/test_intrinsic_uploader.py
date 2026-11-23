# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `cli.alora.intrinsic_uploader` staging and immutability."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("huggingface_hub")


def _load_uploader_with_mock_hub(monkeypatch):
    from cli.alora import intrinsic_uploader as uploader

    monkeypatch.setattr(uploader, "get_token", lambda: "test-token")
    monkeypatch.setattr(
        uploader,
        "create_repo",
        Mock(return_value=SimpleNamespace(url="https://example.invalid")),
    )
    return uploader


def _snapshot(root: Path) -> dict[str, bytes]:
    """Return a SHA-256 digest for every file under `root`."""
    entries: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).digest()
    return entries


def _make_capture_upload_folder(
    staged_files: dict[str, str], upload_kwargs: dict | None = None
):
    def _capture_upload_folder(**kwargs):
        if upload_kwargs is not None:
            upload_kwargs.update(kwargs)
        folder = kwargs["folder_path"]
        for root, _, names in os.walk(folder):
            for name in names:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, folder).replace(os.sep, "/")
                with open(full, encoding="utf-8") as fh:
                    staged_files[rel] = fh.read()

    return _capture_upload_folder


def test_upload_intrinsic_preserves_local_adapter_directory(tmp_path, monkeypatch):
    uploader = _load_uploader_with_mock_hub(monkeypatch)

    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "ibm-granite/granite-3.3-2b-instruct"}',
        encoding="utf-8",
    )
    (weight_dir / "adapter_model.safetensors").write_text(
        "weights-bytes", encoding="utf-8"
    )
    (weight_dir / "README.md").write_text("# my model card\n", encoding="utf-8")
    before = _snapshot(weight_dir)

    io_yaml = tmp_path / "io.yaml"
    io_yaml.write_text("x: 1\n", encoding="utf-8")

    staged_files: dict[str, str] = {}
    upload_kwargs: dict[str, str] = {}
    monkeypatch.setattr(
        uploader,
        "upload_folder",
        _make_capture_upload_folder(staged_files, upload_kwargs),
    )
    upload_file = Mock()
    monkeypatch.setattr(uploader, "upload_file", upload_file)

    uploader.upload_intrinsic(
        weight_path=str(weight_dir),
        model_name="org/carbchecker",
        base_model="ibm-granite/granite-3.3-2b-instruct",
        type="alora",
        io_yaml=str(io_yaml),
    )

    assert sorted(os.listdir(weight_dir)) == [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert _snapshot(weight_dir) == before
    assert (weight_dir / "README.md").read_text(encoding="utf-8") == "# my model card\n"
    assert "README.md" not in staged_files
    assert staged_files == {
        "adapter_config.json": '{"base_model_name_or_path": "ibm-granite/granite-3.3-2b-instruct"}',
        "adapter_model.safetensors": "weights-bytes",
        "io.yaml": "x: 1\n",
    }
    assert upload_kwargs["path_in_repo"] == os.path.join(
        "carbchecker", "granite-3.3-2b-instruct", "alora"
    )
    upload_file.assert_not_called()


def test_upload_intrinsic_overrides_io_yaml_in_staging_only(tmp_path, monkeypatch):
    uploader = _load_uploader_with_mock_hub(monkeypatch)

    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (weight_dir / "README.md").write_text("# local card\n", encoding="utf-8")
    (weight_dir / "INTRINSIC_README.md").write_text("# root readme\n", encoding="utf-8")
    (weight_dir / "io.yaml").write_text("old: true\n", encoding="utf-8")
    before = _snapshot(weight_dir)

    io_yaml = tmp_path / "io.yaml"
    io_yaml.write_text("fresh: true\n", encoding="utf-8")

    staged_files: dict[str, str] = {}
    monkeypatch.setattr(
        uploader, "upload_folder", _make_capture_upload_folder(staged_files)
    )
    upload_file = Mock()
    monkeypatch.setattr(uploader, "upload_file", upload_file)

    uploader.upload_intrinsic(
        weight_path=str(weight_dir),
        model_name="org/carbchecker",
        base_model="granite-3.3-2b-instruct",
        type="lora",
        io_yaml=str(io_yaml),
    )

    assert _snapshot(weight_dir) == before
    assert (weight_dir / "io.yaml").read_text(encoding="utf-8") == "old: true\n"
    assert (weight_dir / "README.md").read_text(encoding="utf-8") == "# local card\n"
    assert "README.md" not in staged_files
    assert staged_files["io.yaml"] == "fresh: true\n"
    upload_file.assert_called_once()
    assert upload_file.call_args.kwargs["path_in_repo"] == "README.md"
    assert upload_file.call_args.kwargs["path_or_fileobj"] == str(
        weight_dir / "INTRINSIC_README.md"
    )


def test_upload_intrinsic_without_readme(tmp_path, monkeypatch):
    uploader = _load_uploader_with_mock_hub(monkeypatch)

    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "ibm-granite/granite-3.3-2b-instruct"}',
        encoding="utf-8",
    )
    (weight_dir / "adapter_model.safetensors").write_text(
        "weights-bytes", encoding="utf-8"
    )
    before = _snapshot(weight_dir)

    io_yaml = tmp_path / "io.yaml"
    io_yaml.write_text("x: 1\n", encoding="utf-8")

    staged_files: dict[str, str] = {}
    upload_kwargs: dict[str, str] = {}
    monkeypatch.setattr(
        uploader,
        "upload_folder",
        _make_capture_upload_folder(staged_files, upload_kwargs),
    )
    upload_file = Mock()
    monkeypatch.setattr(uploader, "upload_file", upload_file)

    uploader.upload_intrinsic(
        weight_path=str(weight_dir),
        model_name="org/carbchecker",
        base_model="ibm-granite/granite-3.3-2b-instruct",
        type="alora",
        io_yaml=str(io_yaml),
    )

    assert sorted(os.listdir(weight_dir)) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert _snapshot(weight_dir) == before
    assert "README.md" not in staged_files
    assert staged_files == {
        "adapter_config.json": '{"base_model_name_or_path": "ibm-granite/granite-3.3-2b-instruct"}',
        "adapter_model.safetensors": "weights-bytes",
        "io.yaml": "x: 1\n",
    }
    assert upload_kwargs["path_in_repo"] == os.path.join(
        "carbchecker", "granite-3.3-2b-instruct", "alora"
    )
    upload_file.assert_not_called()
