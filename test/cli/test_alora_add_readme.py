# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `m alora add-readme` layout detection wiring.

The README generator's LLM and the Hugging Face upload are replaced so these
tests exercise only the layout resolution in `cli/alora/commands.py`.
"""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")

from cli.m import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_hub_downloads(monkeypatch):
    """Fail any unstubbed config download instead of reaching the real Hub.

    Tests that care about the download replace this with their own stub.
    """

    def refuse(**_kwargs):
        raise OSError("Hub downloads are disabled in this test")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", refuse)


class _FakeHfApi:
    """Stand-in for `HfApi` exposing a canned repository file listing."""

    def __init__(self, repo_files: list[str]) -> None:
        self._repo_files = repo_files

    def model_info(self, **_kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=path) for path in self._repo_files]
        )


def _plain(output: str) -> str:
    """Flatten Rich-rendered CLI output: drop ANSI codes and box lines, join wraps."""
    no_ansi = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return " ".join(no_ansi.replace("│", " ").split())


def _invoke_add_readme(
    monkeypatch,
    extra_args: list[str],
    *,
    interactive: bool | None = True,
    token: str | None = "test-token",
    basemodel: str = "ibm-granite/granite-4.1-3b",
    answers: list[str] | None = None,
):
    """Run `add-readme` with generation and uploading stubbed out.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        extra_args: Additional CLI arguments appended after `--name`.
        interactive: Whether standard input counts as a terminal, or `None` to
            leave the real check in place (under `CliRunner` it is not a terminal).
        token: Hugging Face token returned by `get_token`, or `None` for none.
        basemodel: Value passed as `--basemodel`.
        answers: Replies to successive prompts; defaults to "yes" every time.

    Returns:
        Tuple of the CLI result and the kwargs passed to `generate_readme`.
    """
    captured: dict = {}

    def fake_generate_readme(**kwargs):
        captured.update(kwargs)
        with open(kwargs["output_path"], "w") as fh:
            fh.write("# README\n")
        return kwargs["output_path"]

    if interactive is not None:
        monkeypatch.setattr(
            "cli.alora.commands._stdin_is_interactive", lambda: interactive
        )
    monkeypatch.setattr("huggingface_hub.get_token", lambda: token)
    monkeypatch.setattr("huggingface_hub.create_repo", lambda **_kwargs: None)
    monkeypatch.setattr(
        "huggingface_hub.upload_file",
        lambda **kwargs: captured.setdefault("upload_file", kwargs),
    )
    monkeypatch.setattr(
        "cli.alora.readme_generator.generate_readme", fake_generate_readme
    )
    replies = iter(answers) if answers is not None else None
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: next(replies) if replies is not None else "yes",
    )

    result = runner.invoke(
        cli,
        [
            "alora",
            "add-readme",
            "data.jsonl",
            "--basemodel",
            basemodel,
            "--name",
            "acme/stembolts",
            *extra_args,
        ],
    )
    return result, captured


@pytest.mark.parametrize("layout", ["intrinsic", "flat"])
def test_explicit_layout_skips_detection(monkeypatch, layout):
    """`--layout` is forwarded to the generator without inspecting the repo."""
    result, captured = _invoke_add_readme(monkeypatch, ["--layout", layout])

    assert result.exit_code == 0, result.output
    assert captured["layout"] == layout


@pytest.mark.parametrize(
    ("repo_files", "expected_layout"),
    [
        (["stembolts/granite-4.1-3b/alora/adapter_config.json"], "intrinsic"),
        (["adapter_config.json", "adapter_model.safetensors"], "flat"),
    ],
)
def test_auto_detects_repo_layout(monkeypatch, repo_files, expected_layout):
    """`--layout auto` uses the repository's file listing to pick the sample."""
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    assert captured["layout"] == expected_layout


@pytest.mark.parametrize(
    ("repo_files", "expected_type"),
    [
        (["stembolts/granite-4.1-3b/alora/adapter_config.json"], "alora"),
        (["stembolts/granite-4.1-3b/lora/adapter_config.json"], "lora"),
    ],
)
def test_detected_adapter_type_reaches_the_generator(
    monkeypatch, repo_files, expected_type
):
    """The adapter type found for `--basemodel` is documented, not assumed alora."""
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    assert captured["layout"] == "intrinsic"
    assert captured["adapter_type"] == expected_type


def test_both_adapter_types_present_documents_alora(monkeypatch):
    """A repo holding both types is documented as alora, and says so."""
    repo_files = [
        "stembolts/granite-4.1-3b/lora/adapter_config.json",
        "stembolts/granite-4.1-3b/alora/adapter_config.json",
    ]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    assert captured["adapter_type"] == "alora"
    assert "both lora and alora weights" in _plain(result.output)


def test_nested_weights_for_another_base_model_warn_but_still_generate(monkeypatch):
    """Nested weights the loader would not find for `--basemodel` get a warning.

    The README is still generated: the user may have meant a different base
    model, and generation is slow enough that refusing would be hostile.
    """
    repo_files = ["stembolts/granite-3.3-2b-instruct/alora/adapter_config.json"]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    message = _plain(result.output)
    assert "no weights for base model 'ibm-granite/granite-4.1-3b'" in message
    assert "stembolts/granite-4.1-3b/lora/" in message
    assert "stembolts/granite-4.1-3b/alora/" in message
    assert captured["layout"] == "intrinsic"
    # No weights were found, so there is no type to report; the generator defaults.
    assert captured["adapter_type"] is None


def test_unmapped_org_qualified_base_model_is_found_without_a_false_warning(
    monkeypatch,
):
    """Weights uploaded for `myorg/my-finetune` live under `my-finetune/`.

    The backend and `m alora upload --intrinsic` both drop the org, so checking
    the full ID would warn that a correctly uploaded adapter has no weights.
    """
    repo_files = ["stembolts/my-finetune/lora/adapter_config.json"]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(
        monkeypatch, [], basemodel="myorg/my-finetune"
    )

    assert result.exit_code == 0, result.output
    assert "no weights for base model" not in _plain(result.output)
    assert captured["layout"] == "intrinsic"
    assert captured["adapter_type"] == "lora"


@pytest.mark.parametrize(
    ("layout", "expected_message"),
    [("intrinsic", "Upload intrinsic README."), ("flat", "Upload README.")],
)
def test_upload_commit_message_matches_the_layout(
    monkeypatch, layout, expected_message
):
    """A flat repository's history must not call its README an intrinsic README."""
    result, captured = _invoke_add_readme(monkeypatch, ["--layout", layout])

    assert result.exit_code == 0, result.output
    assert captured["upload_file"]["commit_message"] == expected_message


@pytest.mark.parametrize(
    "extra_args", [["--layout", "intrinsic"], ["--layout", "flat"]]
)
def test_explicit_layout_reports_no_adapter_type(monkeypatch, extra_args):
    """Without inspecting the repo there is nothing to read the type from."""
    result, captured = _invoke_add_readme(monkeypatch, extra_args)

    assert result.exit_code == 0, result.output
    assert captured["adapter_type"] is None


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        # PEFT writes the key for every LoRA config and nulls it unless activated.
        ({"alora_invocation_tokens": None, "r": 6}, "lora"),
        ({"alora_invocation_tokens": [1, 2, 3], "r": 32}, "alora"),
    ],
)
def test_flat_repo_adapter_type_is_read_from_its_config(
    monkeypatch, tmp_path, config, expected_type
):
    """A flat repo's type comes from the root `adapter_config.json`.

    The file is downloaded into a throwaway directory, not the shared Hugging Face
    cache, and that directory is gone once the type has been read.
    """
    downloaded: dict = {}

    def fake_download(**kwargs):
        downloaded.update(kwargs)
        config_path = Path(kwargs["cache_dir"]) / "adapter_config.json"
        config_path.write_text(json.dumps(config))
        return str(config_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: _FakeHfApi(["adapter_config.json"])
    )

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    assert captured["layout"] == "flat"
    assert captured["adapter_type"] == expected_type
    # Only the one small config file is fetched, from the target repo.
    assert downloaded["repo_id"] == "acme/stembolts"
    assert downloaded["filename"] == "adapter_config.json"
    # ...into a temporary directory outside the shared cache, removed afterwards.
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(downloaded["cache_dir"])
    assert not str(cache_dir).startswith(str(HF_HUB_CACHE))
    assert not cache_dir.exists()


def test_hub_requests_are_announced_before_they_are_made(monkeypatch, tmp_path):
    """The user is told about each read-only Hub request, not just its result."""

    def fake_download(**kwargs):
        path = Path(kwargs["cache_dir"]) / "adapter_config.json"
        path.write_text(json.dumps({"alora_invocation_tokens": None}))
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: _FakeHfApi(["adapter_config.json"])
    )

    result, _ = _invoke_add_readme(monkeypatch, [])

    message = _plain(result.output)
    assert "Checking acme/stembolts on Hugging Face (read-only)" in message
    assert "Reading adapter_config.json from acme/stembolts" in message
    # The announcement comes before the outcome it precedes.
    assert message.index("Checking acme/stembolts") < message.index(
        "Detected the flat adapter layout"
    )


@pytest.mark.parametrize("layout", ["intrinsic", "flat"])
def test_explicit_layout_makes_and_announces_no_hub_requests(monkeypatch, layout):
    """`--layout` skips detection entirely, so there is nothing to announce."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an explicit --layout must not contact Hugging Face")

    monkeypatch.setattr("huggingface_hub.HfApi", forbidden)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)

    result, _ = _invoke_add_readme(monkeypatch, ["--layout", layout])

    assert result.exit_code == 0, result.output
    assert "on Hugging Face (read-only)" not in _plain(result.output)


def test_not_found_does_not_claim_the_repository_is_missing(monkeypatch):
    """Hugging Face also raises not-found for a private repo the token cannot see.

    The message must not state the repository does not exist, which could be false.
    """
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    class NotFoundApi:
        def model_info(self, **_kwargs):
            raise RepositoryNotFoundError(
                "401 Client Error. Repository Not Found",
                response=httpx.Response(
                    401,
                    request=httpx.Request(
                        "GET", "https://huggingface.co/api/models/acme/stembolts"
                    ),
                ),
            )

    monkeypatch.setattr("huggingface_hub.HfApi", NotFoundApi)

    result, captured = _invoke_add_readme(monkeypatch, [])

    message = _plain(result.output)
    assert "Could not find acme/stembolts" in message
    assert "may not exist yet, or your token may not have access" in message
    assert "does not exist yet" not in message
    # Unverifiable, so it falls through to asking, as before.
    assert captured["layout"] == "intrinsic"


def test_unreadable_flat_config_warns_and_reports_no_type(monkeypatch):
    """If the config cannot be fetched the README makes no claim about the type."""

    def failing_download(**_kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failing_download)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: _FakeHfApi(["adapter_config.json"])
    )

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    assert captured["layout"] == "flat"
    assert captured["adapter_type"] is None
    assert "could not read adapter_config.json" in _plain(result.output)


def test_unrecognized_repo_layout_asks_user(monkeypatch):
    """An empty/unrecognized repo prompts for the intended upload layout."""
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(["README.md"]))

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    # The stubbed `input` answers "yes" to the --intrinsic upload question.
    assert captured["layout"] == "intrinsic"
    assert "has no recognizable adapter layout" in result.output


def test_unknown_layout_without_terminal_fails_before_generation(monkeypatch):
    """With nobody to ask, an undetectable layout fails fast and names `--layout`.

    Waiting on `input()` here would hang or die with `EOFError` only after the
    slow README generation, so the command must stop before generating.
    """
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(["README.md"]))

    result, captured = _invoke_add_readme(monkeypatch, [], interactive=False)

    assert result.exit_code != 0
    message = _plain(result.output)
    assert "no terminal to ask" in message
    assert "--layout intrinsic" in message
    assert "--layout flat" in message
    assert captured == {}


def test_missing_token_without_terminal_fails_before_generation(monkeypatch):
    """No token means no repo inspection, which falls through to the same failure."""
    result, captured = _invoke_add_readme(
        monkeypatch, [], interactive=False, token=None
    )

    assert result.exit_code != 0
    assert "no terminal to ask" in _plain(result.output)
    assert captured == {}


@pytest.mark.parametrize(
    ("extra_args", "repo_files", "expected_layout"),
    [
        (["--layout", "flat"], None, "flat"),
        (["--layout", "intrinsic"], None, "intrinsic"),
        ([], ["stembolts/granite-4.1-3b/alora/adapter_config.json"], "intrinsic"),
        ([], ["adapter_config.json"], "flat"),
    ],
)
def test_no_terminal_needs_no_prompt_when_layout_is_known(
    monkeypatch, extra_args, repo_files, expected_layout
):
    """An explicit `--layout`, or a detectable repo, never needs a terminal."""
    if repo_files is not None:
        monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(repo_files))

    result, captured = _invoke_add_readme(monkeypatch, extra_args, interactive=False)

    assert result.exit_code == 0, result.output
    assert captured["layout"] == expected_layout


def test_layout_question_is_asked_again_after_an_invalid_answer(monkeypatch):
    """Anything but yes/no repeats the question rather than guessing a layout."""
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(["README.md"]))

    # "maybe" is rejected, "no" means a flat upload, and the last "yes" approves the README.
    result, captured = _invoke_add_readme(
        monkeypatch, [], answers=["maybe", "no", "yes"]
    )

    assert result.exit_code == 0, result.output
    assert "Please answer with only 'yes' or 'no'." in _plain(result.output)
    assert captured["layout"] == "flat"


def test_a_failed_listing_warns_and_asks_instead_of_failing(monkeypatch):
    """Offline, rate-limited, or any other Hub error must not abort the command."""

    class FailingApi:
        def model_info(self, **_kwargs):
            raise OSError("connection reset by peer")

    monkeypatch.setattr("huggingface_hub.HfApi", FailingApi)

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    message = _plain(result.output)
    assert "could not inspect the layout of acme/stembolts" in message
    assert "connection reset by peer" in message
    # Unverifiable, so it asks (the stubbed answer "yes" means an intrinsic upload).
    assert captured["layout"] == "intrinsic"


def test_unknown_layout_is_refused_when_stdin_is_really_not_a_terminal(monkeypatch):
    """With the real terminal check in place, a non-interactive run fails fast.

    `CliRunner` replaces standard input with a stream that is not a terminal, so
    this exercises the actual `isatty()` probe rather than a stand-in for it.
    """
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(["README.md"]))

    result, captured = _invoke_add_readme(monkeypatch, [], interactive=None)

    assert result.exit_code != 0
    assert "no terminal to ask" in _plain(result.output)
    assert captured == {}


def test_a_timeout_asks_which_layout_to_use(monkeypatch):
    """A listing that never answers must not hang: give up and ask.

    `list_repo_files` takes no timeout, so a connection that accepts and then
    goes quiet would block the command before anything is generated. The listing
    now goes through `model_info` with a 5 second limit.
    """
    import httpx

    calls: list[dict] = []

    class TimingOutApi:
        def model_info(self, **kwargs):
            calls.append(kwargs)
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("huggingface_hub.HfApi", TimingOutApi)

    result, captured = _invoke_add_readme(monkeypatch, [])

    assert result.exit_code == 0, result.output
    message = _plain(result.output)
    assert "timed out after 5 seconds" in message
    # The request itself must carry the limit; otherwise a silent server hangs it.
    assert calls[0]["timeout"] == 5
    # Unverifiable, so it asks (the stubbed answer "yes" means an intrinsic upload).
    assert captured["layout"] == "intrinsic"


def test_a_timeout_without_a_terminal_fails_before_generation(monkeypatch):
    """A timeout in a non-interactive run fails fast, naming `--layout`."""
    import httpx

    class TimingOutApi:
        def model_info(self, **_kwargs):
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("huggingface_hub.HfApi", TimingOutApi)

    result, captured = _invoke_add_readme(monkeypatch, [], interactive=False)

    assert result.exit_code != 0
    message = _plain(result.output)
    assert "timed out after 5 seconds" in message
    assert "no terminal to ask" in message
    assert captured == {}


def test_invalid_layout_rejected(monkeypatch):
    """An unsupported `--layout` value is a usage error before generation."""
    result, captured = _invoke_add_readme(monkeypatch, ["--layout", "bogus"])

    assert result.exit_code != 0
    assert "Unsupported layout" in result.output
    assert captured == {}
