# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer sub-application for the `m alora` command group.

Provides three commands: `train` (fine-tune a base causal language model on a JSONL
dataset to produce a LoRA or aLoRA adapter), `upload` (push adapter weights to
Hugging Face Hub, optionally packaging the adapter as an intrinsic with an
`io.yaml` configuration), and `add-readme` (use an LLM to auto-generate and
upload an `INTRINSIC_README.md` for the trained adapter, with usage code matched
to the repository's detected adapter layout).
"""

import json
import os
import tempfile
from typing import Literal

import typer

from cli.alora.layout import (
    AdapterTypeName,
    RepoLayout,
    adapter_type_from_config,
    adapter_weight_dirs,
    detect_repo_layout,
    find_adapter_types,
)

alora_app = typer.Typer(
    name="alora", help="Train or upload aLoRAs for requirement validation."
)


def alora_train(
    datafile: str = typer.Argument(..., help="JSONL file with item/label pairs"),
    basemodel: str = typer.Option(..., help="Base model ID or path"),
    outfile: str = typer.Option(..., help="Path to save adapter weights"),
    promptfile: str = typer.Option(None, help="Path to load the prompt format file"),
    adapter: str = typer.Option("alora", help="Adapter type: alora or lora"),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or mps"),
    epochs: int = typer.Option(6, help="Number of training epochs"),
    learning_rate: float = typer.Option(6e-6, help="Learning rate"),
    batch_size: int = typer.Option(2, help="Per-device batch size"),
    max_length: int = typer.Option(1024, help="Max sequence length"),
    grad_accum: int = typer.Option(4, help="Gradient accumulation steps"),
):
    """Train an aLoRA or LoRA adapter on a labelled dataset.

    Fine-tunes a base causal language model using a JSONL dataset of item/label
    pairs. Supports both aLoRA (asymmetric LoRA) and standard LoRA adapters.

    Prerequisites:
        Mellea installed with adapter extras (`uv add mellea[adapters]`).
        A CUDA, MPS, or CPU device available for training.

    Output:
        Saves adapter weights to the path specified by `--outfile`. The output
        directory contains an `adapter_config.json` and the trained weight
        files, ready for upload or local inference.

    Examples:
        m alora train data.jsonl --basemodel ibm-granite/granite-3.3-2b-instruct --outfile ./adapter

    See Also:
        guide: advanced/lora-and-alora-adapters

    Args:
        datafile: JSONL file with item/label pairs for training.
        basemodel: Base model ID or path.
        outfile: Path to save adapter weights.
        promptfile: Path to load the prompt format file.
        adapter: Adapter type; `"alora"` or `"lora"`.
        device: Device to train on: `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Per-device training batch size.
        max_length: Maximum sequence length.
        grad_accum: Number of gradient accumulation steps.
    """
    from cli.alora.train import train_model

    train_model(
        dataset_path=datafile,
        base_model=basemodel,
        output_file=outfile,
        adapter=adapter,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        grad_accum=grad_accum,
        prompt_file=promptfile,
    )


def alora_upload(
    weight_path: str = typer.Argument(..., help="Path to saved adapter weights"),
    name: str = typer.Option(
        ..., help="Destination model name (e.g., acme/carbchecker-alora)"
    ),
    intrinsic: bool = typer.Option(
        default=False,
        help="True if the uploaded adapter implements an intrinsic. "
        "If true, the caller must provide an io.yaml file.",
    ),
    io_yaml: str = typer.Option(
        default=None,
        help="Location of the io.yaml file that configures input and output "
        "processing if the model is invoked as an intrinsic.",
    ),
):
    """Upload a trained adapter to a remote model registry.

    Pushes adapter weights to Hugging Face Hub, optionally packaging the adapter
    as an intrinsic with an `io.yaml` configuration file.

    Prerequisites:
        Hugging Face token available (set `HF_TOKEN` or run `hf auth login`).

    Output:
        Creates or updates a Hugging Face Hub repository at the name specified
        by `--name` and uploads the adapter weight files. Without `--intrinsic`
        the weights go to the repository root (flat layout), which PEFT can load
        but Mellea's intrinsic loading cannot; `--intrinsic` writes
        `<name>/<base-model>/<lora|alora>/` instead.

    Examples:
        m alora upload ./adapter --name acme/my-alora

    See Also:
        guide: advanced/lora-and-alora-adapters

    Args:
        weight_path: Path to saved adapter weights directory.
        name: Destination model name on Hugging Face Hub
            (e.g. `"acme/carbchecker-alora"`).
        intrinsic: If `True`, the adapter implements an intrinsic and an
            `io.yaml` file must also be provided.
        io_yaml: Path to the `io.yaml` file configuring input/output processing
            when the model is invoked as an intrinsic.
    """
    from cli.alora.intrinsic_uploader import upload_intrinsic
    from cli.alora.upload import upload_model

    assert not intrinsic or io_yaml, (
        "If --intrinsic is set then you must provide an io.yaml"
    )

    # Change the structure of the repo so that it's an intrinsic.
    if intrinsic:
        # get the base model and adapter type from the adapter config file.
        with open(os.path.join(weight_path, "adapter_config.json")) as fh:
            config = json.load(fh)
            assert "base_model_name_or_path" in config.keys(), (
                "All adapter config files should have a base_model_name_or_path."
            )
            base_model = config["base_model_name_or_path"]
            adapter_type = "alora" if "alora_invocation_tokens" in config else "lora"

        assert adapter_type in ["lora", "alora"]
        upload_intrinsic(
            weight_path=weight_path,
            model_name=name,
            base_model=base_model,
            type=adapter_type,  # type: ignore
            io_yaml=io_yaml,
        )
    else:
        upload_model(weight_path=weight_path, model_name=name)
        print(
            "⚠️  Uploaded with the flat layout: weights are at the repository root. "
            "Mellea's intrinsic loading expects "
            "`<name>/<base-model>/<lora|alora>/`, so this repository cannot be loaded "
            "as an adapter function. Re-run with --intrinsic --io-yaml ./io.yaml "
            "if that is what you intended."
        )

    print("✅ Upload complete!")


def _stdin_is_interactive() -> bool:
    """Report whether a person can answer prompts on standard input."""
    import sys

    return sys.stdin.isatty()


def _ask_adapter_layout(model_name: str) -> Literal["intrinsic", "flat"]:
    """Ask which upload layout the target repository uses.

    Args:
        model_name: Target Hugging Face repository name.

    Returns:
        `"intrinsic"` if the user will upload with `--intrinsic --io-yaml`,
        otherwise `"flat"`.

    Raises:
        typer.BadParameter: If standard input is not a terminal, so there is
            nobody to ask. Waiting on `input()` there would hang or die with
            `EOFError`, so fail before the slow README generation instead.
    """
    if not _stdin_is_interactive():
        raise typer.BadParameter(
            f"Cannot determine the adapter layout of {model_name} and there is "
            "no terminal to ask. Pass --layout intrinsic or --layout flat.",
            param_hint="--layout",
        )

    answer: str | None = None
    while answer not in ("yes", "no"):
        answer = (
            input(
                f"Will {model_name} be uploaded with "
                "`m alora upload ... --intrinsic --io-yaml ./io.yaml`? (yes/no) "
            )
            .strip()
            .lower()
        )
        if answer not in ("yes", "no"):
            print("Please answer with only 'yes' or 'no'.")
    return "intrinsic" if answer == "yes" else "flat"


def _read_flat_adapter_type(model_name: str, token: str) -> AdapterTypeName | None:
    """Read a flat repository's adapter type from its `adapter_config.json`.

    Downloads only that one small file, into a temporary directory that is
    removed afterwards, so nothing is left in the shared Hugging Face cache. A
    fresh directory also means a failed request cannot silently fall back to a
    stale cached copy. Best-effort like the rest of the inspection: any failure
    returns `None`, and the README then describes the adapter without claiming a
    type.

    Args:
        model_name: Target Hugging Face repository name.
        token: Hugging Face authentication token.

    Returns:
        `"alora"` or `"lora"`, or `None` if the config could not be read.
    """
    from huggingface_hub import hf_hub_download

    print(f"Reading adapter_config.json from {model_name} (read-only, temporary copy).")
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="adapter_config.json",
                token=token,
                cache_dir=cache_dir,
            )
            with open(config_path, encoding="utf-8") as fh:
                return adapter_type_from_config(json.load(fh))
    except Exception as e:
        print(f"Warning: could not read adapter_config.json from {model_name}: {e}")
        return None


def _inspect_repo(
    model_name: str, token: str, base_model: str
) -> tuple[RepoLayout, AdapterTypeName | None]:
    """Inspect a Hub repository for its adapter layout and adapter type.

    Verification is best-effort: a repository that does not exist yet (the
    README may be generated before the weights are uploaded) and any other Hub
    error both return `"unknown"`, so the caller can ask the user instead of
    failing.

    For the intrinsic layout, also checks that weights exist where the loader
    will look for them given `base_model`, and warns when they do not. The
    generated code resolves weights from the backend's base model, so a nested
    upload made for a different base model would still fail to load.

    Args:
        model_name: Target Hugging Face repository name.
        token: Hugging Face authentication token.
        base_model: Base model the README's code will load the adapter against.

    Returns:
        The detected layout, or `"unknown"` if it cannot be determined, and the
        adapter type. For the intrinsic layout that is the type found for
        `base_model` (`None` if no weights exist for it; `"alora"` when both a
        LoRA and an aLoRA adapter are present). For the flat layout it is read
        from the root `adapter_config.json` (`None` if that cannot be read).
    """
    # `alora_add_readme`, the only caller, has already imported `huggingface_hub`
    # and raised the friendly missing-extra error if it is absent.
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    print(
        f"Checking {model_name} on Hugging Face (read-only) to detect its adapter layout."
    )
    try:
        # `list_repo_files` takes no timeout, so a connection that accepts and
        # then goes quiet (a dead proxy, a captive portal) would hang the command
        # before anything is generated. `model_info` returns the same file list
        # through `siblings` and does take a timeout.
        info = HfApi().model_info(repo_id=model_name, token=token, timeout=5)
        repo_files = [sibling.rfilename for sibling in info.siblings or []]
    except RepositoryNotFoundError:
        # Hugging Face also raises this for a private repository the token cannot
        # see, so it does not prove the repository is missing.
        print(
            f"Could not find {model_name} on Hugging Face: it may not exist yet, or "
            "your token may not have access to it. Its adapter layout is unknown."
        )
        return "unknown", None
    except httpx.TimeoutException:
        print(
            f"Warning: checking {model_name} on Hugging Face timed out after 5 "
            "seconds. Its adapter layout is unknown."
        )
        return "unknown", None
    except Exception as e:
        print(f"Warning: could not inspect the layout of {model_name}: {e}")
        return "unknown", None

    intrinsic_name = model_name.split("/")[-1]
    detected = detect_repo_layout(repo_files, intrinsic_name)
    if detected == "unknown":
        print(f"{model_name} has no recognizable adapter layout.")
        return detected, None
    print(f"Detected the {detected} adapter layout in {model_name}.")
    if detected == "flat":
        return detected, _read_flat_adapter_type(model_name, token)

    found = find_adapter_types(repo_files, intrinsic_name, base_model, model_name)
    if not found:
        expected = ", ".join(
            f"{directory}/"
            for directory in adapter_weight_dirs(
                intrinsic_name, base_model, model_name
            ).values()
        )
        print(
            f"Warning: {model_name} has no weights for base model '{base_model}'. "
            f"The generated code looks in {expected} and will not find this "
            "adapter. Pass the --basemodel you uploaded with, or re-upload for "
            "this base model."
        )
        return detected, None
    if len(found) > 1:
        print(
            f"{model_name} has both lora and alora weights for '{base_model}'; "
            "documenting alora."
        )
        return detected, "alora"
    return detected, found[0]


def alora_add_readme(
    datafile: str = typer.Argument(..., help="JSONL file with item/label pairs"),
    basemodel: str = typer.Option(..., help="Base model ID or path"),
    promptfile: str = typer.Option(None, help="Path to load the prompt format file"),
    name: str = typer.Option(
        ..., help="Destination model name (e.g., acme/carbchecker-alora)"
    ),
    hints: str = typer.Option(
        default=None, help="File containing any additional hints."
    ),
    io_yaml: str = typer.Option(
        default=None,
        help="Location of the io.yaml file that configures input and output "
        "processing if the model is invoked as an intrinsic.",
    ),
    layout: str = typer.Option(
        "auto",
        help="Adapter layout of the target repository: 'auto' looks it up on "
        "Hugging Face with read-only requests and asks when it cannot be "
        "verified, 'intrinsic' is <name>/<base-model>/<lora|alora>/, and 'flat' "
        "is the repository root. 'intrinsic' and 'flat' make no requests.",
    ),
):
    """Generate and upload an INTRINSIC_README.md for a trained adapter.

    Uses an LLM to auto-generate documentation for a trained adapter based on
    the training data and model configuration, then uploads it to the Hugging
    Face Hub repository.

    The generated usage code depends on how the adapter was published. The
    command detects the repository's layout by default (weights nested under
    `<name>/<base-model>/<lora|alora>/` versus flat at the root) and renders
    matching code. Detection makes read-only requests to Hugging Face with your
    token, before anything is generated or uploaded: a listing of the
    repository's files and, for a flat repository, its small
    `adapter_config.json`, which is downloaded to a temporary directory and
    removed. If the layout cannot be verified — for example the README is
    generated before the weights are uploaded — it asks which layout to use, or
    exits with an error when there is no terminal to ask on. Pass `--layout` to
    skip detection and those requests, for example in scripts and CI. Without a
    token, `auto` skips the requests and asks which layout you intend to upload
    instead; a token is only required to upload the README.

    Prerequisites:
        Hugging Face token available (set `HF_TOKEN` or run `hf auth login`).
        An LLM backend available for README generation.

    Output:
        Generates a README.md file, displays it for confirmation, and uploads
        it to the Hugging Face Hub repository specified by `--name`.

    Examples:
        m alora add-readme data.jsonl --basemodel ibm-granite/granite-3.3-2b-instruct --name acme/my-alora

    See Also:
        guide: advanced/lora-and-alora-adapters

    Args:
        datafile: JSONL file with item/label pairs used to train the adapter.
        basemodel: Base model ID or path.
        promptfile: Path to the prompt format file, or `None`.
        name: Destination model name on Hugging Face Hub.
        hints: Path to a file containing additional domain hints, or `None`.
        io_yaml: Path to the `io.yaml` intrinsic configuration file, or `None`.
        layout: Adapter layout of the target repository — `"auto"`, `"intrinsic"`,
            or `"flat"`.

    Raises:
        OSError: If no Hugging Face authentication token is found.
        typer.BadParameter: If `layout` is not `"auto"`, `"intrinsic"`, or `"flat"`,
            or if it is `"auto"`, the repository's layout cannot be detected, and
            standard input is not a terminal to ask on.
        SystemExit: If the user declines to upload the generated README.
    """
    try:
        from huggingface_hub import create_repo, get_token, upload_file
    except ImportError as e:
        raise ImportError(
            "The 'm alora' command requires extra dependencies. "
            'Please install them with: pip install "mellea[hf]"'
        ) from e

    from cli.alora.readme_generator import generate_readme

    if layout not in ("auto", "intrinsic", "flat"):
        raise typer.BadParameter(
            f"Unsupported layout '{layout}'. Choose from: auto, intrinsic, flat."
        )

    # A token is only required to upload; keep local, unauthenticated generation
    # working as before.
    token = get_token()

    resolved_layout: Literal["intrinsic", "flat"]
    # Only inspecting the repository can tell lora from alora; the generator
    # decides what to document when it could not.
    detected_adapter_type: AdapterTypeName | None = None
    if layout == "auto":
        if token is None:
            print(
                "Warning: no Hugging Face token found, so the repository layout "
                "cannot be inspected."
            )
            detected: RepoLayout = "unknown"
        else:
            detected, detected_adapter_type = _inspect_repo(name, token, basemodel)
        if detected == "unknown":
            resolved_layout = _ask_adapter_layout(name)
        else:
            resolved_layout = detected
    elif layout == "intrinsic":
        resolved_layout = "intrinsic"
    else:
        resolved_layout = "flat"

    with tempfile.TemporaryDirectory() as tmp_dir:
        readme_path = os.path.join(tmp_dir, "README.md")
        generate_readme(
            dataset_path=datafile,
            base_model=basemodel,
            prompt_file=promptfile,
            output_path=readme_path,
            name=name,
            hints=open(hints).read() if hints is not None else None,
            layout=resolved_layout,
            adapter_type=detected_adapter_type,
        )

        print(open(readme_path).read())
        continue_answer: str | None = None
        while continue_answer is None or continue_answer not in ["yes", "no"]:
            if continue_answer is not None:
                print("Please answer with only 'yes' or 'no'.")
            answer = input(
                f"\nWe auto-generated a README using Mellea. Should we upload this README to {name} (yes/no)? "
            )
            continue_answer = answer.strip().lower()
        if continue_answer == "no":
            print("ABORTING.")
            import sys

            sys.exit(-1)
        else:
            assert continue_answer == "yes"

        if token is None:
            raise OSError(
                "Hugging Face token not found. Set `HF_TOKEN` or run `hf auth login` first."
            )

        create_repo(repo_id=name, token=token, private=True, exist_ok=True)
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=name,
            commit_message=(
                "Upload intrinsic README."
                if resolved_layout == "intrinsic"
                else "Upload README."
            ),
            token=token,
        )

    print(f"README uploaded to {name}")


alora_app.command("train")(alora_train)
alora_app.command("upload")(alora_upload)
alora_app.command("add-readme")(alora_add_readme)
