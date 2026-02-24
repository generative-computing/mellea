import json
import os
import tempfile

import typer

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
    """Train an aLoRA or LoRA model on your dataset."""
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
        help="Formats model upload using granite-intrinsic. An io.yaml file must be provided.",
    ),
    io_yaml: str = typer.Option(
        default=None,
        help="Location of a granite-common io.yaml file. See https://nfulton.org/blog/alora_io_yaml.html",
    ),
):
    """Upload trained adapter to remote model registry."""
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

    print("✅ Upload complete!")


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
        help="Location of a granite-common io.yaml file. See https://nfulton.org/blog/alora_io_yaml.html",
    ),
):
    """Generate and upload an INTRINSIC_README.md for a trained adapter."""
    from huggingface_hub import HfFolder, create_repo, upload_file

    from cli.alora.readme_generator import generate_readme

    with tempfile.TemporaryDirectory() as tmp_dir:
        readme_path = os.path.join(tmp_dir, "README.md")
        generate_readme(
            dataset_path=datafile,
            base_model=basemodel,
            prompt_file=promptfile,
            output_path=readme_path,
            name=name,
            hints=open(hints).read() if hints is not None else None,
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

        token = HfFolder.get_token()
        if token is None:
            raise OSError(
                "Hugging Face token not found. Run `huggingface-cli login` first."
            )

        create_repo(repo_id=name, token=token, private=True, exist_ok=True)
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=name,
            commit_message="Upload intrinsic README.",
            token=token,
        )

    print(f"README uploaded to {name}")


alora_app.command("train")(alora_train)
alora_app.command("upload")(alora_upload)
alora_app.command("add-readme")(alora_add_readme)
