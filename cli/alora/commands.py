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
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        grad_accum=grad_accum,
        prompt_file=promptfile,
    )


def alora_upload(
    weightfile: str = typer.Argument(..., help="Path to saved adapter weights"),
    name: str = typer.Option(
        ..., help="Destination model name (e.g., acme/carbchecker-alora)"
    ),
):
    """Upload trained adapter to remote model registry."""
    from cli.alora.upload import upload_model

    upload_model(weight_path=weightfile, model_name=name)


alora_app.command("train")(alora_train)
alora_app.command("upload")(alora_upload)
