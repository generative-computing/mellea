import typer

eval_app = typer.Typer(name="eval")


def eval_run(
    test_files: list[str] = typer.Argument(
        ..., help="List of paths to json/jsonl files containing test cases"
    ),
    backend: str = typer.Option("ollama", "--backend", "-b", help="Generation backend"),
    model: str = typer.Option(None, "--model", help="Generation model name"),
    judge_backend: str = typer.Option(
        None, "--judge-backend", "-jb", help="Judge backend"
    ),
    judge_model: str = typer.Option(None, "--judge-model", help="Judge model name"),
    output_path: str = typer.Option(
        "eval_results", "--output-path", "-o", help="Output path for results"
    ),
    output_format: str = typer.Option(
        "json", "--output-format", help="Either json or jsonl format for results"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error"),
):
    from cli.eval.runner import run_evaluations

    run_evaluations(
        test_files=test_files,
        backend=backend,
        model=model,
        judge_backend=judge_backend,
        judge_model=judge_model,
        output_path=output_path,
        output_format=output_format,
        verbose=verbose,
        continue_on_error=continue_on_error,
    )


eval_app.command("run")(eval_run)
