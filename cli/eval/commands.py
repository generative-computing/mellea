"""Use the eval command for LLM-as-a-judge evaluation, given a (set of) test file(s) consisting of prompts, instructions, and optionally, targets.
Instantiate a generator model to produce candidate responses, and a judge model to determine whether the instructions have been followed.
"""

import typer

eval_app = typer.Typer(name="eval")


def eval_run(
    test_files: list[str] = typer.Argument(
        ..., help="List of paths to json/jsonl files containing test cases"
    ),
    backend: str = typer.Option("ollama", "--backend", "-b", help="Generation backend"),
    model: str = typer.Option(None, "--model", help="Generation model name"),
    max_gen_tokens: int = typer.Option(
        256, "--max-gen-tokens", help="Max tokens to generate for responses"
    ),
    judge_backend: str = typer.Option(
        None, "--judge-backend", "-jb", help="Judge backend"
    ),
    judge_model: str = typer.Option(None, "--judge-model", help="Judge model name"),
    max_judge_tokens: int = typer.Option(
        256, "--max-judge-tokens", help="Max tokens for the judge model's judgement."
    ),
    output_path: str = typer.Option(
        "eval_results", "--output-path", "-o", help="Output path for results"
    ),
    output_format: str = typer.Option(
        "json", "--output-format", help="Either json or jsonl format for results"
    ),
    continue_on_error: bool = typer.Option(True, "--continue-on-error"),
):
    """Run LLM-as-a-judge evaluation on one or more test files.

    Loads test cases from JSON/JSONL files, generates candidate responses using
    the specified generation backend, scores them with a judge model, and writes
    aggregated results to a file.

    Args:
        test_files: Paths to JSON/JSONL files containing test cases.
        backend: Generation backend name.
        model: Generation model name, or `None` for the default.
        max_gen_tokens: Maximum tokens to generate for each response.
        judge_backend: Judge backend name, or `None` to reuse the generation
            backend.
        judge_model: Judge model name, or `None` for the default.
        max_judge_tokens: Maximum tokens for the judge model's output.
        output_path: File path prefix for the results file.
        output_format: Output format -- `"json"` or `"jsonl"`.
        continue_on_error: If `True`, skip failed tests instead of raising.
    """
    from cli.eval.runner import run_evaluations

    run_evaluations(
        test_files=test_files,
        backend=backend,
        model=model,
        max_gen_tokens=max_gen_tokens,
        judge_backend=judge_backend,
        judge_model=judge_model,
        max_judge_tokens=max_judge_tokens,
        output_path=output_path,
        output_format=output_format,
        continue_on_error=continue_on_error,
    )


eval_app.command("run")(eval_run)
