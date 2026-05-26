"""Use the eval command for LLM-as-a-judge evaluation, given a (set of) test file(s) consisting of prompts, instructions, and optionally, targets.
Instantiate a generator model to produce candidate responses, and a judge model to determine whether the instructions have been followed.
"""

import typer

eval_app = typer.Typer(name="eval", help="LLM-as-a-judge evaluation pipelines.")


def eval_run(
    test_files: list[str] = typer.Argument(
        ..., help="List of paths to json/jsonl files containing test cases"
    ),
    backend: str = typer.Option(
        "ollama",
        "--backend",
        "-b",
        help="Inference backend for generating candidate responses (e.g. ollama, openai)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Model name/id for the generation backend; uses backend default if omitted",
    ),
    max_gen_tokens: int = typer.Option(
        256, "--max-gen-tokens", help="Max tokens to generate for responses"
    ),
    judge_backend: str = typer.Option(
        None,
        "--judge-backend",
        "-jb",
        help="Inference backend for the judge model; reuses --backend if omitted",
    ),
    judge_model: str = typer.Option(
        None,
        "--judge-model",
        help="Model name/id for the judge; uses judge backend default if omitted",
    ),
    max_judge_tokens: int = typer.Option(
        256, "--max-judge-tokens", help="Max tokens for the judge model's judgement."
    ),
    output_path: str = typer.Option(
        "eval_results", "--output-path", "-o", help="Output path for results"
    ),
    output_format: str = typer.Option(
        "json", "--output-format", help="Either json or jsonl format for results"
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error",
        help="Skip failed test cases instead of aborting the entire run",
    ),
):
    """Run LLM-as-a-judge evaluation on one or more test files.

    Loads test cases from JSON/JSONL files, generates candidate responses using
    the specified generation backend, scores them with a judge model, and writes
    aggregated results to a file.

    Prerequisites:
        Mellea installed (``uv add mellea``). At least one inference backend
        available (Ollama by default). A separate judge backend/model is
        recommended but optional (defaults to the generation backend).

    Output:
        Writes evaluation results to ``<output-path>.<output-format>`` (default
        ``eval_results.json``). The file contains per-test-case scores, judge
        verdicts, and aggregate statistics.

    Examples:
        m eval run tests.jsonl --backend ollama --model granite3.3:2b

    See Also:
        guide: evaluation-and-observability/evaluate-with-llm-as-a-judge

    Args:
        test_files: Paths to JSON/JSONL files containing test cases.
        backend: Generation backend name.
        model: Generation model name, or ``None`` for the default.
        max_gen_tokens: Maximum tokens to generate for each response.
        judge_backend: Judge backend name, or ``None`` to reuse the generation
            backend.
        judge_model: Judge model name, or ``None`` for the default.
        max_judge_tokens: Maximum tokens for the judge model's output.
        output_path: File path prefix for the results file.
        output_format: Output format -- ``"json"`` or ``"jsonl"``.
        continue_on_error: If ``True``, skip failed tests instead of raising.
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
