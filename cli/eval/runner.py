# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution engine for the test-based LLM evaluation pipeline.

Loads JSON test files into `TestBasedEval` objects and, for each test, runs a
generator model to produce responses and a separate judge model to score them. Parses
the judge output for a `{"score": ..., "justification": ...}` JSON fragment,
aggregates per-input pass/fail counts, and saves the full results to JSON or JSONL.
"""

import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

import mellea
from mellea.backends import ModelOption
from mellea.backends.backend import Backend
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.components.unit_test_eval import TestBasedEval
from mellea.stdlib.start_backend import _resolve_model_id_str

console = Console()


class InputEvalResult:
    """Store results of a single input evaluation (within a unit test).

    Args:
        input_text (str): The raw input text sent to the generation model.
        model_output (str): The text response produced by the generation model.
        validation_passed (bool): Whether the judge scored this response as passing.
        score (int): Numeric score assigned by the judge (`1` for pass, `0` for fail).
        validation_reason (str): Justification text returned by the judge model.

    """

    def __init__(
        self,
        input_text: str,
        model_output: str,
        validation_passed: bool,
        score: int,
        validation_reason: str,  # add input_id
    ):
        self.input_text = input_text
        self.model_output = model_output
        self.validation_passed = validation_passed
        self.score = score
        self.validation_reason = validation_reason

    def to_dict(self) -> dict:
        """Serialise the input evaluation result to a plain dictionary.

        Returns:
            dict: A dictionary with keys `"input"`, `"model_output"`,
            `"passed"`, `"score"`, and `"justification"`.
        """
        return {
            "input": self.input_text,
            "model_output": self.model_output,
            "passed": self.validation_passed,
            "score": self.score,
            "justification": self.validation_reason,
        }


class TestEvalResult:
    """Store results of a single test evaluation.

    Args:
        test_eval (TestBasedEval): The unit test specification containing
            the test ID, name, instructions, inputs, and expected targets.
        input_results (list[InputEvalResult]): Per-input evaluation outcomes
            produced by running the generation and judge models.

    Attributes:
        passed_count (int): Number of inputs that received a passing score.
        total_count (int): Total number of inputs evaluated.
        pass_rate (float): Fraction of inputs that passed (`passed_count / total_count`).
    """

    def __init__(self, test_eval: TestBasedEval, input_results: list[InputEvalResult]):
        self.test_eval = test_eval
        self.input_results = input_results

    def to_dict(self) -> dict:
        """Serialise the test evaluation result to a plain dictionary.

        Returns:
            dict: A dictionary containing the test metadata (`"test_id"`,
            `"source"`, `"name"`, `"instructions"`), per-input results
            under `"input_results"`, expected targets under
            `"expected_targets"`, and summary counts (`"passed"`,
            `"total_count"`, `"pass_rate"`).
        """
        return {
            "test_id": self.test_eval.test_id,
            "source": self.test_eval.source,
            "name": self.test_eval.name,
            "instructions": self.test_eval.instructions,
            "input_results": [r.to_dict() for r in self.input_results],
            "expected_targets": self.test_eval.targets,
            "passed": self.passed_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
        }

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.input_results if r.validation_passed)

    @property
    def total_count(self) -> int:
        return len(self.input_results)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0


def _auto_thinking(backend_instance: Backend, backend_name: str) -> bool | None:
    """Choose the eval thinking default from the built backend's own state.

    Reasoning counts against eval's small token budget, so thinking is turned
    off wherever the backend supports it.

    Args:
        backend_instance: The constructed backend.
        backend_name: Lower-case backend name.

    Returns:
        The value for `ModelOption.THINKING`, or `None` to leave thinking at
        the model's default.
    """
    from mellea.helpers.server_type import _ServerType

    if backend_name in ("ollama", "hf", "huggingface"):
        return False
    if backend_name == "openai":
        # On the OpenAI platform a bool adds `chat_template_kwargs`, so leave it.
        server_type = getattr(backend_instance, "_server_type")
        return None if server_type == _ServerType.OPENAI else False
    if backend_name == "litellm":
        # LiteLLM maps THINKING only for Ollama-routed models.
        model_id = str(getattr(backend_instance, "_model_id"))
        return False if "ollama" in model_id.split("/")[0] else None
    return None  # watsonx has no thinking mapping


def create_session(
    backend: str,
    model: str | None,
    max_tokens: int | None,
    thinking: bool | None = None,
) -> mellea.MelleaSession:
    """Create a mellea session with the specified backend and model.

    Args:
        backend: Backend name: `"ollama"`, `"openai"`, `"hf"`,
            `"watsonx"`, or `"litellm"`.
        model: Model ID or `ModelIdentifier` attribute name, or `None`
            to let the backend choose its own default model. The `"hf"`
            backend has no default of its own, so `None` selects Granite 4.2 3B.
        max_tokens: Maximum number of tokens to generate, or `None` for
            the backend default.
        thinking: Force reasoning on or off, or `None` for the automatic
            per-backend default. Ignored with a warning on `watsonx`.

    Returns:
        A configured `MelleaSession` ready for generation.

    Raises:
        ValueError: If `backend` is not one of the supported backend names, or
            if `backend` is `"openai"`, `model` is `None`, and
            `OPENAI_BASE_URL` is not the OpenAI platform.
        Exception: Re-raised from backend or session construction if
            initialisation fails.
    """
    model_id: str | mellea.model_ids.ModelIdentifier | None = None
    if model:
        if model.isupper() or "_" in model:
            if hasattr(mellea.model_ids, model):
                model_id = getattr(mellea.model_ids, model)
            else:
                model_id = model
        else:
            model_id = model

    # Only forward a model when the user named one, so each backend otherwise
    # applies its own default instead of a value forced by eval.
    model_kwargs: dict[str, Any] = {} if model_id is None else {"model_id": model_id}

    try:
        backend_lower = backend.lower()
        backend_instance: Backend

        if backend_lower == "ollama":
            from mellea.backends.ollama import OllamaModelBackend

            backend_instance = OllamaModelBackend(
                **model_kwargs, model_options={ModelOption.MAX_NEW_TOKENS: max_tokens}
            )

        elif backend_lower == "openai":
            from mellea.backends.openai import OpenAIBackend
            from mellea.helpers.server_type import _ServerType

            openai_backend = OpenAIBackend(
                **model_kwargs, model_options={ModelOption.MAX_NEW_TOKENS: max_tokens}
            )
            if model_id is None and openai_backend._server_type != _ServerType.OPENAI:
                raise ValueError(
                    "No --model given and OPENAI_BASE_URL "
                    f"({openai_backend._base_url}) is not the OpenAI platform, "
                    f"where the default {openai_backend._model_id} lives. "
                    "Pass --model or --judge-model."
                )
            backend_instance = openai_backend

        elif backend_lower in ["hf", "huggingface"]:
            from mellea.backends.huggingface import LocalHFBackend

            if model_id is None:
                # LocalHFBackend requires a model, so eval supplies the default.
                console.print(
                    "No model given for the hf backend; using Granite 4.2 3B "
                    "(downloaded on first use)."
                )
                model_id = mellea.model_ids.IBM_GRANITE_4_2_3B
            backend_instance = LocalHFBackend(
                model_id=model_id,
                model_options={ModelOption.MAX_NEW_TOKENS: max_tokens},
            )

        elif backend_lower == "watsonx":
            from mellea.backends.watsonx import WatsonxAIBackend

            backend_instance = WatsonxAIBackend(
                **model_kwargs, model_options={ModelOption.MAX_NEW_TOKENS: max_tokens}
            )

        elif backend_lower == "litellm":
            from mellea.backends.litellm import LiteLLMBackend

            # LiteLLM takes a plain string, so convert a resolved ModelIdentifier.
            litellm_kwargs: dict[str, Any] = (
                {} if model_id is None else {"model_id": str(model_id)}
            )
            backend_instance = LiteLLMBackend(
                **litellm_kwargs, model_options={ModelOption.MAX_NEW_TOKENS: max_tokens}
            )

        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: ollama, openai, hf, watsonx, litellm"
            )

        # THINKING is read at request time, so it can be set after construction.
        if thinking is None:
            auto_thinking = _auto_thinking(backend_instance, backend_lower)
            if auto_thinking is not None:
                backend_instance.model_options[ModelOption.THINKING] = auto_thinking
        elif backend_lower == "watsonx":
            console.print(
                "[yellow]warning:[/yellow] watsonx has no thinking mapping; "
                "ignoring --thinking/--no-thinking."
            )
        else:
            backend_instance.model_options[ModelOption.THINKING] = thinking

        # create session with backend instance
        from mellea.stdlib.context import SimpleContext

        session = mellea.MelleaSession(backend=backend_instance, ctx=SimpleContext())
        return session

    except Exception as e:
        console.print(
            f"[red]Error creating session with backend={backend}, model={model or 'backend default'}: {e}[/red]"
        )
        raise


def _resolved_model_name(session: mellea.MelleaSession, backend_name: str) -> str:
    """Return the model name a session's backend will actually request.

    Args:
        session: A session built by `create_session`.
        backend_name: The backend name the session was created with.

    Returns:
        The model name as the backend sends it, or `"unknown"` if the backend
        does not expose one.
    """
    model_id = getattr(session.backend, "model_id", None)
    if model_id is None:
        return "unknown"
    return _resolve_model_id_str(model_id, backend_name.lower())


def run_evaluations(
    test_files: list[str],
    backend: str,
    model: str | None,
    max_gen_tokens: int | None,
    judge_backend: str | None,
    judge_model: str | None,
    max_judge_tokens: int | None,
    output_path: str,
    output_format: str,
    continue_on_error: bool,
    thinking: bool | None = None,
):
    """Run all unit-test evaluations against a generation model and a judge model.

    Args:
        test_files: List of paths to JSON test files. Each file should contain
            `"id"`, `"source"`, `"name"`, `"instructions"`, and
            `"examples"` fields.
        backend: Backend name for the generation model.
        model: Model ID for the generator, or `None` for the default.
        max_gen_tokens: Maximum tokens for the generator, or `None` for the
            backend default.
        judge_backend: Backend name for the judge model, or `None` to reuse
            the generation backend.
        judge_model: Model ID for the judge, or `None` for the default.
        max_judge_tokens: Maximum tokens for the judge, or `None` for the
            backend default.
        output_path: File path prefix for saving results.
        output_format: Output format: `"json"` or `"jsonl"`.
        continue_on_error: If `True`, skip failed test evaluations instead of
            raising.
        thinking: Force reasoning on or off for both the generation and judge
            models, or `None` for the automatic per-backend default.
    """
    all_test_evals: list[TestBasedEval] = []

    for test_file in test_files:
        try:
            test_evals = TestBasedEval.from_json_file(test_file)
            all_test_evals.extend(test_evals)
            console.print(f"Loaded {len(test_evals)} test evaluations from {test_file}")
        except Exception:
            console.print(f"Error loading {test_file}")

    if not all_test_evals:
        console.print("Failed to load any test evaluations")
        return

    console.print(f"Total test evals to run: {len(all_test_evals)}")
    total_inputs = sum(len(test_eval.inputs) for test_eval in all_test_evals)
    console.print(f"Total inputs to run: {total_inputs}")

    m = create_session(
        backend=backend, model=model, max_tokens=max_gen_tokens, thinking=thinking
    )
    # Use same backend as generator if judge_backend not specified
    judge_backend_name = judge_backend if judge_backend else backend
    judge_session = create_session(
        backend=judge_backend_name,
        model=judge_model,
        max_tokens=max_judge_tokens,
        thinking=thinking,
    )

    # Report what the backends resolved, since an omitted model is chosen by the backend.
    console.print(f"Generation model: {_resolved_model_name(m, backend)}")
    console.print(
        f"Judge model: {_resolved_model_name(judge_session, judge_backend_name)}"
    )

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Running evals", total=len(all_test_evals))
        for test_eval in all_test_evals:
            try:
                result = execute_test_eval(
                    test_eval=test_eval,
                    generation_session=m,
                    judge_session=judge_session,
                )
                all_results.append(result)
            except Exception as e:
                console.print(f"Error {e} on test {test_eval.test_id}")
                if not continue_on_error:
                    raise

            progress.advance(task)

    summary_stats(all_results)
    save_results(all_results, output_path, output_format)

    m.cleanup()
    judge_session.cleanup()


def execute_test_eval(
    test_eval: TestBasedEval,
    generation_session: mellea.MelleaSession,
    judge_session: mellea.MelleaSession,
) -> TestEvalResult:
    """Execute a single test evaluation.

    For each input in the test, generates a response using `generation_session`,
    then validates using `judge_session`.

    Args:
        test_eval: The `TestBasedEval` object containing inputs and targets.
        generation_session: `MelleaSession` used to produce model responses.
        judge_session: `MelleaSession` used to score model responses.

    Returns:
        A `TestEvalResult` with per-input pass/fail outcomes.
    """
    input_results = []

    # for all inputs, generate responses with generator
    for idx, input_text in enumerate(test_eval.inputs):
        result: ModelOutputThunk = generation_session.act(
            SimpleComponent(instruction=input_text)
        )
        model_output = str(result)

        targets_for_input = (
            test_eval.targets[idx] if idx < len(test_eval.targets) else []
        )

        # query the judge
        test_eval.set_judge_context(
            input_text=input_text,
            prediction=model_output,
            targets_for_input=targets_for_input,
        )
        judge_output_thunk = judge_session.act(test_eval)
        judge_output = str(judge_output_thunk)
        score, justification = parse_judge_output(judge_output)
        passed = score == 1 if score is not None else False

        input_result = InputEvalResult(
            input_text=input_text,
            model_output=model_output,
            validation_passed=passed,
            score=score if score is not None else 0,
            validation_reason=justification,
        )
        input_results.append(input_result)

        # reset both generator and judge
        generation_session.reset()
        judge_session.reset()

    test_result = TestEvalResult(test_eval=test_eval, input_results=input_results)
    return test_result


def _extract_first_json(text: str) -> dict | None:
    """Return the first JSON object containing a `"score"` key, or `None`."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                if "score" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def parse_judge_output(judge_output: str) -> tuple[int | None, str]:
    """Parse score and justification from a judge model's output string.

    Args:
        judge_output: Raw text output from the judge model.

    Returns:
        A `(score, justification)` tuple where `score` is an integer (or
        `None` if parsing failed) and `justification` is an explanatory
        string.
    """
    data = _extract_first_json(judge_output)
    if data is not None:
        score = data.get("score")
        justification = data.get("justification")
        return score, (
            justification if isinstance(justification, str) else judge_output
        )

    # if the above fails, search the text for the score
    score_match = re.search(r'score["\s:]+(\d+)', judge_output, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        return score, judge_output

    return None, judge_output


def save_results(results: list[TestEvalResult], output_path: str, output_format: str):
    """Persist evaluation results to disk in JSON or JSONL format.

    Args:
        results: List of `TestEvalResult` objects to serialise.
        output_path: Destination file path (extension may be appended if it
            does not match `output_format`).
        output_format: Format string: `"json"` or `"jsonl"`.
    """
    output_path_obj = Path(output_path)
    if output_path_obj.suffix != f".{output_format}":
        output_path_obj = Path(f"{output_path}.{output_format}")

    total_inputs = sum(r.total_count for r in results)
    passed_inputs = sum(r.passed_count for r in results)
    overall_pass_rate = passed_inputs / total_inputs if total_inputs > 0 else 0.0

    if output_format == "jsonl":
        with output_path_obj.open("w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")
    else:  # json
        summary = {
            "total_tests": len(results),
            "total_inputs": total_inputs,
            "passed_inputs": passed_inputs,
            "failed_inputs": total_inputs - passed_inputs,
            "overall_pass_rate": overall_pass_rate,
        }

        with output_path_obj.open("w") as f:
            json.dump(
                {"summary": summary, "results": [r.to_dict() for r in results]},
                f,
                indent=2,
            )

    console.print(f"Results saved to {output_path}")


def summary_stats(results: list[TestEvalResult]):
    """Print aggregated pass-rate statistics for a set of evaluation results.

    Args:
        results: List of `TestEvalResult` objects to summarise.
    """
    total_inputs = sum(r.total_count for r in results)
    passed_inputs = sum(r.passed_count for r in results)
    overall_pass_rate = passed_inputs / total_inputs if total_inputs > 0 else 0.0

    console.print(f"Total number of inputs across tests: {total_inputs}")
    console.print(f"Number of inputs passed across tests: {passed_inputs}")
    console.print(f"Cumulative Pass Rate: {overall_pass_rate * 100:.1f}%")

    if len(results) > 1:
        console.print("Per-Test Breakdown:")
        for result in results:
            console.print(
                f"{result.test_eval.name}:\n\t{result.passed_count}/{result.total_count} ({result.pass_rate * 100:.1f}%)\n\n"
            )
