import json
import re
from pathlib import Path
from typing import List

import mellea
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.requirement import Requirement
from mellea.stdlib.test_based_eval import TestBasedEval
from mellea.backends.types import ModelOption

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class InputEvalResult:
    """Store results of a single input evaluation (within a unit test)"""

    def __init__(
        self,
        input_text: str,
        model_output: str,
        validation_passed: bool,
        score: int,
        validation_reason: str,
    ):
        self.input_text = input_text
        self.model_output = model_output
        self.validation_passed = validation_passed
        self.score = score
        self.validation_reason = validation_reason

    def to_dict(self):
        return {
            "input": self.input_text,
            "model_output": self.model_output,
            "passed": self.validation_passed,
            "score": self.score,
            "justification": self.validation_reason,
        }


class TestEvalResult:
    """Store results of a single test evaluation"""

    def __init__(self, test_eval: TestBasedEval, input_results: list[InputEvalResult]):
        self.test_eval = test_eval
        self.input_results = input_results

    def to_dict(self):
        return {
            "conversation_id": self.test_eval.conversation_id,
            "category": self.test_eval.category,
            "input_results": [r.to_dict() for r in self.input_results],
            "expected_targets": self.test_eval.targets,
            "unit_test_instructions": self.test_eval.unit_test_instructions,
            "passed": self.passed_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
            "metadata": self.test_eval.metadata,
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


def create_session(backend: str, model: str | None) -> mellea.MelleaSession:
    """Create a mellea session with the specified backend and  model."""

    model_id = None
    if model:
        if model.isupper() or "_" in model:
            if hasattr(mellea.model_ids, model):
                model_id = getattr(mellea.model_ids, model)
            else:
                model_id = model
        else:
            model_id = model
    else:
        model_id = mellea.model_ids.IBM_GRANITE_4_MICRO_3B

    try:
        backend_lower = backend.lower()

        if backend_lower == "ollama":
            from mellea.backends.ollama import OllamaModelBackend

            backend_instance = OllamaModelBackend(
                model_id=model_id, model_options={ModelOption.MAX_NEW_TOKENS: 256}
            )

        elif backend_lower == "openai":
            from mellea.backends.openai import OpenAIBackend

            backend_instance = OpenAIBackend(
                model_id=model_id, model_options={ModelOption.MAX_NEW_TOKENS: 256}
            )

        elif backend_lower in ["hf", "huggingface"]:
            from mellea.backends.huggingface import LocalHFBackend

            backend_instance = LocalHFBackend(
                model_id=model_id, model_options={ModelOption.MAX_NEW_TOKENS: 256}
            )

        elif backend_lower == "watsonx":
            from mellea.backends.watsonx import WatsonxAIBackend

            backend_instance = WatsonxAIBackend(
                model_id=model_id, model_options={ModelOption.MAX_NEW_TOKENS: 256}
            )

        elif backend_lower == "litellm":
            from mellea.backends.litellm import LiteLLMBackend

            backend_instance = LiteLLMBackend(
                model_id=model_id, model_options={ModelOption.MAX_NEW_TOKENS: 256}
            )

        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: ollama, openai, hf, watsonx, litellm"
            )

        # create session with backend instance
        from mellea.stdlib.base import SimpleContext

        session = mellea.MelleaSession(
            backend=backend_instance, ctx=SimpleContext()
        )  # need to reset to SimpleContext? print what is being judged by the judge (input)
        return session

    except Exception as e:
        console.print(
            f"[red]Error creating session with backend={backend}, model={model_id}: {e}[/red]"
        )
        raise


def run_evaluations(
    test_files: List[str],
    backend: str,
    model: str | None,
    judge_backend: str | None,
    judge_model: str | None,
    output_path: str,
    output_format: str,
    verbose: bool,
    continue_on_error: bool,
):
    """Run all 'unit test' evaluations"""
    all_test_evals: List[TestBasedEval] = []

    for test_file in test_files:
        try:
            test_evals = TestBasedEval.from_json_file(test_file)
            all_test_evals.extend(test_evals)
            console.print(f"Loaded {len(test_evals)} test evaluations from {test_file}")
        except Exception as e:
            console.print(f"Error loading {test_file}")

    if not all_test_evals:
        console.print("Failed to load any test evaluations")
        return

    console.print(f"Total test evals to run: {len(all_test_evals)}")

    console.print(f"Generation model: {model}")
    console.print(f"Judge model: {judge_model}")

    m = create_session(backend=backend, model=model)
    judge_session = create_session(backend=judge_backend, model=judge_model)

    all_results = []

    # some visuals on progress with rich, we can take out / modify
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
                console.print(f"Error {e} on test {test_eval.conversation_id}")
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
    """Execute a single test evaluation
    For each input in the test, generate a response using generation_session
    Then, after all inputs are processed, validate using judge_session
    """

    input_results = []

    # for all inputs, generate responses with generator
    for input_text in test_eval.inputs:
        result: ModelOutputThunk = generation_session.act(input_text)
        model_output = str(result)
        console.print(model_output)

        judge_session.ctx = judge_session.ctx.add(result)

        requirement = Requirement(
            description=create_judge_requirement(test_eval, input_text, model_output)
        )
        validation_results = judge_session.validate(requirement)
        validation_result = validation_results[0]

        judge_output = validation_result.reason or ""
        score, justification = parse_judge_output(judge_output)

        passed = score == 1 if score is not None else validation_result.as_bool()

        input_result = InputEvalResult(
            input_text=input_text,
            model_output=model_output,
            validation_passed=passed,
            score=score,
            validation_reason=justification,
        )
        input_results.append(input_result)

        # reset both generator and judge -- might not be necessary since SimpleContext doesn't retain history
        generation_session.reset()
        judge_session.reset()

    test_result = TestEvalResult(test_eval=test_eval, input_results=input_results)
    return test_result


def create_judge_requirement(
    test_eval: TestBasedEval, input_text: str, model_output: str
):
    """Create judge requirement description"""

    if len(test_eval.targets) == 0:  # no reference
        target_text = "N/A"  # another way to handle this?
    elif len(test_eval.targets) == 1:
        target_text = test_eval.targets[0]
    else:  # enumerate the multiple targets
        target_text = "\n".join(
            [f"{i}. {target}" for i, target in enumerate(test_eval.targets, 1)]
        )

    judge_prompt = test_eval.judge_prompt.format(
        input=input_text,
        prediction=model_output,
        target=target_text,
        guidelines=test_eval.unit_test_instructions,
    )

    return judge_prompt


def parse_judge_output(judge_output: str):
    try:
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', judge_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            score = data.get("score")
            justification = data.get("justification")
            return score, justification
    except (json.JSONDecodeError, AttributeError):
        pass

    # if the above fails, search the text for the score
    score_match = re.search(r'score["\s:]+(\d+)', judge_output, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        return score, judge_output

    return None, judge_output


def save_results(results: List[TestEvalResult], output_path: str, output_format: str):
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
            "total_unit_tests": len(results),
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


def summary_stats(results: List[TestEvalResult]):
    total_inputs = sum(r.total_count for r in results)
    passed_inputs = sum(r.passed_count for r in results)
    overall_pass_rate = passed_inputs / total_inputs if total_inputs > 0 else 0.0

    console.print(f"Total number of inputs across tests: {total_inputs}")
    console.print(f"Number of inputs passed across tests: {passed_inputs}")
    console.print(f"Cumulative Pass Rate: {overall_pass_rate}")

    if len(results) > 1:
        console.print("Per-Test Breakdown:")
        for result in results:
            console.print(
                f"{result.test_eval.conversation_id}:\n\t{result.passed_count}/{result.total_count} ({result.pass_rate * 100:.1f}%)\n\n"
            )
