#!/usr/bin/env python3
"""
Evaluation Script

This script evaluates QA results by comparing predictions against ground truth answers.
It can either re-evaluate existing results or evaluate results from a progress file.

Usage:
    python run_eval.py --reeval results/model_results.json
    python run_eval.py --result-path results/_results.json
    python run_eval.py --prefix exp1 --postfix test1 --verbose
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as async_tqdm

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend, TemplateFormatter
from mellea.stdlib.genslot import generative
from mellea.stdlib.requirement import Requirement

from utils.logger import QAProgressLogger, logger
from utils.utils import token_counter

# Load environment variables
load_dotenv()


# Pydantic models for type-safe outputs
class EvaluationResult(BaseModel):
    """Result of evaluating a single prediction."""
    score: int = Field(description="Score: 1 if correct, 0 if incorrect")
    explanation: str = Field(description="Brief explanation of the evaluation")


@dataclass
class EvaluationStats:
    """Statistics for evaluation operations."""
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    accuracy: float
    avg_score: float
    processing_time: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "incorrect_answers": self.incorrect_answers,
            "accuracy": self.accuracy,
            "score": self.avg_score,
            "eval_prompt_tokens": self.prompt_tokens,
            "eval_completion_tokens": self.completion_tokens,
            "eval_total_tokens": self.total_tokens
        }


# Define validation requirement
VALID_EVAL_SCORE = Requirement(
    description="Score must be 0 or 1",
    validation_fn=lambda o: o.score in [0, 1]
)


@generative
async def evaluate_single_prediction(
    query: str,
    ground_truth: str,
    prediction: str
) -> EvaluationResult:
    """Evaluate a single prediction against ground truth.

    You are an expert human evaluator. Judge if the prediction matches the ground truth answer.

    Instructions:
    1. Take it as granted that the Ground Truth is always correct.
    2. If the Prediction indicates uncertainty, score=0; otherwise, go to next step.
    3. If the Prediction exactly matches the Ground Truth, score=1.
    4. If the Prediction does not exactly match, go through the following steps:
       - If Ground Truth is a number, score=1 only if Prediction gives an almost exact match.
       - If Prediction is self-contradictory, score=0.
       - If Prediction is not answering the question, score=0.
       - If Prediction is a concise and correct summary of ground truth, score=1.
       - If ground truth contains a set of items, prediction must contain exactly same items for score=1.
       - Otherwise, score=0.

    Key Examples:
    - Question: "who is taller, a or b?"
      Ground Truth: "a"
      Prediction: "The answer is a. a is 1.75 m and b is 1.82 m. So b is taller."
      Score: 0 (self-contradictory)

    - Question: "who authored the taming of the shrew?"
      Ground Truth: "william shakespeare"
      Prediction: "w shakespeare"
      Score: 1 (abbreviation matches)

    - Question: "what is the state bird of california?"
      Ground Truth: "california quail"
      Prediction: "california valley quail"
      Score: 1 (same bird, different name)

    - Question: "how deep is the deepest lake of new york?"
      Ground Truth: "618 ft"
      Prediction: "the deepest lake in new york is seneca lake, with a depth of 618.23 feet."
      Score: 1 (number matches after rounding)

    - Question: "on which days did xxx distribute dividends in the last year?"
      Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
      Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
      Score: 0 (one item doesn't match)

    Now evaluate:
    Question: {query}
    Ground Truth: {ground_truth}
    Prediction: {prediction}

    Return your evaluation as:
    {{
        "score": 0 or 1,
        "explanation": "Brief explanation as short as possible"
    }}
    """
    pass


class MelleaEvaluator:
    """Mellea-native evaluator using @generative functions."""

    def __init__(self, session: MelleaSession, batch_size: int = 64):
        """Initialize evaluator.

        Args:
            session: Mellea session for evaluation
            batch_size: Batch size for processing
        """
        self.session = session
        self.batch_size = batch_size

    async def evaluate_batch(
        self,
        queries: List[str],
        ground_truths: List[str],
        predictions: List[str]
    ) -> List[EvaluationResult]:
        """Evaluate a batch of predictions.

        Args:
            queries: List of questions
            ground_truths: List of ground truth answers
            predictions: List of model predictions

        Returns:
            List of evaluation results
        """
        tasks = []
        for query, truth, pred in zip(queries, ground_truths, predictions):
            task = evaluate_single_prediction(
                query=query,
                ground_truth=truth,
                prediction=pred
            )
            tasks.append(task)

        # Process with progress bar
        results = []
        for task in async_tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating"
        ):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                # Add failed result
                results.append(EvaluationResult(
                    score=0,
                    explanation=f"Evaluation error: {str(e)}"
                ))

        return results

    async def evaluate_all(
        self,
        queries: List[str],
        ground_truths_list: List[List[str]],
        predictions: List[str]
    ) -> tuple[EvaluationStats, List[Dict[str, Any]]]:
        """Evaluate all predictions.

        Args:
            queries: List of questions
            ground_truths_list: List of ground truth lists (each can have multiple answers)
            predictions: List of model predictions

        Returns:
            Tuple of (statistics, history list)
        """
        start_time = datetime.now()

        # Flatten ground truths (take first one)
        ground_truths = [truths[0] for truths in ground_truths_list]

        # Evaluate all
        results = await self.evaluate_batch(queries, ground_truths, predictions)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Calculate statistics
        total = len(results)
        correct = sum(1 for r in results if r.score == 1)
        incorrect = total - correct
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_score = sum(r.score for r in results) / total if total > 0 else 0

        # Get token usage
        token_usage = token_counter.get_token_usage()

        stats = EvaluationStats(
            total_questions=total,
            correct_answers=correct,
            incorrect_answers=incorrect,
            accuracy=accuracy,
            avg_score=avg_score,
            processing_time=processing_time,
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0)
        )

        # Convert results to history format
        history = [
            {"score": r.score, "explanation": r.explanation}
            for r in results
        ]

        return stats, history


def evaluate_predictions(
    queries: List[str],
    ground_truths_list: List[List[str]],
    predictions: List[str],
    evaluation_model_name: str,
    batch_size: int = 64
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Backward-compatible wrapper for evaluate_predictions.

    This function provides the same interface as the old eval.py for compatibility
    with run_qa.py, but uses the Mellea-based evaluator internally.

    Args:
        queries: List of queries
        ground_truths_list: List of lists of ground truth answers
        predictions: List of predictions
        evaluation_model_name: Name of evaluation model (for logging)
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (results dict, history list)
    """
    import os

    # Get environment variables
    EVAL_API_KEY = os.getenv("EVAL_API_KEY", os.getenv("API_KEY", "dummy"))
    EVAL_API_BASE = os.getenv("EVAL_API_BASE", os.getenv("API_BASE", "http://localhost:8000/v1"))
    EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", os.getenv("MODEL_NAME", "gpt-4"))
    EVAL_TIME_OUT = int(os.getenv("EVAL_TIME_OUT", os.getenv("TIME_OUT", "1800")))

    MODEL_NAME = os.getenv("MODEL_NAME", "")
    EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "")

    # Create evaluation session
    eval_session = MelleaSession(
        backend=OpenAIBackend(
            model_id=EVAL_MODEL_NAME,
            base_url=EVAL_API_BASE,
            api_key=EVAL_API_KEY,
            timeout=EVAL_TIME_OUT,
        )
    )

    # Create evaluator
    evaluator = MelleaEvaluator(session=eval_session, batch_size=batch_size)

    # Run evaluation
    stats, history = asyncio.run(
        evaluator.evaluate_all(queries, ground_truths_list, predictions)
    )

    # Convert stats to dict format compatible with old eval.py
    n_correct = stats.correct_answers
    n_miss = 0  # Mellea evaluator doesn't track "I don't know" separately
    n = stats.total_questions

    # Handle "I don't know" cases
    for i, pred in enumerate(predictions):
        if "i don't know" in pred.lower():
            n_miss += 1
            if history[i]["score"] == 0:
                history[i]["explanation"] = "I don't know."

    # Adjust counts
    n_hallucination = n - n_correct - n_miss

    results = {
        "score": ((2 * n_correct + n_miss) / n - 1) * 100.0 if n > 0 else 0.0,
        "accuracy": stats.accuracy,
        "hallucination": (n_hallucination / n) * 100.0 if n > 0 else 0.0,
        "missing": (n_miss / n) * 100.0 if n > 0 else 0.0,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n_hallucination,
        "total": n,
        "llm": MODEL_NAME,
        "emb_llm": EMB_MODEL_NAME,
        "eval_llm": EVAL_MODEL_NAME
    }

    logger.info(results)
    return results, history


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate QA results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --reeval results/model_results.json
  %(prog)s --result-path results/_results.json
  %(prog)s --prefix exp1 --postfix test1 --verbose
        """
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--reeval",
        type=str,
        help="Path to .json file to re-evaluate"
    )
    input_group.add_argument(
        "--result-path",
        type=str,
        help="Path to results file to evaluate"
    )

    # Output configuration
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for result file name"
    )

    parser.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Postfix for result file name"
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="movie",
        help="Dataset name (default: movie)"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_results_from_progress(
    prefix: Optional[str],
    postfix: Optional[str],
    dataset: str
) -> tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Load results from progress file.

    Args:
        prefix: Optional prefix for file name
        postfix: Optional postfix for file name
        dataset: Dataset name

    Returns:
        Tuple of (results list, result path, empty stats dict)
    """
    prefix_str = f"_{prefix}" if prefix else ""
    postfix_str = f"_{postfix}" if postfix else ""

    progress_path = f"results/{prefix_str}_progress{postfix_str}.json"
    result_path = f"results/{prefix_str}_results{postfix_str}.json"

    logger.info(f"Loading progress from: {progress_path}")

    progress_logger = QAProgressLogger(progress_path=progress_path)

    if len(progress_logger.progress_data["stats"]) == 0:
        logger.error(f"No progress found in {progress_path}")
        sys.exit(1)

    results = [
        {
            "id": int(stat["id"]),
            "query": stat["query"],
            "query_time": stat["query_time"],
            "ans": stat["ans"],
            "prediction": stat["prediction"],
            "processing_time": stat["processing_time"]
        }
        for stat in progress_logger.progress_data["stats"]
    ]

    return results, result_path, {}


def load_results_from_file(reeval_path: str) -> tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Load results from existing results file for re-evaluation.

    Args:
        reeval_path: Path to existing results file

    Returns:
        Tuple of (results list, result path, existing stats dict)
    """
    logger.info(f"Loading results from: {reeval_path}")

    if not Path(reeval_path).exists():
        logger.error(f"Results file not found: {reeval_path}")
        sys.exit(1)

    with open(reeval_path, "r", encoding="utf-8") as f:
        temp_results = json.load(f)

    results = []
    other_stats = {}

    for result in temp_results:
        if "id" in result:
            results.append(result)
        else:
            # This is the stats entry
            other_stats = result.copy()
            # Remove eval_llm as it will be replaced
            other_stats.pop("eval_llm", None)

    return results, reeval_path, other_stats


def create_eval_session() -> MelleaSession:
    """Create Mellea session for evaluation.

    Returns:
        Mellea session
    """
    import os

    model_name = os.getenv("EVAL_MODEL_NAME", os.getenv("MODEL_NAME", ""))
    api_base = os.getenv("API_BASE", "http://localhost:7878/v1")
    api_key = os.getenv("API_KEY", "dummy")
    timeout = int(os.getenv("TIME_OUT", "1800"))
    rits_api_key = os.getenv("RITS_API_KEY")

    logger.info(f"Creating evaluation session with model: {model_name}")

    headers = {}
    if rits_api_key:
        headers['RITS_API_KEY'] = rits_api_key

    return MelleaSession(
        backend=OpenAIBackend(
            model_id=model_name,
            formatter=TemplateFormatter(model_id=model_name),
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            default_headers=headers if headers else None
        )
    )


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    try:
        logger.info("=" * 60)
        logger.info("Evaluation Configuration")
        logger.info("=" * 60)

        # Load results based on input method
        if args.reeval:
            results, result_path, existing_stats = load_results_from_file(args.reeval)
        else:
            results, result_path, existing_stats = load_results_from_progress(
                args.prefix,
                args.postfix,
                args.dataset
            )

        logger.info(f"Evaluation method: {args.eval_method}")
        logger.info(f"Batch size: {args.eval_batch_size}")
        logger.info(f"Number of results: {len(results)}")
        logger.info(f"Result path: {result_path}")
        logger.info("=" * 60)

        # Sort results by ID
        results = sorted(results, key=lambda x: x["id"])

        # Prepare evaluation data
        queries = [item["query"] for item in results]
        ground_truths_list = [[str(item["ans"])] for item in results]
        predictions = [str(item["prediction"]) for item in results]

        # Create evaluation session
        session = create_eval_session()

        # Create evaluator
        evaluator = MelleaEvaluator(session, batch_size=args.eval_batch_size)

        # Reset token counter for evaluation
        token_counter.reset_token_usage()

        # Run evaluation
        logger.info("Running evaluation with Mellea-native patterns...")
        stats, history = await evaluator.evaluate_all(
            queries,
            ground_truths_list,
            predictions
        )

        logger.info(f"Evaluation complete in {stats.processing_time:.2f}s")

        # Merge stats with existing
        final_stats = {**existing_stats, **stats.to_dict()}

        # Add scores to results
        for idx in range(len(results)):
            results[idx]['score'] = history[idx]['score']
            results[idx]['explanation'] = history[idx]['explanation']

        # Save final results
        Path(result_path).parent.mkdir(exist_ok=True)
        final_results = [final_stats] + results

        logger.info(f"Saving results to: {result_path}")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Evaluation completed!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {result_path}")
        logger.info(f"Total questions: {stats.total_questions}")
        logger.info(f"Correct answers: {stats.correct_answers}")
        logger.info(f"Accuracy: {stats.accuracy:.2f}%")
        logger.info(f"Avg score: {stats.avg_score:.3f}")
        logger.info(f"Evaluation tokens: {stats.total_tokens:,}")
        logger.info(f"Processing time: {stats.processing_time:.2f}s")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
