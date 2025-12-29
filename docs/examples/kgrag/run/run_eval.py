#!/usr/bin/env python3
"""
Evaluation Script (Refactored)

This script evaluates QA results by comparing predictions against ground truth answers.
It can either re-evaluate existing results or evaluate results from a progress file.

Usage:
    python run_eval.py --reeval results/model_results.json
    python run_eval.py --result-path results/_results.json
    python run_eval.py --prefix exp1 --postfix test1
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from eval import evaluate_predictions
from utils.logger import QAProgressLogger, logger
from utils.utils import token_counter

# Load environment variables
load_dotenv()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate QA results against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-evaluate existing results file
  %(prog)s --reeval results/model_results.json

  # Evaluate from result path
  %(prog)s --result-path results/_results.json

  # Evaluate with custom prefix/postfix
  %(prog)s --prefix exp1 --postfix test1

  # Specify evaluation batch size
  %(prog)s --reeval results/model_results.json --eval-batch-size 128
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
        help="Prefix for result file name (used if --result-path not specified)"
    )

    parser.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Postfix for result file name (used if --result-path not specified)"
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)"
    )

    parser.add_argument(
        "--eval-method",
        type=str,
        default="llama",
        help="Evaluation method (default: llama)"
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

    Raises:
        SystemExit: If progress file not found or empty
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

    Raises:
        FileNotFoundError: If results file not found
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


def prepare_evaluation_data(
    results: List[Dict[str, Any]]
) -> tuple[List[str], List[List[str]], List[str]]:
    """Prepare data for evaluation.

    Args:
        results: List of result dictionaries

    Returns:
        Tuple of (queries, ground_truths_list, predictions)
    """
    queries = [item["query"] for item in results]
    ground_truths_list = [[str(item["ans"])] for item in results]
    predictions = [str(item["prediction"]) for item in results]

    return queries, ground_truths_list, predictions


def merge_stats(
    eval_stats: Dict[str, Any],
    existing_stats: Dict[str, Any],
    eval_token_usage: Dict[str, int]
) -> Dict[str, Any]:
    """Merge evaluation stats with existing stats and token usage.

    Args:
        eval_stats: Stats from evaluation
        existing_stats: Existing stats from previous run
        eval_token_usage: Token usage from evaluation

    Returns:
        Merged stats dictionary
    """
    # Start with existing stats
    merged = existing_stats.copy()

    # Update with new eval stats
    merged.update(eval_stats)

    # Add evaluation token usage
    merged.update({
        "eval_prompt_tokens": eval_token_usage.get("prompt_tokens"),
        "eval_completion_tokens": eval_token_usage.get("completion_tokens"),
        "eval_total_tokens": eval_token_usage.get("total_tokens")
    })

    return merged


def add_scores_to_results(
    results: List[Dict[str, Any]],
    history: List[Dict[str, Any]]
) -> None:
    """Add evaluation scores and explanations to results.

    Args:
        results: List of result dictionaries (modified in place)
        history: List of evaluation history entries
    """
    for idx in range(len(results)):
        results[idx]['score'] = history[idx]['score']
        results[idx]['explanation'] = history[idx]['explanation']


def save_results(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    result_path: str
) -> None:
    """Save results with stats to file.

    Args:
        results: List of result dictionaries
        stats: Statistics dictionary
        result_path: Path to save results
    """
    # Insert stats at the beginning
    final_results = [stats] + results

    # Save to JSON file
    logger.info(f"Saving results to: {result_path}")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logger.setLevel("DEBUG")

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

        # Save intermediate results (without scores)
        Path(result_path).parent.mkdir(exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Prepare evaluation data
        queries, ground_truths_list, predictions = prepare_evaluation_data(results)

        # Reset token counter for evaluation
        token_counter.reset_token_usage()

        # Run evaluation
        logger.info("Running evaluation...")
        stats, history = evaluate_predictions(
            queries,
            ground_truths_list,
            predictions,
            args.eval_method,
            batch_size=args.eval_batch_size
        )

        logger.info(f"Evaluation stats: {stats}")

        # Get evaluation token usage
        eval_token_usage = token_counter.get_token_usage()
        logger.info(f"Evaluation token usage: {eval_token_usage}")

        # Merge stats
        final_stats = merge_stats(stats, existing_stats, eval_token_usage)

        # Add scores to results
        add_scores_to_results(results, history)

        # Save final results
        save_results(results, final_stats, result_path)

        logger.info("=" * 60)
        logger.info("✅ Evaluation completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {result_path}")
        logger.info(f"Total questions: {len(results)}")
        logger.info(f"Accuracy: {final_stats.get('accuracy', 'N/A')}%")
        logger.info(f"Score: {final_stats.get('score', 'N/A')}")
        logger.info(f"Evaluation tokens: {eval_token_usage.get('total_tokens', 0)}")

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
