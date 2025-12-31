#!/usr/bin/env python3
"""
Knowledge Graph QA Script (Mellea-Native Implementation)

This script demonstrates KG-RAG using Mellea's native patterns:
- @generative decorator for LLM functions
- Requirements for output validation
- RejectionSamplingStrategy for robustness
- Component-based architecture

Usage:
    python run_qa_mellea.py --dataset data/crag_movie_dev.jsonl
    python run_qa_mellea.py --num-workers 256 --verbose
    python run_qa_mellea.py --prefix exp1 --postfix test1
    python run_qa_mellea.py --config route=5 width=30 depth=3
"""

import argparse
import asyncio
import functools
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import openai
import torch

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend, TemplateFormatter

# Import Mellea-native KG-RAG component
from kg.kg_rag import KGRagComponent, Query
from kg.kg_qa_models import QAConfig, QASessionConfig, QADatasetConfig
from dataset.movie_dataset import MovieDatasetLoader
from utils.logger import BaseProgressLogger, DefaultProgressLogger, QAProgressLogger
from utils.utils import token_counter
from eval import evaluate_predictions
from utils.logger import logger

# Load environment variables
load_dotenv()


def parse_key_value(arg: str) -> tuple:
    """Parse key=value string into a (key, value) pair.

    Args:
        arg: String in format "key=value"

    Returns:
        Tuple of (key, value) with value converted to int/float if possible

    Raises:
        argparse.ArgumentTypeError: If argument format is invalid
    """
    if '=' not in arg:
        raise argparse.ArgumentTypeError("Arguments must be in key=value format")

    key, value = arg.split('=', 1)

    # Try to convert to numeric types
    try:
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  # Keep as string

    return key, value


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QA evaluation using Mellea-native KG-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset data/crag_movie_dev.jsonl
  %(prog)s --num-workers 256 --queue-size 256
  %(prog)s --prefix mellea --postfix test1
  %(prog)s --config route=5 width=30 depth=3
  %(prog)s --verbose --keep
        """
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file (overrides env KG_BASE_DIRECTORY)"
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="movie",
        help="Knowledge domain (default: movie)"
    )

    # Worker configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=128,
        help="Number of concurrent workers (default: 128)"
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=128,
        help="Queue size for data loading (default: 128)"
    )

    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="Dataset split index (default: 0)"
    )

    # Output configuration
    parser.add_argument(
        "--prefix",
        type=str,
        default="mellea",
        help="Prefix for output files (default: mellea)"
    )

    parser.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Postfix for output files"
    )

    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep progress file after completion"
    )

    # Model configuration
    parser.add_argument(
        "--config",
        nargs="*",
        type=parse_key_value,
        help="Override model config as key=value pairs (e.g., route=5 width=30)"
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

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def create_session_config(args: argparse.Namespace) -> QASessionConfig:
    """Create session configuration from environment and args.

    Args:
        args: Parsed command-line arguments

    Returns:
        Session configuration
    """
    return QASessionConfig(
        # Main LLM
        api_base=os.getenv("API_BASE", "http://localhost:7878/v1"),
        api_key=os.getenv("API_KEY", "dummy"),
        model_name=os.getenv("MODEL_NAME", ""),
        timeout=int(os.getenv("TIME_OUT", "1800")),
        rits_api_key=os.getenv("RITS_API_KEY"),

        # Evaluation LLM
        eval_api_base=os.getenv("EVAL_API_BASE"),
        eval_api_key=os.getenv("EVAL_API_KEY", "dummy"),
        eval_model_name=os.getenv("EVAL_MODEL_NAME"),
        eval_timeout=int(os.getenv("EVAL_TIME_OUT", "1800")) if os.getenv("EVAL_TIME_OUT") else None,

        # Embedding
        emb_api_base=os.getenv("EMB_API_BASE"),
        emb_api_key=os.getenv("EMB_API_KEY", "dummy"),
        emb_model_name=os.getenv("EMB_MODEL_NAME"),
        emb_timeout=int(os.getenv("EMB_TIME_OUT", "1800")) if os.getenv("EMB_TIME_OUT") else None,
    )


def create_qa_config(args: argparse.Namespace) -> QAConfig:
    """Create QA configuration from args.

    Args:
        args: Parsed command-line arguments

    Returns:
        QA configuration
    """
    return QAConfig(
        num_workers=args.num_workers,
        queue_size=args.queue_size,
        split=args.split,
        eval_batch_size=args.eval_batch_size,
        eval_method=args.eval_method,
    )


def create_dataset_config(args: argparse.Namespace) -> QADatasetConfig:
    """Create dataset configuration from args and environment.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dataset configuration
    """
    # Determine dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        base_dir = os.getenv(
            "KG_BASE_DIRECTORY",
            os.path.join(os.path.dirname(__file__), "..", "dataset")
        )
        # Try compressed version first, then uncompressed
        compressed_path = os.path.join(base_dir, "crag_movie_dev.jsonl.bz2")
        uncompressed_path = os.path.join(base_dir, "crag_movie_dev.jsonl")
        if os.path.exists(compressed_path):
            dataset_path = compressed_path
        else:
            dataset_path = uncompressed_path

    # Create output paths with optional prefix/postfix
    prefix_str = f"_{args.prefix}" if args.prefix else ""
    postfix_str = f"_{args.postfix}" if args.postfix else ""

    progress_path = f"results/{prefix_str}_progress{postfix_str}.json"
    result_path = f"results/{prefix_str}_results{postfix_str}.json"

    return QADatasetConfig(
        dataset_path=dataset_path,
        domain=args.domain,
        result_path=result_path,
        progress_path=progress_path,
        prefix=args.prefix,
        postfix=args.postfix,
        keep_progress=args.keep,
    )


def create_model_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Create model configuration from CLI arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary of model configuration or None
    """
    if args.config:
        return dict(args.config)
    return None


def create_mellea_session(session_config: QASessionConfig) -> MelleaSession:
    """Create Mellea session for LLM.

    Args:
        session_config: Session configuration

    Returns:
        Mellea session
    """
    logger.info(f"Creating main session with model: {session_config.model_name}")
    logger.info(f"API base: {session_config.api_base}")

    headers = {}
    if session_config.rits_api_key:
        headers['RITS_API_KEY'] = session_config.rits_api_key

    return MelleaSession(
        backend=OpenAIBackend(
            model_id=session_config.model_name,
            formatter=TemplateFormatter(model_id=session_config.model_name),
            base_url=session_config.api_base,
            api_key=session_config.api_key,
            timeout=session_config.timeout,
            default_headers=headers if headers else None
        )
    )


def create_eval_session(session_config: QASessionConfig) -> MelleaSession:
    """Create evaluation session for LLM.

    Args:
        session_config: Session configuration

    Returns:
        Evaluation Mellea session
    """
    # Use eval-specific config if provided, otherwise fall back to main
    eval_api_base = session_config.eval_api_base or session_config.api_base
    eval_api_key = session_config.eval_api_key or session_config.api_key
    eval_model_name = session_config.eval_model_name or session_config.model_name
    eval_timeout = session_config.eval_timeout or session_config.timeout

    logger.info(f"Creating eval session with model: {eval_model_name}")

    headers = {}
    if session_config.rits_api_key:
        headers['RITS_API_KEY'] = session_config.rits_api_key

    return MelleaSession(
        backend=OpenAIBackend(
            model_id=eval_model_name,
            formatter=TemplateFormatter(model_id=eval_model_name),
            base_url=eval_api_base,
            api_key=eval_api_key,
            timeout=eval_timeout,
            default_headers=headers if headers else None
        )
    )


def create_embedding_session(session_config: QASessionConfig) -> Any:
    """Create embedding session (OpenAI or local model).

    Args:
        session_config: Session configuration

    Returns:
        Embedding session object
    """
    if session_config.emb_api_base:
        logger.info(f"Using OpenAI embedding API at {session_config.emb_api_base}")
        logger.info(f"Model: {session_config.emb_model_name}")

        headers = {}
        if session_config.rits_api_key:
            headers['RITS_API_KEY'] = session_config.rits_api_key

        return openai.AsyncOpenAI(
            base_url=session_config.emb_api_base,
            api_key=session_config.emb_api_key or session_config.api_key,
            timeout=session_config.emb_timeout or session_config.timeout,
            default_headers=headers if headers else None
        )
    else:
        logger.info("Using local SentenceTransformer for embeddings")
        logger.info(f"Model: {session_config.emb_model_name}")

        from sentence_transformers import SentenceTransformer

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        logger.info(f"Using device: {device}")

        return SentenceTransformer(
            session_config.emb_model_name,
            device=device
        )


def snapshot_token_usage() -> Dict[str, int]:
    """Snapshot current token usage.

    Returns:
        Dictionary of token counts
    """
    return deepcopy(token_counter.get_token_usage()) if token_counter else {}


def compute_token_usage_delta(start_usage: Dict[str, int]) -> Dict[str, int]:
    """Compute delta in token usage since snapshot.

    Args:
        start_usage: Starting token usage snapshot

    Returns:
        Dictionary of token usage deltas
    """
    if not token_counter:
        return {}

    end_usage = token_counter.get_token_usage()
    keys = set(start_usage.keys()) | set(end_usage.keys())
    return {key: end_usage.get(key, 0) - start_usage.get(key, 0) for key in keys}


async def generate_prediction(
    kg_rag: KGRagComponent,
    id: str = "",
    query: str = "",
    query_time: datetime = None,
    ans: str = "",
    logger: BaseProgressLogger = DefaultProgressLogger(),
    **kwargs
) -> None:
    """Generate a prediction for a single question using Mellea-native KG-RAG.

    Args:
        kg_rag: KGRagComponent instance
        id: Question ID
        query: Question text
        query_time: Query timestamp
        ans: Ground truth answer
        logger: Progress logger
        **kwargs: Additional arguments
    """
    start_time = time.perf_counter()
    token_usage_start = snapshot_token_usage()

    # Generate answer using Mellea-native component
    prediction = await kg_rag.execute(query=query, query_time=query_time)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    token_usage_delta = compute_token_usage_delta(token_usage_start)

    logger.add_stat({
        "id": id,
        "query": query,
        "query_time": query_time,
        "ans": ans,
        "prediction": prediction,
        "processing_time": round(elapsed_time, 2),
        "token_usage": token_usage_delta
    })

    print(f"Processed questions: {len(logger.processed_questions)}")
    logger.update_progress({"last_question_total": round(elapsed_time, 2)})


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        # Create configurations
        session_config = create_session_config(args)
        qa_config = create_qa_config(args)
        dataset_config = create_dataset_config(args)
        model_config = create_model_config(args)

        logger.info("=" * 60)
        logger.info("Mellea-Native KG QA Configuration:")
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_config.dataset_path}")
        logger.info(f"Domain: {dataset_config.domain}")
        logger.info(f"Workers: {qa_config.num_workers}")
        logger.info(f"Queue size: {qa_config.queue_size}")
        logger.info(f"Split: {qa_config.split}")
        logger.info(f"Results: {dataset_config.result_path}")
        logger.info(f"Progress: {dataset_config.progress_path}")
        if model_config:
            logger.info(f"Model config: {model_config}")
        logger.info("Using Mellea-native implementation with:")
        logger.info("  ✓ @generative decorator")
        logger.info("  ✓ Requirements validation")
        logger.info("  ✓ RejectionSamplingStrategy")
        logger.info("  ✓ Component architecture")
        logger.info("=" * 60)

        # Verify dataset exists
        if not Path(dataset_config.dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_config.dataset_path}")
            return 1

        # Ensure results directory exists
        Path("results").mkdir(exist_ok=True)

        # Create sessions
        session = create_mellea_session(session_config)
        eval_session = create_eval_session(session_config)
        emb_session = create_embedding_session(session_config)

        # Create progress logger
        qa_logger = QAProgressLogger(progress_path=dataset_config.progress_path)
        logger.info(f"Processed questions at start: {len(qa_logger.processed_questions)}")

        # Create Mellea-native KG-RAG component
        kg_rag = KGRagComponent(
            session=session,
            eval_session=eval_session,
            emb_session=emb_session,
            domain=dataset_config.domain,
            config=model_config,
            logger=qa_logger
        )

        # Create dataset loader
        loader = MovieDatasetLoader(
            dataset_config.dataset_path,
            qa_config.model_dump(),
            "qa",
            qa_logger,
            processor=functools.partial(
                generate_prediction,
                kg_rag=kg_rag,
                logger=qa_logger
            )
        )

        # Run QA generation
        logger.info("Starting QA generation with Mellea-native KG-RAG...")
        await loader.run()

        # Get inference token usage
        inf_token_usage = token_counter.get_token_usage()
        logger.info(f"Inference complete. Token usage: {inf_token_usage}")

        # Prepare results
        token_counter.reset_token_usage()
        results = [
            {
                "id": int(stat["id"]),
                "query": stat["query"],
                "query_time": stat["query_time"],
                "ans": stat["ans"],
                "prediction": stat["prediction"],
                "processing_time": stat["processing_time"],
                "token_usage": stat.get("token_usage", {})
            }
            for stat in qa_logger.progress_data["stats"]
        ]
        results = sorted(results, key=lambda x: x["id"])

        # Save intermediate results
        with open(dataset_config.result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        logger.info(f"Intermediate results saved to {dataset_config.result_path}")

        # Run evaluation
        logger.info("Running evaluation...")
        queries = [item["query"] for item in results]
        ground_truths_list = [[str(item["ans"])] for item in results]
        predictions = [str(item["prediction"]) for item in results]

        stats, history = evaluate_predictions(
            queries,
            ground_truths_list,
            predictions,
            qa_config.eval_method,
            batch_size=qa_config.eval_batch_size
        )

        eval_token_usage = token_counter.get_token_usage()
        logger.info(f"Evaluation complete. Token usage: {eval_token_usage}")

        # Add token usage stats
        stats.update({
            "inf_prompt_tokens": inf_token_usage.get("prompt_tokens"),
            "inf_completion_tokens": inf_token_usage.get("completion_tokens"),
            "inf_total_tokens": inf_token_usage.get("total_tokens"),
            "eval_prompt_tokens": eval_token_usage.get("prompt_tokens"),
            "eval_completion_tokens": eval_token_usage.get("completion_tokens"),
            "eval_total_tokens": eval_token_usage.get("total_tokens"),
            "implementation": "mellea_native"
        })

        # Add scores to results
        for idx in range(len(results)):
            results[idx]['score'] = history[idx]['score']
            results[idx]['explanation'] = history[idx]['explanation']

        # Insert stats at the beginning
        results.insert(0, stats)

        # Save final results
        with open(dataset_config.result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        logger.info("=" * 60)
        logger.info("✅ Mellea-native QA evaluation completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {dataset_config.result_path}")
        logger.info(f"Total questions: {len(results) - 1}")  # -1 for stats entry
        logger.info(f"Accuracy: {stats.get('accuracy', 'N/A')}")
        logger.info(f"Inference tokens: {inf_token_usage.get('total_tokens', 0)}")
        logger.info(f"Evaluation tokens: {eval_token_usage.get('total_tokens', 0)}")

        # Cleanup progress file if requested
        if not dataset_config.keep_progress:
            Path(dataset_config.progress_path).unlink(missing_ok=True)
            logger.info(f"Progress file removed: {dataset_config.progress_path}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  QA evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ QA evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
