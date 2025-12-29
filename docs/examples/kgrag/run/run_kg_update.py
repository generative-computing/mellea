#!/usr/bin/env python3
"""
Knowledge Graph Update Script (Refactored)

This script updates the knowledge graph by processing documents and extracting
entities and relations using modern patterns.

Usage:
    python run_kg_update_refactored.py --dataset path/to/dataset.jsonl.bz2
    python run_kg_update_refactored.py --num-workers 128 --verbose
    python run_kg_update_refactored.py --domain movie --progress-path results/progress.json
"""

import argparse
import asyncio
import functools
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import openai
import torch

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend, TemplateFormatter

from kg.kg_updater import KG_Updater
from kg.kg_updater_models import UpdaterConfig, SessionConfig, DatasetConfig
from dataset.movie_dataset import MovieDatasetLoader
from utils.logger import KGProgressLogger
from utils.utils import token_counter
from utils.logger import logger

# Load environment variables
load_dotenv()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update knowledge graph from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset data/corpus.jsonl.bz2
  %(prog)s --num-workers 128 --queue-size 128
  %(prog)s --domain movie --progress-path results/progress.json
  %(prog)s --verbose
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
        default=64,
        help="Number of concurrent workers (default: 64)"
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help="Queue size for data loading (default: 64)"
    )

    # Progress tracking
    parser.add_argument(
        "--progress-path",
        type=str,
        default="results/update_movie_kg_progress.json",
        help="Progress log file path"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def create_session_config(args: argparse.Namespace) -> SessionConfig:
    """Create session configuration from environment and args.

    Args:
        args: Parsed command-line arguments

    Returns:
        Session configuration
    """
    return SessionConfig(
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


def create_updater_config(args: argparse.Namespace) -> UpdaterConfig:
    """Create updater configuration from args.

    Args:
        args: Parsed command-line arguments

    Returns:
        Updater configuration
    """
    return UpdaterConfig(
        num_workers=args.num_workers,
        queue_size=args.queue_size,
    )


def create_dataset_config(args: argparse.Namespace) -> DatasetConfig:
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
        dataset_path = os.path.join(base_dir, "crag_movie_dev.jsonl.bz2")

    return DatasetConfig(
        dataset_path=dataset_path,
        domain=args.domain,
        progress_path=args.progress_path
    )


def create_mellea_session(session_config: SessionConfig) -> MelleaSession:
    """Create Mellea session for LLM.

    Args:
        session_config: Session configuration

    Returns:
        Mellea session
    """
    logger.info(f"Creating Mellea session with model: {session_config.model_name}")
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


def create_embedding_session(session_config: SessionConfig) -> Any:
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


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        # Create configurations
        session_config = create_session_config(args)
        updater_config = create_updater_config(args)
        dataset_config = create_dataset_config(args)

        logger.info("=" * 60)
        logger.info("KG Update Configuration:")
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_config.dataset_path}")
        logger.info(f"Domain: {dataset_config.domain}")
        logger.info(f"Workers: {updater_config.num_workers}")
        logger.info(f"Queue size: {updater_config.queue_size}")
        logger.info(f"Progress path: {dataset_config.progress_path}")
        logger.info("=" * 60)

        # Verify dataset exists
        if not Path(dataset_config.dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_config.dataset_path}")
            return 1

        # Create sessions
        session = create_mellea_session(session_config)
        emb_session = create_embedding_session(session_config)

        # Create progress logger
        kg_logger = KGProgressLogger(progress_path=dataset_config.progress_path)

        # Create updater
        updater = KG_Updater(
            session=session,
            emb_session=emb_session,
            config=updater_config.model_dump(),  # Convert to dict for backward compatibility
            logger=kg_logger
        )

        # Create dataset loader
        loader = MovieDatasetLoader(
            dataset_config.dataset_path,
            updater_config.model_dump(),
            "doc",
            kg_logger,
            processor=functools.partial(
                updater.process_doc,
                domain=dataset_config.domain
            )
        )

        logger.info(f"Processed docs at start: {kg_logger.processed_docs}")

        # Run the update process
        await loader.run()

        logger.info("=" * 60)
        logger.info("✅ KG update completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total processed docs: {kg_logger.processed_docs}")
        logger.info(f"Token usage: {token_counter.get_token_usage()}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  KG update interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ KG update failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
