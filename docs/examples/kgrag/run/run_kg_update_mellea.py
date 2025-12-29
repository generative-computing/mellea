#!/usr/bin/env python3
"""
Knowledge Graph Update Script (Mellea-Native Implementation)

This script demonstrates KG updates using Mellea's native patterns:
- @generative decorator for extraction functions
- Requirements for output validation
- RejectionSamplingStrategy for robustness
- Component-based architecture

Usage:
    python run_kg_update_mellea.py --dataset path/to/dataset.jsonl.bz2
    python run_kg_update_mellea.py --num-workers 64 --verbose
    python run_kg_update_mellea.py --domain movie --progress-path results/progress.json
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

from kg.kg_updater_component import KGUpdaterComponent
from kg.kg_driver import KG_Driver
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
        description="Update knowledge graph using Mellea-native patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset data/corpus.jsonl.bz2
  %(prog)s --num-workers 64 --queue-size 64
  %(prog)s --domain movie --progress-path results/progress_mellea.json
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
        default="results/update_movie_kg_progress_mellea.json",
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
    """Create session configuration from environment.

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
    """Create dataset configuration from args.

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
        progress_path=args.progress_path,
    )


def create_mellea_session(session_config: SessionConfig) -> MelleaSession:
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


def create_embedding_session(session_config: SessionConfig) -> Any:
    """Create embedding session.

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


async def process_document(
    kg_updater: KGUpdaterComponent,
    doc_id: str = "",
    context: str = "",
    reference: str = "",
    logger: KGProgressLogger = None,
    **kwargs
) -> None:
    """Process a single document using Mellea-native KG updater.

    Args:
        kg_updater: KGUpdaterComponent instance
        doc_id: Document ID
        context: Document text
        reference: Reference/source
        logger: Progress logger
        **kwargs: Additional arguments
    """
    from datetime import datetime
    import time

    start_time = time.perf_counter()

    try:
        stats = await kg_updater.update_kg_from_document(
            doc_id=doc_id,
            context=context,
            reference=reference,
            created_at=datetime.now()
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        logger.add_stat({
            "doc_id": doc_id,
            "entities_extracted": stats.get("entities_extracted", 0),
            "entities_new": stats.get("entities_new", 0),
            "relations_extracted": stats.get("relations_extracted", 0),
            "relations_new": stats.get("relations_new", 0),
            "processing_time": round(elapsed_time, 2),
        })

        print(f"Processed documents: {len(logger.processed_questions)}")
        logger.update_progress({"last_doc_time": round(elapsed_time, 2)})

    except Exception as e:
        logger.error(f"Failed to process document {doc_id}: {e}")
        raise


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
        logger.info("Mellea-Native KG Update Configuration:")
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_config.dataset_path}")
        logger.info(f"Domain: {dataset_config.domain}")
        logger.info(f"Workers: {updater_config.num_workers}")
        logger.info(f"Queue size: {updater_config.queue_size}")
        logger.info(f"Progress: {dataset_config.progress_path}")
        logger.info("Using Mellea-native implementation with:")
        logger.info("  ✓ @generative decorator for extractions")
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
        emb_session = create_embedding_session(session_config)

        # Create KG driver
        kg_driver = KG_Driver(
            database=None,  # Uses default from env
            emb_session=emb_session
        )

        # Create progress logger
        kg_logger = KGProgressLogger(progress_path=dataset_config.progress_path)
        logger.info(f"Processed documents at start: {len(kg_logger.processed_questions)}")

        # Create Mellea-native KG Updater component
        kg_updater = KGUpdaterComponent(
            session=session,
            emb_session=emb_session,
            kg_driver=kg_driver,
            domain=dataset_config.domain,
            config={
                "align_entity": True,
                "merge_entity": True,
                "align_relation": True,
                "merge_relation": True,
                "extraction_loop_budget": 3,
            },
            logger=kg_logger
        )

        # Create dataset loader
        loader = MovieDatasetLoader(
            dataset_config.dataset_path,
            updater_config.model_dump(),
            "update",
            kg_logger,
            processor=functools.partial(
                process_document,
                kg_updater=kg_updater,
                logger=kg_logger
            )
        )

        # Run KG update
        logger.info("Starting KG update with Mellea-native implementation...")
        await loader.run()

        # Get token usage
        token_usage = token_counter.get_token_usage()
        logger.info(f"Update complete. Token usage: {token_usage}")

        # Compute statistics
        stats = kg_logger.progress_data.get("stats", [])
        total_entities = sum(s.get("entities_new", 0) for s in stats)
        total_relations = sum(s.get("relations_new", 0) for s in stats)

        logger.info("=" * 60)
        logger.info("✅ Mellea-native KG update completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Processed documents: {len(stats)}")
        logger.info(f"Total new entities: {total_entities}")
        logger.info(f"Total new relations: {total_relations}")
        logger.info(f"Total tokens: {token_usage.get('total_tokens', 0)}")
        logger.info(f"Progress saved to: {dataset_config.progress_path}")

        # Close KG driver
        await kg_driver.close()

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
