#!/usr/bin/env python3
"""
Knowledge Graph Embedding Script (Mellea-Native Implementation)
This script generates and stores embeddings for entities, relations, and schemas
in the knowledge graph using modern patterns.

Usage:
    python run_kg_embed.py
    python run_kg_embed.py --verbose
    python run_kg_embed.py --batch-size 10000
"""

import argparse
import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv

from kg.kg_embedder import MelleaKGEmbedder, test_embedding_session
from kg.kg_embed_models import EmbeddingConfig

from utils.logger import logger
from utils.utils_mellea import create_embedding_session

# Load environment variables
load_dotenv()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and store KG embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use default configuration
  %(prog)s --batch-size 10000        # Custom batch size
  %(prog)s --verbose                 # Enable verbose logging
  %(prog)s --dimensions 1024         # Custom vector dimensions
        """
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding generation"
    )

    parser.add_argument(
        "--storage-batch-size",
        type=int,
        default=None,
        help="Batch size for storing embeddings"
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Vector embedding dimensions"
    )

    parser.add_argument(
        "--concurrent-batches",
        type=int,
        default=None,
        help="Number of concurrent batches for embedding"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()




def create_config(args: argparse.Namespace) -> EmbeddingConfig:
    """Create embedding configuration from args and environment.

    Args:
        args: Parsed command-line arguments

    Returns:
        Embedding configuration with Pydantic validation
    """
    # Start with env-based config
    config = EmbeddingConfig(
        api_key=os.getenv("API_KEY", "dummy"),
        api_base=os.getenv("EMB_API_BASE"),
        model_name=os.getenv("EMB_MODEL_NAME", ""),
        timeout=int(os.getenv("EMB_TIME_OUT", "1800")),
        rits_api_key=os.getenv("RITS_API_KEY"),
        vector_dimensions=int(os.getenv("VECTOR_DIMENSIONS", "768")),
        batch_size=int(os.getenv("EMB_BATCH_SIZE", "8192")),
        concurrent_batches=int(os.getenv("EMB_CONCURRENT_BATCHES", "64")),
        storage_batch_size=int(os.getenv("EMB_STORAGE_BATCH_SIZE", "50000")),
    )

    # Override with CLI arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.storage_batch_size is not None:
        config.storage_batch_size = args.storage_batch_size

    if args.dimensions is not None:
        config.vector_dimensions = args.dimensions

    if args.concurrent_batches is not None:
        config.concurrent_batches = args.concurrent_batches

    return config


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    try:
        # Create configuration
        config = create_config(args)
        logger.info("Configuration:")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Storage batch size: {config.storage_batch_size}")
        logger.info(f"  Vector dimensions: {config.vector_dimensions}")
        logger.info(f"  Concurrent batches: {config.concurrent_batches}")

        # Create embedding session
        emb_session = create_embedding_session(
            api_base=config.api_base,
            api_key=config.api_key,
            model_name=config.model_name,
            timeout=config.timeout,
            rits_api_key=config.rits_api_key
        )

        # Test embedding session
        if not await test_embedding_session(emb_session, config):
            logger.error("Embedding session test failed. Please check your configuration.")
            return 1

        logger.info("")

        # Create Mellea-native embedder
        embedder = MelleaKGEmbedder(emb_session, config)

        # Run embedding pipeline
        logger.info("Starting embedding pipeline...")
        stats = await embedder.embed_all()

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Mellea-native KG embedding completed!")
        logger.info("=" * 60)
        logger.info(f"Entities embedded: {stats.entities_embedded}")
        logger.info(f"Relations embedded: {stats.relations_embedded}")
        logger.info(f"Schemas embedded: {stats.schemas_embedded}")
        logger.info(f"Total embeddings: {stats.total_embeddings}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Embedding interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
