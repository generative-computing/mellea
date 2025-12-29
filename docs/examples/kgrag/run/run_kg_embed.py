#!/usr/bin/env python3
"""
Knowledge Graph Embedding Script (Refactored)

This script generates and stores embeddings for entities, relations, and schemas
in the knowledge graph using modern patterns.

Usage:
    python run_kg_embed_refactored.py
    python run_kg_embed_refactored.py --verbose
    python run_kg_embed_refactored.py --batch-size 10000
"""

import argparse
import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv
import openai

from kg.kg_embedder import KGEmbedder
from kg.kg_embed_models import EmbeddingConfig
from utils.logger import logger

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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def create_embedding_session(config: EmbeddingConfig) -> Any:
    """Create embedding session (OpenAI or local model).

    Args:
        config: Embedding configuration

    Returns:
        Embedding session object
    """
    if config.api_base:
        logger.info(f"Using OpenAI API at {config.api_base}")
        logger.info(f"Model: {config.model_name}")

        headers = {}
        if config.rits_api_key:
            headers['RITS_API_KEY'] = config.rits_api_key

        return openai.AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
            timeout=config.timeout,
            default_headers=headers if headers else None
        )
    else:
        logger.info("Using local SentenceTransformer model")
        logger.info(f"Model: {config.model_name}")

        import torch
        from sentence_transformers import SentenceTransformer

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        logger.info(f"Using device: {device}")

        return SentenceTransformer(
            config.model_name,
            device=device
        )


def create_config(args: argparse.Namespace) -> EmbeddingConfig:
    """Create embedding configuration from args and environment.

    Args:
        args: Parsed command-line arguments

    Returns:
        Embedding configuration
    """
    # Start with env-based config
    config = EmbeddingConfig(
        api_key=os.getenv("API_KEY", "dummy"),
        api_base=os.getenv("EMB_API_BASE"),
        model_name=os.getenv("EMB_MODEL_NAME", ""),
        timeout=int(os.getenv("EMB_TIME_OUT", "1800")),
        rits_api_key=os.getenv("RITS_API_KEY"),
    )

    # Override with CLI arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.storage_batch_size is not None:
        config.storage_batch_size = args.storage_batch_size

    if args.dimensions is not None:
        config.vector_dimensions = args.dimensions

    return config


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        # Create configuration
        config = create_config(args)
        logger.info("Configuration:")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Storage batch size: {config.storage_batch_size}")
        logger.info(f"  Vector dimensions: {config.vector_dimensions}")
        logger.info(f"  Concurrent batches: {config.concurrent_batches}")

        # Create embedding session
        emb_session = create_embedding_session(config)

        # Create embedder
        embedder = KGEmbedder(emb_session, config)

        # Run embedding pipeline
        stats = await embedder.embed_all()

        logger.info("=" * 60)
        logger.info("✅ Neo4j KG embedding completed!")
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
