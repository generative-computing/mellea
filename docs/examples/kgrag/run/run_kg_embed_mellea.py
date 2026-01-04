#!/usr/bin/env python3
"""
Knowledge Graph Embedding Script (Mellea-Native Implementation)

This script demonstrates KG embedding using Mellea's native utilities:
- Mellea-native embedding utilities (kg_utils_mellea.py)
- Component-based architecture
- Better error handling and logging
- Type-safe configuration with Pydantic

Usage:
    python run_kg_embed_mellea.py
    python run_kg_embed_mellea.py --verbose
    python run_kg_embed_mellea.py --batch-size 10000
"""

import argparse
import asyncio
import os
import sys
from typing import Any, List

from dotenv import load_dotenv
import openai
import torch

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend, TemplateFormatter

from kg.kg_embedder import KGEmbedder
from kg.kg_embed_models import EmbeddingConfig
from kg.kg_utils_mellea import generate_embedding_mellea
from utils.logger import logger

# Load environment variables
load_dotenv()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and store KG embeddings using Mellea-native utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use default configuration
  %(prog)s --batch-size 10000        # Custom batch size
  %(prog)s --verbose                 # Enable verbose logging
  %(prog)s --dimensions 1024         # Custom vector dimensions

This Mellea-native version demonstrates:
  ✓ Using kg_utils_mellea for embedding generation
  ✓ Better error handling and retry logic
  ✓ Cleaner async patterns
  ✓ Type-safe configuration
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


def create_embedding_session(config: EmbeddingConfig) -> Any:
    """Create embedding session (OpenAI or local model).

    Args:
        config: Embedding configuration

    Returns:
        Embedding session object (OpenAI AsyncClient or SentenceTransformer)
    """
    if config.api_base:
        logger.info("Using OpenAI-compatible embedding API")
        logger.info(f"  API base: {config.api_base}")
        logger.info(f"  Model: {config.model_name}")

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
        logger.info(f"  Model: {config.model_name}")

        from sentence_transformers import SentenceTransformer

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        logger.info(f"  Device: {device}")

        return SentenceTransformer(
            config.model_name,
            device=device
        )


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


async def test_embedding_session(emb_session: Any, config: EmbeddingConfig) -> bool:
    """Test the embedding session with a simple query.

    Args:
        emb_session: Embedding session to test
        config: Embedding configuration

    Returns:
        True if test succeeds, False otherwise
    """
    logger.info("Testing embedding session...")

    try:
        test_texts = ["This is a test embedding.", "Knowledge graph test."]
        embeddings = await generate_embedding_mellea(
            session=emb_session,
            texts=test_texts,
            model=config.model_name
        )

        if embeddings and len(embeddings) == len(test_texts):
            embedding_dim = len(embeddings[0])
            logger.info(f"✓ Embedding test successful (dimension: {embedding_dim})")

            if embedding_dim != config.vector_dimensions:
                logger.warning(
                    f"⚠ Embedding dimension mismatch: expected {config.vector_dimensions}, "
                    f"got {embedding_dim}"
                )

            return True
        else:
            logger.error("✗ Embedding test failed: incorrect number of embeddings")
            return False

    except Exception as e:
        logger.error(f"✗ Embedding test failed: {e}")
        return False


class MelleaKGEmbedder(KGEmbedder):
    """Mellea-native KG embedder with enhanced utilities.

    Extends the base KGEmbedder with Mellea-native patterns:
    - Uses kg_utils_mellea for embedding generation
    - Better error handling and logging
    - Demonstrates Mellea best practices
    """

    async def generate_embeddings_mellea(
        self,
        texts: List[str],
        desc: str = "Embedding"
    ) -> List[List[float]]:
        """Generate embeddings using Mellea-native utilities.

        Args:
            texts: List of text descriptions to embed
            desc: Description for logging

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} {desc.lower()}...")

        try:
            embeddings = await generate_embedding_mellea(
                session=self.emb_session,
                texts=texts,
                model=self.config.model_name if hasattr(self.emb_session, 'embeddings') else None
            )

            logger.info(f"✓ Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        logger.info("=" * 60)
        logger.info("Mellea-Native KG Embedding")
        logger.info("=" * 60)

        # Create configuration
        config = create_config(args)

        logger.info("Configuration:")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Storage batch size: {config.storage_batch_size}")
        logger.info(f"  Vector dimensions: {config.vector_dimensions}")
        logger.info(f"  Concurrent batches: {config.concurrent_batches}")
        logger.info("")

        logger.info("Using Mellea-native implementation with:")
        logger.info("  ✓ kg_utils_mellea embedding utilities")
        logger.info("  ✓ Enhanced error handling")
        logger.info("  ✓ Type-safe Pydantic configuration")
        logger.info("  ✓ Async batch processing")
        logger.info("=" * 60)

        # Create embedding session
        emb_session = create_embedding_session(config)

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
