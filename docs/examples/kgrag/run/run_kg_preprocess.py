#!/usr/bin/env python3
"""
Knowledge Graph Preprocessing Script

This script preprocesses and loads knowledge graph data into Neo4j.
Supports multiple domains: movie, soccer, nba, music, multitq, timequestions.

Usage:
    python run_kg_preprocess.py --domain movie
    python run_kg_preprocess.py --domain all
    python run_kg_preprocess.py --domain movie soccer nba
"""

import argparse
import asyncio
import sys
from typing import List

from kg.kg_preprocessor import (
    KGPreprocessorBase as KG_Preprocessor,
    MovieKGPreprocessor as MovieKG_Preprocessor,
)

from utils.logger import logger


# Domain to preprocessor class mapping
DOMAIN_PREPROCESSORS = {
    "movie": MovieKG_Preprocessor,
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess and load knowledge graph data into Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --domain movie              # Process movie domain only
  %(prog)s --domain all                # Process all available domains
  %(prog)s --domain movie soccer       # Process multiple specific domains
  %(prog)s --domain movie --dry-run    # Preview without executing
        """
    )

    parser.add_argument(
        "--domain",
        nargs="+",
        choices=list(DOMAIN_PREPROCESSORS.keys()) + ["all"],
        default=["movie"],
        help="Domain(s) to preprocess. Use 'all' for all domains (default: movie)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which preprocessors would run without executing"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def get_preprocessors(domains: List[str]) -> List[KG_Preprocessor]:
    """
    Get preprocessor instances for the specified domains.

    Args:
        domains: List of domain names or ["all"]

    Returns:
        List of preprocessor instances
    """
    if "all" in domains:
        domains = list(DOMAIN_PREPROCESSORS.keys())

    preprocessors = []
    for domain in domains:
        try:
            preprocessor_class = DOMAIN_PREPROCESSORS[domain]
            logger.info(f"Initializing {domain} preprocessor...")
            preprocessors.append(preprocessor_class())
            logger.info(f"✓ {domain} preprocessor initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize {domain} preprocessor: {e}")
            raise

    return preprocessors


async def run_preprocessing(preprocessors: List[KG_Preprocessor]) -> None:
    """
    Run preprocessing for all provided preprocessors.

    Args:
        preprocessors: List of preprocessor instances
    """
    total = len(preprocessors)
    logger.info(f"Starting preprocessing for {total} domain(s)...")

    for idx, preprocessor in enumerate(preprocessors, 1):
        domain_name = preprocessor.__class__.__name__.replace("KG_Preprocessor", "").replace("Preprocessor", "")
        try:
            logger.info(f"[{idx}/{total}] Processing {domain_name}...")

            # Connect to Neo4j (refactored preprocessors need explicit connect)
            if hasattr(preprocessor, 'connect'):
                await preprocessor.connect()

            await preprocessor.preprocess()
            logger.info(f"[{idx}/{total}] ✓ {domain_name} completed")
        except Exception as e:
            logger.error(f"[{idx}/{total}] ✗ {domain_name} failed: {e}")
            raise
        finally:
            # Always close the connection, even if preprocessing fails
            try:
                await preprocessor.close()
            except Exception as e:
                logger.warning(f"Failed to close {domain_name} preprocessor: {e}")


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging verbosity
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        # Get preprocessors for selected domains
        preprocessors = get_preprocessors(args.domain)

        if args.dry_run:
            logger.info("DRY RUN MODE - No data will be processed")
            logger.info(f"Would process {len(preprocessors)} domain(s):")
            for p in preprocessors:
                domain_name = p.__class__.__name__.replace("KG_Preprocessor", "").replace("Preprocessor", "")
                logger.info(f"  - {domain_name}")
            return 0

        # Run preprocessing
        await run_preprocessing(preprocessors)

        logger.info("=" * 60)
        logger.info("✅ All data successfully imported to Neo4j!")
        logger.info("=" * 60)
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Preprocessing interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
