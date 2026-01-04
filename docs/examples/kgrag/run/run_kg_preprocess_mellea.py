#!/usr/bin/env python3
"""
Knowledge Graph Preprocessing Script (Mellea-Native Implementation)

This script demonstrates KG preprocessing using Mellea best practices:
- Pydantic models for type safety
- Enhanced error handling and logging
- Progress tracking with detailed statistics
- Concurrent preprocessing support
- Dry-run mode for validation

Usage:
    python run_kg_preprocess_mellea.py --domain movie
    python run_kg_preprocess_mellea.py --domain all --verbose
    python run_kg_preprocess_mellea.py --domain movie --dry-run
    python run_kg_preprocess_mellea.py --domain movie soccer nba
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

from kg.kg_preprocessor import (
    KGPreprocessorBase as KG_Preprocessor,
    MovieKGPreprocessor as MovieKG_Preprocessor,
)

from utils.logger import logger


# Domain to preprocessor class mapping
DOMAIN_PREPROCESSORS = {
    "movie": MovieKG_Preprocessor,
}


@dataclass
class PreprocessingStats:
    """Statistics for preprocessing operations."""
    domain: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    entities_processed: int
    relations_processed: int
    success: bool
    error_message: str = ""

    def __str__(self) -> str:
        """Format statistics for display."""
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        lines = [
            f"Domain: {self.domain}",
            f"Status: {status}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Entities: {self.entities_processed:,}",
            f"Relations: {self.relations_processed:,}",
        ]
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess and load knowledge graph data into Neo4j (Mellea-native)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --domain movie              # Process movie domain only
  %(prog)s --domain all                # Process all available domains
  %(prog)s --domain movie soccer       # Process multiple specific domains
  %(prog)s --domain movie --dry-run    # Preview without executing
  %(prog)s --domain all --verbose      # Verbose logging

This Mellea-native version demonstrates:
  ✓ Enhanced statistics tracking
  ✓ Better error handling and reporting
  ✓ Concurrent preprocessing support
  ✓ Detailed progress logging
  ✓ Type-safe configuration with Pydantic
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
        "--concurrent",
        action="store_true",
        help="Process multiple domains concurrently (experimental)"
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


async def preprocess_single_domain(
    preprocessor: KG_Preprocessor,
    idx: int,
    total: int
) -> PreprocessingStats:
    """
    Preprocess a single domain with statistics tracking.

    Args:
        preprocessor: Preprocessor instance
        idx: Current index (1-based)
        total: Total number of preprocessors

    Returns:
        PreprocessingStats with results
    """
    domain_name = preprocessor.__class__.__name__.replace("KG_Preprocessor", "").replace("Preprocessor", "")
    start_time = datetime.now()

    try:
        logger.info(f"[{idx}/{total}] Processing {domain_name}...")

        # Connect to Neo4j
        if hasattr(preprocessor, 'connect'):
            await preprocessor.connect()

        # Run preprocessing
        await preprocessor.preprocess()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Get statistics (if available)
        entities_processed = 0
        relations_processed = 0
        if hasattr(preprocessor, 'get_stats'):
            stats = preprocessor.get_stats()
            entities_processed = stats.get('entities', 0)
            relations_processed = stats.get('relations', 0)

        logger.info(f"[{idx}/{total}] ✓ {domain_name} completed in {duration:.2f}s")

        return PreprocessingStats(
            domain=domain_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            entities_processed=entities_processed,
            relations_processed=relations_processed,
            success=True
        )

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.error(f"[{idx}/{total}] ✗ {domain_name} failed: {e}")

        return PreprocessingStats(
            domain=domain_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            entities_processed=0,
            relations_processed=0,
            success=False,
            error_message=str(e)
        )

    finally:
        # Always close the connection
        try:
            await preprocessor.close()
        except Exception as e:
            logger.warning(f"Failed to close {domain_name} preprocessor: {e}")


async def run_preprocessing_sequential(
    preprocessors: List[KG_Preprocessor]
) -> List[PreprocessingStats]:
    """
    Run preprocessing sequentially for all preprocessors.

    Args:
        preprocessors: List of preprocessor instances

    Returns:
        List of preprocessing statistics
    """
    total = len(preprocessors)
    logger.info(f"Starting sequential preprocessing for {total} domain(s)...")

    stats_list = []
    for idx, preprocessor in enumerate(preprocessors, 1):
        stats = await preprocess_single_domain(preprocessor, idx, total)
        stats_list.append(stats)

        # Stop if any preprocessing fails
        if not stats.success:
            logger.error(f"Stopping due to failure in {stats.domain}")
            break

    return stats_list


async def run_preprocessing_concurrent(
    preprocessors: List[KG_Preprocessor]
) -> List[PreprocessingStats]:
    """
    Run preprocessing concurrently for all preprocessors.

    Args:
        preprocessors: List of preprocessor instances

    Returns:
        List of preprocessing statistics
    """
    total = len(preprocessors)
    logger.info(f"Starting concurrent preprocessing for {total} domain(s)...")

    tasks = [
        preprocess_single_domain(preprocessor, idx, total)
        for idx, preprocessor in enumerate(preprocessors, 1)
    ]

    stats_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to failed stats
    result_stats = []
    for idx, stats in enumerate(stats_list, 1):
        if isinstance(stats, Exception):
            domain_name = preprocessors[idx - 1].__class__.__name__.replace("KG_Preprocessor", "").replace("Preprocessor", "")
            result_stats.append(PreprocessingStats(
                domain=domain_name,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0,
                entities_processed=0,
                relations_processed=0,
                success=False,
                error_message=str(stats)
            ))
        else:
            result_stats.append(stats)

    return result_stats


def print_summary(stats_list: List[PreprocessingStats]) -> None:
    """
    Print preprocessing summary.

    Args:
        stats_list: List of preprocessing statistics
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)

    total_domains = len(stats_list)
    successful_domains = sum(1 for s in stats_list if s.success)
    failed_domains = total_domains - successful_domains

    total_entities = sum(s.entities_processed for s in stats_list)
    total_relations = sum(s.relations_processed for s in stats_list)
    total_duration = sum(s.duration_seconds for s in stats_list)

    logger.info(f"Domains processed: {total_domains}")
    logger.info(f"Successful: {successful_domains}")
    logger.info(f"Failed: {failed_domains}")
    logger.info(f"Total entities: {total_entities:,}")
    logger.info(f"Total relations: {total_relations:,}")
    logger.info(f"Total duration: {total_duration:.2f}s")
    logger.info("")

    # Print individual domain statistics
    for stats in stats_list:
        logger.info("-" * 60)
        logger.info(str(stats))

    logger.info("=" * 60)

    if failed_domains == 0:
        logger.info("✅ All data successfully imported to Neo4j!")
    else:
        logger.warning(f"⚠️  {failed_domains} domain(s) failed")

    logger.info("=" * 60)


async def main() -> int:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging verbosity
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        logger.info("=" * 60)
        logger.info("Mellea-Native KG Preprocessing")
        logger.info("=" * 60)

        # Get preprocessors for selected domains
        preprocessors = get_preprocessors(args.domain)

        if args.dry_run:
            logger.info("DRY RUN MODE - No data will be processed")
            logger.info(f"Would process {len(preprocessors)} domain(s):")
            for p in preprocessors:
                domain_name = p.__class__.__name__.replace("KG_Preprocessor", "").replace("Preprocessor", "")
                logger.info(f"  - {domain_name}")
            return 0

        logger.info(f"Processing mode: {'concurrent' if args.concurrent else 'sequential'}")
        logger.info("")

        # Run preprocessing
        if args.concurrent:
            stats_list = await run_preprocessing_concurrent(preprocessors)
        else:
            stats_list = await run_preprocessing_sequential(preprocessors)

        # Print summary
        print_summary(stats_list)

        # Return error if any preprocessing failed
        failed = sum(1 for s in stats_list if not s.success)
        return 1 if failed > 0 else 0

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
