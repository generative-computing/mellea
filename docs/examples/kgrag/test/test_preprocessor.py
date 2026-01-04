#!/usr/bin/env python3
"""
Test script for the refactored KG preprocessor.

This script demonstrates the usage of the refactored preprocessor
and validates that it works correctly with the new architecture.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from kg.kg_preprocessor import MovieKGPreprocessor
from kg.kg_entity_models import Neo4jConfig, PreprocessorConfig
from utils.logger import logger


async def test_basic_usage():
    """Test basic usage with default configuration."""
    logger.info("=" * 60)
    logger.info("Test 1: Basic Usage with Default Config")
    logger.info("=" * 60)

    try:
        preprocessor = MovieKGPreprocessor()
        logger.info("✓ MovieKGPreprocessor initialized successfully")

        # Test connection
        await preprocessor.connect()
        logger.info("✓ Connected to Neo4j successfully")

        # Test basic query
        result = await preprocessor.execute_query("RETURN 1 as test")
        assert len(result) > 0, "Query should return results"
        logger.info(f"✓ Basic query test passed: {result}")

        await preprocessor.close()
        logger.info("✓ Connection closed successfully")

        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_custom_config():
    """Test with custom configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Custom Configuration")
    logger.info("=" * 60)

    try:
        # Create custom configuration
        config = PreprocessorConfig(
            neo4j=Neo4jConfig(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",  # Update with your password
                max_concurrency=25,
                max_retries=3,
                retry_delay=1.0
            ),
            batch_size=5000,
            sample_fractions={"Movie": 0.8, "Person": 0.8}
        )

        preprocessor = MovieKGPreprocessor(config)
        logger.info("✓ MovieKGPreprocessor with custom config initialized")
        logger.info(f"  - Max concurrency: {config.neo4j.max_concurrency}")
        logger.info(f"  - Batch size: {config.batch_size}")
        logger.info(f"  - Sample fractions: {config.sample_fractions}")

        await preprocessor.connect()
        logger.info("✓ Connected with custom config")

        await preprocessor.close()
        logger.info("✓ Closed successfully")

        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_insert():
    """Test batch insert functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Batch Insert")
    logger.info("=" * 60)

    try:
        preprocessor = MovieKGPreprocessor()
        await preprocessor.connect()

        # Create test data
        test_data = [
            {"name": f"TEST_MOVIE_{i}", "year": 2020 + i}
            for i in range(10)
        ]

        # Test batch insert
        query = """
        UNWIND $batch AS movie
        MERGE (m:TestMovie {name: movie.name})
        SET m.year = movie.year
        """

        await preprocessor.batch_insert(query, test_data, desc="Test Movies")
        logger.info("✓ Batch insert completed")

        # Verify insertion
        result = await preprocessor.execute_query(
            "MATCH (m:TestMovie) RETURN count(m) as count"
        )
        count = result[0]["count"] if result else 0
        logger.info(f"✓ Inserted {count} test movies")

        # Cleanup
        await preprocessor.execute_query("MATCH (m:TestMovie) DELETE m")
        logger.info("✓ Test data cleaned up")

        await preprocessor.close()
        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_index_creation():
    """Test index creation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Index Creation")
    logger.info("=" * 60)

    try:
        preprocessor = MovieKGPreprocessor()
        await preprocessor.connect()

        # Test index creation
        await preprocessor.create_index_if_not_exists("TestNode", "test_property")
        logger.info("✓ Index creation method works")

        await preprocessor.close()
        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests."""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 RUNNING REFACTORED PREPROCESSOR TESTS")
    logger.info("=" * 70)

    tests = [
        ("Basic Usage", test_basic_usage),
        ("Custom Config", test_custom_config),
        ("Batch Insert", test_batch_insert),
        ("Index Creation", test_index_creation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {name}")

    logger.info("=" * 70)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 70)

    return passed == total


async def main():
    """Main entry point."""
    try:
        success = await run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
