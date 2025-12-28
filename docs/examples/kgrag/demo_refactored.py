"""Demo script for refactored KG-RAG using Mellea patterns.

This script demonstrates how to use the refactored KGRagComponent that follows
Mellea's design patterns including @generative functions, Requirements, and
Component architecture.

Usage:
    uv run --with mellea python demo_refactored.py
"""
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from kg_rag_refactored import KGRagComponent
from utils.logger import DefaultProgressLogger

# Try to import SentenceTransformer for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


async def main():
    """Run a simple KG-RAG demo."""
    # Load environment variables
    load_dotenv()

    # Configuration
    API_KEY = os.getenv("API_KEY", "dummy")
    API_BASE = os.getenv("API_BASE", "http://localhost:8000/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    TIME_OUT = int(os.getenv("TIME_OUT", "1800"))

    EMB_API_KEY = os.getenv("EMB_API_KEY", API_KEY)
    EMB_API_BASE = os.getenv("EMB_API_BASE", "")
    EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    EVAL_API_KEY = os.getenv("EVAL_API_KEY", API_KEY)
    EVAL_API_BASE = os.getenv("EVAL_API_BASE", API_BASE)
    EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", MODEL_NAME)

    print("=" * 80)
    print("KG-RAG Refactored Demo - Using Mellea Patterns")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API Base: {API_BASE}")
    print(f"  Embedding: {EMB_MODEL_NAME}")
    print(f"  Domain: movie")
    print()

    # Initialize main session
    print("Initializing Mellea session...")
    session = MelleaSession(
        backend=OpenAIBackend(
            model_id=MODEL_NAME,
            base_url=API_BASE,
            api_key=API_KEY,
            timeout=TIME_OUT,
        )
    )

    # Initialize evaluation session
    eval_session = MelleaSession(
        backend=OpenAIBackend(
            model_id=EVAL_MODEL_NAME,
            base_url=EVAL_API_BASE,
            api_key=EVAL_API_KEY,
            timeout=TIME_OUT,
        )
    )

    # Initialize embedding session
    if EMB_API_BASE:
        print(f"Using API-based embeddings: {EMB_MODEL_NAME}")
        from openai import AsyncOpenAI
        emb_session = AsyncOpenAI(
            api_key=EMB_API_KEY,
            base_url=EMB_API_BASE,
        )
    else:
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Using local embeddings: {EMB_MODEL_NAME}")
            emb_session = SentenceTransformer(EMB_MODEL_NAME)
        else:
            print("ERROR: sentence-transformers not installed and no EMB_API_BASE provided")
            print("Install with: pip install sentence-transformers")
            return

    # Initialize KG-RAG component
    print("\nInitializing KG-RAG component...")
    logger = DefaultProgressLogger()
    kg_rag = KGRagComponent(
        session=session,
        eval_session=eval_session,
        emb_session=emb_session,
        domain="movie",
        config={
            "route": 3,  # Explore 3 solving routes
            "width": 20,  # Consider top 20 relations
            "depth": 2,   # 2-hop graph traversal
        },
        logger=logger,
    )

    # Example queries
    queries = [
        {
            "query": "Who won the best actor Oscar in 2020?",
            "query_time": datetime(2024, 3, 19, 23, 49, 30),
        },
        {
            "query": "Which animated film won the best animated feature Oscar in 2024?",
            "query_time": datetime(2024, 3, 19, 23, 49, 30),
        },
    ]

    # Run queries
    for i, query_data in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"Query {i}: {query_data['query']}")
        print(f"Query Time: {query_data['query_time']}")
        print("=" * 80)

        try:
            # Execute KG-RAG pipeline
            answer, details = await kg_rag.execute(
                query=query_data["query"],
                query_time=query_data["query_time"],
                return_details=True,
            )

            print(f"\n{'Answer':-^80}")
            print(f"{answer}")
            print()

            # Show route details
            if details:
                print(f"\n{'Route Details':-^80}")
                for j, route_result in enumerate(details, 1):
                    print(f"\nRoute {j}:")
                    print(f"  Sub-objectives: {route_result['query'].subqueries}")
                    print(f"  Answer: {route_result['ans'][:100]}...")
                    print(f"  Entities found: {len(route_result['entities'])}")
                    print(f"  Relations found: {len(route_result['relations'])}")

        except Exception as e:
            print(f"\nERROR: {e}")
            logger.error("Query failed", exc_info=True)

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
