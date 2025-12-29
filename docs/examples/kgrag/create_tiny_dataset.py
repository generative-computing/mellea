#!/usr/bin/env python3
"""Create a tiny dataset for quick testing (10-20 documents).

This script creates an extremely small dataset for rapid testing and development.
It takes the first N documents from the full dataset.

Usage:
    python create_tiny_dataset.py --num-docs 10
    python create_tiny_dataset.py --num-docs 20 --output dataset/crag_movie_tiny.jsonl.bz2
"""

import argparse
import bz2
import json
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a tiny dataset for quick testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 10-document dataset
  %(prog)s --num-docs 10

  # Create 20-document dataset
  %(prog)s --num-docs 20

  # Create 5-document dataset for ultra-fast testing
  %(prog)s --num-docs 5 --output dataset/crag_movie_micro.jsonl.bz2
        """
    )

    parser.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Number of documents to include (default: 10)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="dataset/crag_movie_dev.jsonl.bz2",
        help="Input dataset file (default: dataset/crag_movie_dev.jsonl.bz2)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/crag_movie_tiny.jsonl.bz2",
        help="Output dataset file (default: dataset/crag_movie_tiny.jsonl.bz2)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    print("=" * 60)
    print("Creating Tiny Test Dataset")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Number of documents: {args.num_docs}")
    print("=" * 60)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Read first N documents from input
    documents = []
    try:
        with bz2.open(args.input, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= args.num_docs:
                    break
                try:
                    doc = json.loads(line.strip())
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i+1}: {e}")
                    continue

        print(f"\nRead {len(documents)} documents from input")

        # Write to output
        with bz2.open(args.output, 'wt', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        print(f"Wrote {len(documents)} documents to {args.output}")

        # Print some statistics
        print("\n" + "=" * 60)
        print("Tiny Dataset Created Successfully!")
        print("=" * 60)
        print(f"Total documents: {len(documents)}")
        print(f"Output file: {args.output}")

        # Show first document as example
        if documents:
            print("\nFirst document fields:")
            for key in documents[0].keys():
                print(f"  - {key}")

        print("\nTo use this dataset, either:")
        print("  1. Update KG_BASE_DIRECTORY in your .env to point to the dataset directory")
        print(f"  2. Or pass --dataset {args.output} to run_kg_update.py")

        return 0

    except FileNotFoundError:
        print(f"\nError: Input file not found: {args.input}")
        print("\nMake sure you have the full dataset at:")
        print(f"  {args.input}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
