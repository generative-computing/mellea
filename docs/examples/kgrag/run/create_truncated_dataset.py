#!/usr/bin/env python3
"""
Create a truncated version of the dataset with shorter page_result documents.
This speeds up KG update by reducing the amount of text to process.

Usage:
    python create_truncated_dataset.py --input dataset/crag_movie_tiny.jsonl.bz2 --output dataset/crag_movie_tiny_truncated.jsonl.bz2 --max-chars 50000
"""

import argparse
import bz2
import json
from pathlib import Path


def truncate_document(doc_text: str, max_chars: int) -> str:
    """Truncate document text to max_chars while trying to end at a sentence boundary.

    Args:
        doc_text: Original document text
        max_chars: Maximum number of characters

    Returns:
        Truncated document text
    """
    if len(doc_text) <= max_chars:
        return doc_text

    # Try to find a sentence boundary near the max_chars limit
    truncated = doc_text[:max_chars]

    # Look for sentence endings in the last 500 chars
    search_back = min(500, max_chars // 10)
    search_region = truncated[-search_back:]

    # Try to find good breaking points (in order of preference)
    for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n', '\n']:
        pos = search_region.rfind(delimiter)
        if pos != -1:
            # Found a good breaking point
            actual_pos = len(truncated) - search_back + pos + len(delimiter)
            return truncated[:actual_pos]

    # No good breaking point found, just truncate at max_chars
    return truncated


def main():
    parser = argparse.ArgumentParser(
        description="Truncate documents in CRAG dataset to speed up processing"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset file (e.g., crag_movie_tiny.jsonl.bz2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset file (e.g., crag_movie_tiny_truncated.jsonl.bz2)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50000,
        help="Maximum characters per document (default: 50000)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Max chars per document: {args.max_chars}")
    print()

    total_docs = 0
    total_pages = 0
    chars_before = 0
    chars_after = 0

    # Process the dataset
    with bz2.open(input_path, 'rt', encoding='utf-8') as infile:
        with bz2.open(output_path, 'wt', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    doc = json.loads(line)
                    total_docs += 1

                    # Process each search result
                    for i, result in enumerate(doc.get('search_results', [])):
                        if 'page_result' in result:
                            total_pages += 1
                            original = result['page_result']
                            chars_before += len(original)

                            truncated = truncate_document(original, args.max_chars)
                            result['page_result'] = truncated
                            chars_after += len(truncated)

                            if len(truncated) < len(original):
                                print(f"Doc {total_docs}, Page {i+1}: {len(original):,} -> {len(truncated):,} chars")

                    # Write modified document
                    outfile.write(json.dumps(doc) + '\n')

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to decode line {line_num}: {e}")
                    continue

    print()
    print("=" * 60)
    print("Truncation complete!")
    print("=" * 60)
    print(f"Documents processed: {total_docs}")
    print(f"Total pages: {total_pages}")
    print(f"Total chars before: {chars_before:,}")
    print(f"Total chars after: {chars_after:,}")
    print(f"Reduction: {(1 - chars_after/chars_before)*100:.1f}%")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    exit(main())
