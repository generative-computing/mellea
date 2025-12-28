#!/usr/bin/env python3
"""Create a smaller demo dataset from the full CRAG movie database.

This script extracts a focused subset of movies, people, and years to create
a lightweight demo database that's faster to process and easier to work with.

Usage:
    python create_demo_dataset.py --year-start 2020 --year-end 2024 --max-movies 100
    python create_demo_dataset.py --topics "oscar,animated,marvel" --max-movies 150
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a smaller demo dataset from the full CRAG movie database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recent movies (2020-2024) with up to 100 movies
  %(prog)s --year-start 2020 --year-end 2024 --max-movies 100

  # Award winners and nominees
  %(prog)s --topics "oscar,golden globe,bafta" --max-movies 150

  # Specific franchises
  %(prog)s --topics "marvel,star wars,harry potter" --max-movies 200

  # Animated films
  %(prog)s --topics "animated,pixar,disney" --max-movies 100
        """
    )

    parser.add_argument(
        "--year-start",
        type=int,
        default=2020,
        help="Start year for movies (default: 2020)"
    )

    parser.add_argument(
        "--year-end",
        type=int,
        default=2024,
        help="End year for movies (default: 2024)"
    )

    parser.add_argument(
        "--max-movies",
        type=int,
        default=100,
        help="Maximum number of movies to include (default: 100)"
    )

    parser.add_argument(
        "--topics",
        type=str,
        default="",
        help="Comma-separated topics to filter by (e.g., 'oscar,marvel,animated')"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/movie",
        help="Input directory with full database (default: dataset/movie)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/movie_demo",
        help="Output directory for demo database (default: dataset/movie_demo)"
    )

    parser.add_argument(
        "--include-related",
        action="store_true",
        help="Include all people and years related to selected movies"
    )

    return parser.parse_args()


def load_json_db(file_path: str) -> Dict[str, Any]:
    """Load a JSON database file."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_db(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON database file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def matches_topics(movie: Dict[str, Any], topics: List[str]) -> bool:
    """Check if a movie matches any of the specified topics."""
    if not topics:
        return True

    # Convert movie data to lowercase for case-insensitive matching
    movie_text = json.dumps(movie, default=str).lower()

    # Check if any topic appears in the movie data
    return any(topic.lower() in movie_text for topic in topics)


def extract_person_ids(movie: Dict[str, Any]) -> Set[str]:
    """Extract all person IDs referenced in a movie."""
    person_ids = set()

    # Check various fields that might contain person references
    for field in ['cast', 'director', 'producer', 'writer', 'crew']:
        if field in movie and isinstance(movie[field], list):
            for person in movie[field]:
                if isinstance(person, dict) and 'id' in person:
                    person_ids.add(str(person['id']))
                elif isinstance(person, str):
                    person_ids.add(person)

    return person_ids


def extract_year_ids(movie: Dict[str, Any]) -> Set[str]:
    """Extract all year IDs referenced in a movie."""
    year_ids = set()

    # Check release date and other date fields
    for field in ['release_date', 'year', 'premiere_date']:
        if field in movie:
            value = movie[field]
            if isinstance(value, str):
                # Extract year from date string
                try:
                    year = value.split('-')[0] if '-' in value else value[:4]
                    if year.isdigit():
                        year_ids.add(year)
                except:
                    pass
            elif isinstance(value, int):
                year_ids.add(str(value))

    return year_ids


def create_demo_dataset(args: argparse.Namespace) -> None:
    """Create a demo dataset based on the specified criteria."""
    print("=" * 60)
    print("Creating Demo Dataset")
    print("=" * 60)
    print(f"Year range: {args.year_start}-{args.year_end}")
    print(f"Max movies: {args.max_movies}")
    if args.topics:
        print(f"Topics: {args.topics}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load full databases
    movie_db = load_json_db(os.path.join(args.input_dir, "movie_db.json"))
    person_db = load_json_db(os.path.join(args.input_dir, "person_db.json"))
    year_db = load_json_db(os.path.join(args.input_dir, "year_db.json"))

    print(f"\nFull database sizes:")
    print(f"  Movies: {len(movie_db)}")
    print(f"  People: {len(person_db)}")
    print(f"  Years: {len(year_db)}")

    # Parse topics
    topics = [t.strip() for t in args.topics.split(',')] if args.topics else []

    # Filter movies
    selected_movies = {}
    all_person_ids = set()
    all_year_ids = set()

    print(f"\nFiltering movies...")
    for movie_id, movie in movie_db.items():
        # Skip if we've reached max movies
        if len(selected_movies) >= args.max_movies:
            break

        # Check year range
        release_year = None
        if 'release_date' in movie:
            try:
                year_str = movie['release_date'].split('-')[0]
                release_year = int(year_str)
            except:
                pass

        if release_year and (release_year < args.year_start or release_year > args.year_end):
            continue

        # Check topics
        if not matches_topics(movie, topics):
            continue

        # Add movie
        selected_movies[movie_id] = movie

        # Track related IDs if requested
        if args.include_related:
            all_person_ids.update(extract_person_ids(movie))
            all_year_ids.update(extract_year_ids(movie))

    print(f"Selected {len(selected_movies)} movies")

    # Filter people
    selected_people = {}
    if args.include_related and all_person_ids:
        print(f"Filtering people (found {len(all_person_ids)} references)...")
        for person_id in all_person_ids:
            if person_id in person_db:
                selected_people[person_id] = person_db[person_id]
    else:
        # Include a subset of people if not doing related filtering
        person_ids = list(person_db.keys())[:min(len(person_db), args.max_movies * 5)]
        selected_people = {pid: person_db[pid] for pid in person_ids}

    print(f"Selected {len(selected_people)} people")

    # Filter years
    selected_years = {}
    if args.include_related and all_year_ids:
        print(f"Filtering years (found {len(all_year_ids)} references)...")
        for year_id in all_year_ids:
            if year_id in year_db:
                selected_years[year_id] = year_db[year_id]
    else:
        # Include years in the specified range
        for year in range(args.year_start, args.year_end + 1):
            year_str = str(year)
            if year_str in year_db:
                selected_years[year_str] = year_db[year_str]

    print(f"Selected {len(selected_years)} years")

    # Save demo databases
    print(f"\nSaving demo databases to {args.output_dir}...")
    save_json_db(selected_movies, os.path.join(args.output_dir, "movie_db.json"))
    save_json_db(selected_people, os.path.join(args.output_dir, "person_db.json"))
    save_json_db(selected_years, os.path.join(args.output_dir, "year_db.json"))

    # Print statistics
    print("\n" + "=" * 60)
    print("Demo Dataset Created Successfully!")
    print("=" * 60)
    print(f"Demo database sizes:")
    print(f"  Movies: {len(selected_movies)} ({len(selected_movies)/len(movie_db)*100:.1f}% of full)")
    print(f"  People: {len(selected_people)} ({len(selected_people)/len(person_db)*100:.1f}% of full)")
    print(f"  Years: {len(selected_years)} ({len(selected_years)/len(year_db)*100:.1f}% of full)")
    print(f"\nOutput directory: {args.output_dir}")
    print("\nTo use the demo database, update your .env file:")
    print(f'  KG_BASE_DIRECTORY="{os.path.abspath(args.output_dir)}/.."')
    print("\nOr create a symbolic link:")
    print(f'  ln -s {args.output_dir} dataset/movie_original')
    print(f'  mv dataset/movie dataset/movie_full')
    print(f'  ln -s movie_demo dataset/movie')


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        create_demo_dataset(args)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure the input directory exists and contains:")
        print(f"  - movie_db.json")
        print(f"  - person_db.json")
        print(f"  - year_db.json")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
