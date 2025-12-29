"""Refactored Knowledge Graph Preprocessor following Mellea patterns.

This module provides a cleaner, more maintainable implementation of KG preprocessing
with:
- Pydantic models for type safety
- Better separation of concerns
- Modern async patterns
- Proper error handling
- Configurable retry logic
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, TypeVar
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import TransientError
from tqdm import tqdm
from pydantic import ValidationError

from kg.kg_entity_models import (
    Neo4jConfig,
    PreprocessorConfig,
    Movie,
    Person,
    KGEntity,
    KGRelation,
)
from kg.kg_rep import normalize_entity, normalize_relation, normalize_value
from utils.logger import logger

# Load environment variables
load_dotenv()

T = TypeVar('T')


class Neo4jConnection:
    """Manages Neo4j database connections with proper resource management."""

    def __init__(self, config: Neo4jConfig):
        """Initialize Neo4j connection.

        Args:
            config: Neo4j configuration with connection details
        """
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self.driver is None:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password)
            )
            logger.info(f"Connected to Neo4j at {self.config.uri}")

    async def close(self) -> None:
        """Close connection to Neo4j."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")

    @asynccontextmanager
    async def session(self):
        """Get a Neo4j session with proper resource management."""
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")

        async with self._semaphore:
            async with self.driver.session() as session:
                yield session

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None,
        delay: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query with retry logic.

        Args:
            query: Cypher query to execute
            parameters: Query parameters
            retries: Number of retries (defaults to config)
            delay: Delay between retries (defaults to config)

        Returns:
            List of result records as dictionaries

        Raises:
            RuntimeError: If max retries exceeded
        """
        retries = retries or self.config.max_retries
        delay = delay or self.config.retry_delay

        for attempt in range(retries):
            try:
                async with self.session() as session:
                    result = await session.run(query, parameters)
                    return [dict(record) async for record in result]
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    if attempt < retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(
                            f"Deadlock detected, retrying {attempt + 1}/{retries} "
                            f"(waiting {wait_time:.2f}s)"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries reached for query: {query[:100]}...")
                        raise RuntimeError(f"Max retries exceeded: {e}")
                else:
                    raise

        raise RuntimeError("Max retries reached")


class KGPreprocessorBase(ABC):
    """Base class for knowledge graph preprocessors.

    Provides common functionality for loading data, connecting to Neo4j,
    and managing preprocessing workflows.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        """Initialize preprocessor.

        Args:
            config: Preprocessor configuration (created from env if None)
        """
        self.config = config or self._load_config_from_env()
        self.connection = Neo4jConnection(self.config.neo4j)

    @staticmethod
    def _load_config_from_env() -> PreprocessorConfig:
        """Load configuration from environment variables."""
        return PreprocessorConfig(
            neo4j=Neo4jConfig(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", ""),
                max_concurrency=int(os.getenv("NEO4J_MAX_CONCURRENCY", "50")),
                max_retries=int(os.getenv("NEO4J_MAX_RETRIES", "5")),
                retry_delay=float(os.getenv("NEO4J_RETRY_DELAY", "0.5"))
            ),
            kg_base_directory=os.getenv("KG_BASE_DIRECTORY", "docs/examples/kgrag/dataset"),
            batch_size=int(os.getenv("KG_BATCH_SIZE", "10000"))
        )

    async def connect(self) -> None:
        """Connect to Neo4j database."""
        await self.connection.connect()

    async def close(self) -> None:
        """Close Neo4j connection."""
        await self.connection.close()

    async def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query.

        Args:
            query: Cypher query
            parameters: Query parameters

        Returns:
            Query results
        """
        return await self.connection.execute_query(query, parameters)

    @abstractmethod
    async def create_indices(self) -> None:
        """Create database indices for faster querying."""
        pass

    @abstractmethod
    async def preprocess(self) -> None:
        """Main preprocessing pipeline."""
        pass

    async def create_index_if_not_exists(self, node_label: str, property_name: str) -> None:
        """Create an index on a node property.

        Args:
            node_label: Node label (e.g., "Movie", "Person")
            property_name: Property to index (e.g., "name", "id")
        """
        query = f"CREATE INDEX IF NOT EXISTS FOR (n:{node_label}) ON (n.{property_name})"
        await self.execute_query(query)
        logger.debug(f"Created index on {node_label}.{property_name}")

    async def batch_insert(
        self,
        query: str,
        data: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        desc: str = "Inserting"
    ) -> None:
        """Insert data in batches for better performance.

        Args:
            query: Cypher query with UNWIND $batch pattern
            data: List of data items to insert
            batch_size: Batch size (defaults to config)
            desc: Progress bar description
        """
        batch_size = batch_size or self.config.batch_size

        for i in tqdm(range(0, len(data), batch_size), desc=desc):
            batch = data[i:i + batch_size]
            await self.execute_query(query, {"batch": batch})

    def load_json_file(self, file_path: str, model_class: Optional[Type[T]] = None) -> Dict[str, Any]:
        """Load and optionally validate JSON file.

        Args:
            file_path: Path to JSON file
            model_class: Optional Pydantic model for validation

        Returns:
            Loaded JSON data (validated if model_class provided)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If validation fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading {path.name}...")
        with open(path) as f:
            data = json.load(f)

        if model_class:
            # Validate each item if it's a dictionary of items
            if isinstance(data, dict):
                validated = {}
                for key, value in data.items():
                    try:
                        validated[key] = model_class(**value)
                    except ValidationError as e:
                        logger.warning(f"Validation error for {key}: {e}")
                        # Keep original data if validation fails
                        validated[key] = value
                return validated
            elif isinstance(data, list):
                return [model_class(**item) for item in data]

        return data


class MovieKGPreprocessor(KGPreprocessorBase):
    """Movie domain knowledge graph preprocessor.

    Loads movie, person, and year data into Neo4j with proper relationships.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        """Initialize movie preprocessor."""
        super().__init__(config)

        # Define data paths
        base_dir = Path(self.config.kg_base_directory)
        movie_dir = base_dir / "movie"

        # Load data with validation
        self.movie_db = self.load_json_file(str(movie_dir / "movie_db.json"))
        self.person_db = self.load_json_file(str(movie_dir / "person_db.json"))
        self.year_db = self.load_json_file(str(movie_dir / "year_db.json"))

        logger.info("Movie data loaded successfully")

    async def create_indices(self) -> None:
        """Create indices for movie domain."""
        indices = [
            ("Movie", "name"),
            ("Person", "name"),
            ("Award", "name"),
            ("Genre", "name"),
            ("Year", "name"),
        ]

        logger.info("Creating indices...")
        for label, prop in indices:
            await self.create_index_if_not_exists(label, prop)

    async def preprocess(self) -> None:
        """Run the full movie preprocessing pipeline."""
        await self.create_indices()

        # Insert entities
        logger.info("Inserting movie entities...")
        await self.insert_all_movies()

        logger.info("Inserting person entities...")
        await self.insert_all_persons()

        logger.info("Inserting year entities...")
        await self.insert_all_years()

        # Insert relations
        logger.info("Inserting movie relations...")
        await self.insert_all_movie_relations()

        logger.info("Inserting person relations...")
        await self.insert_all_person_relations()

        logger.info("Inserting year relations...")
        await self.insert_all_year_relations()

        # Sample KG to simulate incomplete data
        logger.info("Sampling knowledge graph...")
        await self.sample_kg()

    # ===== Entity Insertion Methods =====

    async def insert_all_movies(self) -> None:
        """Insert all movie entities."""
        query = """
        UNWIND $batch AS movie
        MERGE (m:Movie {name: movie.name})
        SET m.release_date = movie.release_date,
            m.original_name = movie.original_name,
            m.original_language = movie.original_language,
            m.budget = movie.budget,
            m.revenue = movie.revenue,
            m.rating = movie.rating
        """

        data = [
            {
                "name": movie["title"].upper(),
                "original_name": movie.get("original_title"),
                "release_date": movie.get("release_date"),
                "original_language": movie.get("original_language"),
                "budget": str(movie["budget"]) if "budget" in movie else None,
                "revenue": str(movie["revenue"]) if "revenue" in movie else None,
                "rating": str(movie["rating"]) if "rating" in movie else None,
            }
            for movie in self.movie_db.values()
        ]

        await self.batch_insert(query, data, desc="Movies")

    async def insert_all_persons(self) -> None:
        """Insert all person entities."""
        query = """
        UNWIND $batch AS person
        MERGE (p:Person {name: person.name})
        SET p.birthday = person.birthday
        """

        data = [
            {
                "name": person["name"].upper(),
                "birthday": person.get("birthday"),
            }
            for person in self.person_db.values()
        ]

        await self.batch_insert(query, data, desc="Persons")

    async def insert_all_years(self) -> None:
        """Insert year entities."""
        query = """
        UNWIND $batch AS year
        MERGE (y:Year {name: year.name})
        """

        data = [{"name": str(year)} for year in range(1990, 2022)]
        await self.batch_insert(query, data, desc="Years")

    # ===== Relation Insertion Methods =====

    async def insert_all_movie_relations(self) -> None:
        """Insert all movie-related relationships."""
        tasks = []
        for movie in self.movie_db.values():
            tasks.extend([
                self.insert_movie_cast(movie),
                self.insert_movie_directors(movie),
                self.insert_movie_genres(movie),
                self.insert_movie_awards(movie),
                self.insert_movie_year(movie),
            ])

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Movie Relations"):
            await task

    async def insert_movie_cast(self, movie: Dict) -> None:
        """Insert cast relationships for a movie."""
        if not movie.get("cast"):
            return

        query = """
        UNWIND $batch AS item
        MATCH (m:Movie {name: $movie_name})
        MATCH (p:Person {name: item.person_name})
        MERGE (p)-[:ACTED_IN {character: item.character, order: item.order, gender: item.gender}]->(m)
        """

        data = [
            {
                "person_name": cast["name"].upper(),
                "character": cast.get("character"),
                "order": cast.get("order"),
                "gender": cast.get("gender"),
            }
            for cast in movie.get("cast", [])
        ]

        if data:
            await self.execute_query(query, {"movie_name": movie["title"].upper(), "batch": data})

    async def insert_movie_directors(self, movie: Dict) -> None:
        """Insert director relationships for a movie."""
        directors = [crew for crew in movie.get("crew", []) if crew["job"] == "Director"]
        if not directors:
            return

        query = """
        UNWIND $batch AS director
        MATCH (m:Movie {name: $movie_name})
        MATCH (p:Person {name: director.name})
        MERGE (p)-[:DIRECTED]->(m)
        """

        data = [{"name": director["name"].upper()} for director in directors]

        if data:
            await self.execute_query(query, {"movie_name": movie["title"].upper(), "batch": data})

    async def insert_movie_genres(self, movie: Dict) -> None:
        """Insert genre relationships for a movie."""
        if not movie.get("genres"):
            return

        query = """
        UNWIND $batch AS genre
        MATCH (m:Movie {name: $movie_name})
        MERGE (g:Genre {name: genre.name})
        MERGE (m)-[:BELONGS_TO_GENRE]->(g)
        """

        data = [{"name": genre["name"].upper()} for genre in movie.get("genres", [])]

        if data:
            await self.execute_query(query, {"movie_name": movie["title"].upper(), "batch": data})

    async def insert_movie_awards(self, movie: Dict) -> None:
        """Insert award relationships for a movie."""
        if not movie.get("oscar_awards"):
            return

        query = """
        UNWIND $batch AS award
        MATCH (m:Movie {name: $movie_name})
        MERGE (a:Award {type: "OSCAR AWARD", name: award.category, year: award.year})
        SET a.ceremony_number = award.ceremony

        MERGE (m)-[r:NOMINATED_FOR]->(a)
        SET r.winner = award.winner, r.person = award.person, r.movie = award.movie

        FOREACH (ignored IN CASE WHEN award.winner = true THEN [1] ELSE [] END |
            MERGE (m)-[win:WON]->(a)
            SET win.winner = true, win.person = award.person, win.movie = award.movie
        )
        """

        data = [
            {
                "category": award["category"].upper(),
                "year": str(award["year_ceremony"]),
                "ceremony": award["ceremony"],
                "winner": award["winner"],
                "person": award["name"].upper(),
                "movie": award["film"].upper(),
            }
            for award in movie.get("oscar_awards", [])
        ]

        if data:
            await self.execute_query(query, {"movie_name": movie["title"].upper(), "batch": data})

    async def insert_movie_year(self, movie: Dict) -> None:
        """Insert year relationship for a movie."""
        release_date = movie.get("release_date")
        if not release_date:
            return

        release_year = int(release_date[:4])
        query = """
        MATCH (m:Movie {name: $movie_name})
        MATCH (y:Year {name: $year_name})
        MERGE (m)-[:RELEASED_IN]->(y)
        """

        await self.execute_query(query, {
            "movie_name": movie["title"].upper(),
            "year_name": str(release_year)
        })

    async def insert_all_person_relations(self) -> None:
        """Insert person-award relationships."""
        tasks = [self.insert_person_awards(person) for person in self.person_db.values()]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Person Relations"):
            await task

    async def insert_person_awards(self, person: Dict) -> None:
        """Insert award relationships for a person."""
        if not person.get("oscar_awards"):
            return

        query = """
        UNWIND $batch AS award
        MATCH (p:Person {name: $person_name})
        MERGE (a:Award {type: "OSCAR AWARD", name: award.category, year: award.year})
        SET a.ceremony_number = award.ceremony

        MERGE (p)-[r:NOMINATED_FOR]->(a)
        SET r.winner = award.winner, r.person = award.person, r.movie = award.movie

        FOREACH (ignored IN CASE WHEN award.winner = true THEN [1] ELSE [] END |
            MERGE (p)-[win:WON]->(a)
            SET win.winner = true, win.person = award.person, win.movie = award.movie
        )
        """

        data = [
            {
                "category": award["category"].upper(),
                "year": str(award["year_ceremony"]),
                "ceremony": award["ceremony"],
                "winner": award["winner"],
                "person": award["name"].upper(),
                "movie": award["film"].upper() if award["film"] else None,
            }
            for award in person.get("oscar_awards", [])
        ]

        if data:
            await self.execute_query(query, {"person_name": person["name"].upper(), "batch": data})

    async def insert_all_year_relations(self) -> None:
        """Insert year-award relationships."""
        tasks = []
        for person in self.person_db.values():
            tasks.append(self.insert_year_award_relations(person))
        for movie in self.movie_db.values():
            tasks.append(self.insert_year_award_relations(movie))

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Year Relations"):
            await task

    async def insert_year_award_relations(self, item: Dict) -> None:
        """Insert year-award relationship."""
        if not item.get("oscar_awards"):
            return

        query = """
        UNWIND $batch AS award
        MATCH (a:Award {type: "OSCAR AWARD", name: award.category, year: award.year})
        MATCH (y:Year {name: award.year})
        MERGE (a)-[:HELD_IN]->(y)
        """

        data = [
            {
                "category": award["category"].upper(),
                "year": str(award["year_ceremony"]),
            }
            for award in item.get("oscar_awards", [])
        ]

        if data:
            await self.execute_query(query, {"batch": data})

    async def sample_kg(self) -> None:
        """Sample the knowledge graph to simulate incomplete data."""
        fractions = self.config.sample_fractions

        for label, keep_fraction in fractions.items():
            query = f"""
            CALL {{
                WITH {keep_fraction} AS keep_fraction
                MATCH (n:{label})
                WHERE rand() > keep_fraction
                RETURN n
            }}
            CALL {{
                WITH n
                DETACH DELETE n
            }} IN TRANSACTIONS OF 10000 ROWS
            """
            await self.execute_query(query)
            logger.debug(f"Sampled {label} nodes (kept {keep_fraction * 100:.0f}%)")


# Export main classes (maintain backward compatibility)
KG_Preprocessor = KGPreprocessorBase
MovieKG_Preprocessor = MovieKGPreprocessor
