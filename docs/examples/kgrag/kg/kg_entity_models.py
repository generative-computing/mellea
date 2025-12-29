"""Pydantic models for Knowledge Graph entities.

These models provide type-safe, validated data structures for KG preprocessing,
following Mellea's pattern of using Pydantic for structured data.
"""

from datetime import date, datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


# ===== Movie Domain Models =====

class MovieCastMember(BaseModel):
    """A cast member in a movie."""
    name: str
    character: Optional[str] = None
    order: Optional[int] = None
    gender: Optional[int] = None


class MovieCrewMember(BaseModel):
    """A crew member for a movie."""
    name: str
    job: str
    department: Optional[str] = None


class MovieGenre(BaseModel):
    """A movie genre."""
    name: str
    id: Optional[int] = None


class MovieAward(BaseModel):
    """An award nomination or win."""
    category: str
    name: str
    year_ceremony: int
    ceremony: int
    winner: bool
    film: Optional[str] = None

    @field_validator("name")
    @classmethod
    def uppercase_name(cls, v: str) -> str:
        """Normalize names to uppercase."""
        return v.upper() if v else v


class Movie(BaseModel):
    """A movie entity with all its properties."""
    title: str
    original_title: Optional[str] = None
    release_date: Optional[str] = None
    original_language: Optional[str] = None
    budget: Optional[int] = None
    revenue: Optional[int] = None
    rating: Optional[float] = None
    cast: List[MovieCastMember] = Field(default_factory=list)
    crew: List[MovieCrewMember] = Field(default_factory=list)
    genres: List[MovieGenre] = Field(default_factory=list)
    oscar_awards: List[MovieAward] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def uppercase_title(cls, v: str) -> str:
        """Normalize titles to uppercase."""
        return v.upper() if v else v


class Person(BaseModel):
    """A person entity (actor, director, etc.)."""
    name: str
    birthday: Optional[str] = None
    oscar_awards: List[MovieAward] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def uppercase_name(cls, v: str) -> str:
        """Normalize names to uppercase."""
        return v.upper() if v else v

# ===== Generic KG Models =====

class KGEntity(BaseModel):
    """A generic knowledge graph entity."""
    name: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class KGRelation(BaseModel):
    """A knowledge graph relationship."""
    head: str  # Head entity name
    relation: str  # Relation type
    tail: str  # Tail entity name
    properties: Dict[str, Any] = Field(default_factory=dict)


class KGTriple(BaseModel):
    """A knowledge graph triple (for generic KG like MultiTQ, TimeQuestions)."""
    head: str
    relation: str
    tail: str
    time: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


# ===== Configuration Models =====

class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str
    max_concurrency: int = Field(default=50, ge=1, le=1000)
    max_retries: int = Field(default=5, ge=1, le=10)
    retry_delay: float = Field(default=0.5, ge=0.1, le=5.0)


class PreprocessorConfig(BaseModel):
    """Configuration for KG preprocessing."""
    neo4j: Neo4jConfig
    kg_base_directory: str = Field(default="docs/examples/kgrag/dataset")
    batch_size: int = Field(default=10000, ge=100, le=100000)
    sample_fractions: Dict[str, float] = Field(
        default_factory=lambda: {
            "Movie": 0.6,
            "Person": 0.6,
            "Award": 1.0,
            "Genre": 1.0,
            "Year": 1.0
        }
    )
