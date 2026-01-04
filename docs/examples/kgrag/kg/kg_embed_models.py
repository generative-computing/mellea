"""Pydantic models for KG embedding configuration and operations.

These models provide type-safe configuration for embedding generation
and storage operations.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    # API Configuration
    api_key: str = Field(default="dummy", description="API key for embedding service")
    api_base: Optional[str] = Field(default=None, description="Base URL for embedding API")
    model_name: str = Field(default="", description="Model name for embeddings")
    timeout: int = Field(default=1800, ge=1, le=3600, description="API timeout in seconds")
    rits_api_key: Optional[str] = Field(default=None, description="RITS API key if needed")

    # Embedding dimensions
    vector_dimensions: int = Field(default=768, ge=1, le=4096, description="Vector embedding dimensions")

    # Batch configuration
    batch_size: int = Field(default=8192, ge=1, le=100000, description="Batch size for embedding generation")
    concurrent_batches: int = Field(default=64, ge=1, le=256, description="Number of concurrent batches")
    storage_batch_size: int = Field(default=50000, ge=100, le=100000, description="Batch size for storing embeddings")

    # Neo4j batch retrieval
    retrieval_batch_size: int = Field(default=500000, ge=1000, le=1000000, description="Batch size for retrieving from Neo4j")

    # Similarity function
    similarity_function: str = Field(default="cosine", description="Similarity function for vector index")


class EmbeddingBatch(BaseModel):
    """A batch of items to embed."""
    ids: List[str] = Field(description="List of IDs for the items")
    texts: List[str] = Field(description="List of text descriptions to embed")

    def __len__(self) -> int:
        return len(self.ids)

    class Config:
        frozen = True


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    id: str = Field(description="ID of the embedded item")
    embedding: List[float] = Field(description="Embedding vector")

    class Config:
        frozen = True


class EntityEmbedding(BaseModel):
    """Entity with its embedding."""
    id: str
    embedding: List[float]


class RelationEmbedding(BaseModel):
    """Relation with its embedding."""
    id: str
    embedding: List[float]


class SchemaEmbedding(BaseModel):
    """Schema with its embedding."""
    name: str
    embedding: List[float]
    source_type: Optional[str] = None
    target_type: Optional[str] = None


class EmbeddingStats(BaseModel):
    """Statistics about embedding operations."""
    total_entities: int = 0
    total_relations: int = 0
    total_entity_schemas: int = 0
    total_relation_schemas: int = 0
    entities_embedded: int = 0
    relations_embedded: int = 0
    schemas_embedded: int = 0
    total_batches: int = 0
    failed_batches: int = 0

    @property
    def total_embeddings(self) -> int:
        """Calculate total embeddings across all types."""
        return self.entities_embedded + self.relations_embedded + self.schemas_embedded
