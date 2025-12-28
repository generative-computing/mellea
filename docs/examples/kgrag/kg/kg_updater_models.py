"""Pydantic models for KG updater configuration.

These models provide type-safe configuration for KG update operations.
"""

from typing import Optional
from pydantic import BaseModel, Field


class UpdaterConfig(BaseModel):
    """Configuration for KG updater operations."""

    # Worker configuration
    num_workers: int = Field(default=64, ge=1, le=256, description="Number of concurrent workers")
    queue_size: int = Field(default=64, ge=1, le=512, description="Queue size for data loading")

    # Alignment and merging options
    align_entity: bool = Field(default=True, description="Whether to align entities")
    align_relation: bool = Field(default=True, description="Whether to align relations")
    merge_entity: bool = Field(default=True, description="Whether to merge entities")
    merge_relation: bool = Field(default=True, description="Whether to merge relations")
    self_reflection: bool = Field(default=False, description="Enable self-reflection step")

    # Batch sizes
    align_topk: int = Field(default=10, ge=1, le=100, description="Top K for alignment")
    align_entity_batch_size: int = Field(default=10, ge=1, le=100)
    merge_entity_batch_size: int = Field(default=10, ge=1, le=100)
    align_relation_batch_size: int = Field(default=10, ge=1, le=100)
    merge_relation_batch_size: int = Field(default=10, ge=1, le=100)

    # Retry and token limits
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    max_generation_tokens: int = Field(default=20000, ge=1000, le=100000)

    # Chunking
    max_chunk: int = Field(default=60000, ge=10000, le=100000, description="Maximum chunk size")
    min_chunk: int = Field(default=10000, ge=1000, le=50000, description="Minimum chunk size")

    # Timing
    stages: int = Field(default=2, ge=1, le=10, description="Number of processing stages")
    expected_time: int = Field(default=600, ge=60, le=3600, description="Expected time in seconds")


class SessionConfig(BaseModel):
    """Configuration for API sessions."""

    # Main LLM configuration
    api_base: str = Field(default="http://localhost:7878/v1", description="API base URL")
    api_key: str = Field(default="dummy", description="API key")
    model_name: str = Field(default="", description="Model name")
    timeout: int = Field(default=1800, ge=1, le=3600, description="Timeout in seconds")
    rits_api_key: Optional[str] = Field(default=None, description="RITS API key if needed")

    # Evaluation LLM configuration (optional, falls back to main if not specified)
    eval_api_base: Optional[str] = Field(default=None)
    eval_api_key: Optional[str] = Field(default=None)
    eval_model_name: Optional[str] = Field(default=None)
    eval_timeout: Optional[int] = Field(default=None, ge=1, le=3600)

    # Embedding configuration (optional, falls back to main if not specified)
    emb_api_base: Optional[str] = Field(default=None)
    emb_api_key: Optional[str] = Field(default=None)
    emb_model_name: Optional[str] = Field(default=None)
    emb_timeout: Optional[int] = Field(default=None, ge=1, le=3600)


class DatasetConfig(BaseModel):
    """Configuration for dataset processing."""

    dataset_path: str = Field(description="Path to dataset file")
    domain: str = Field(default="movie", description="Knowledge domain")
    progress_path: str = Field(
        default="results/update_movie_kg_progress.json",
        description="Path for progress logging"
    )
