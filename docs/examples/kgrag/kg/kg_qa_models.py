"""Pydantic models for KG QA configuration.

These models provide type-safe configuration for question answering operations.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class QAConfig(BaseModel):
    """Configuration for QA operations."""

    # Worker configuration
    num_workers: int = Field(default=128, ge=1, le=512, description="Number of concurrent workers")
    queue_size: int = Field(default=128, ge=1, le=1024, description="Queue size for data loading")

    # Dataset configuration
    split: int = Field(default=0, ge=0, description="Dataset split index")

    # Evaluation configuration
    eval_batch_size: int = Field(default=64, ge=1, le=256, description="Batch size for evaluation")
    eval_method: str = Field(default="llama", description="Evaluation method to use")


class QASessionConfig(BaseModel):
    """Configuration for QA API sessions."""

    # Main LLM configuration
    api_base: str = Field(default="http://localhost:7878/v1", description="API base URL")
    api_key: str = Field(default="dummy", description="API key")
    model_name: str = Field(default="", description="Model name")
    timeout: int = Field(default=1800, ge=1, le=3600, description="Timeout in seconds")
    rits_api_key: Optional[str] = Field(default=None, description="RITS API key if needed")

    # Evaluation LLM configuration
    eval_api_base: Optional[str] = Field(default=None)
    eval_api_key: Optional[str] = Field(default=None)
    eval_model_name: Optional[str] = Field(default=None)
    eval_timeout: Optional[int] = Field(default=None, ge=1, le=3600)

    # Embedding configuration
    emb_api_base: Optional[str] = Field(default=None)
    emb_api_key: Optional[str] = Field(default=None)
    emb_model_name: Optional[str] = Field(default=None)
    emb_timeout: Optional[int] = Field(default=None, ge=1, le=3600)


class QADatasetConfig(BaseModel):
    """Configuration for QA dataset processing."""

    dataset_path: str = Field(description="Path to dataset file")
    domain: str = Field(default="movie", description="Knowledge domain")

    # Output paths
    result_path: str = Field(description="Path for results JSON file")
    progress_path: str = Field(description="Path for progress logging")

    # File naming
    prefix: Optional[str] = Field(default=None, description="Prefix for output files")
    postfix: Optional[str] = Field(default=None, description="Postfix for output files")

    # Cleanup
    keep_progress: bool = Field(default=False, description="Keep progress file after completion")


class KGModelConfig(BaseModel):
    """Configuration for KG model parameters.

    These are model-specific configuration options that can be overridden.
    """

    # Allow arbitrary fields for model-specific configs
    class Config:
        extra = "allow"

    # Common fields with defaults
    route: Optional[int] = Field(default=None, description="Number of routes for question decomposition")
    width: Optional[int] = Field(default=None, description="Width parameter for search")
    depth: Optional[int] = Field(default=None, description="Depth parameter for search")
