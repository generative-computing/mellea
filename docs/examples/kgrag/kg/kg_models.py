"""Pydantic models for KG-RAG structured outputs."""
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class QuestionRoutes(BaseModel):
    """Routes for breaking down a complex question into sub-objectives."""
    reason: str = Field(description="Reasoning for the route ordering")
    routes: List[List[str]] = Field(description="List of solving routes, each containing sub-objectives")


class TopicEntities(BaseModel):
    """Extracted topic entities from a query."""
    entities: List[str] = Field(description="List of extracted entity names")


class RelevantEntities(BaseModel):
    """Relevant entities with their scores."""
    reason: str = Field(description="Reasoning for entity relevance")
    relevant_entities: Dict[str, float] = Field(
        description="Mapping of entity index (e.g., 'ent_0') to relevance score"
    )


class RelevantRelations(BaseModel):
    """Relevant relations with their scores."""
    reason: str = Field(description="Reasoning for relation relevance")
    relevant_relations: Dict[str, float] = Field(
        description="Mapping of relation index (e.g., 'rel_0') to relevance score"
    )


class EvaluationResult(BaseModel):
    """Evaluation result for whether knowledge is sufficient to answer."""
    sufficient: str = Field(description="'Yes' or 'No' indicating if knowledge is sufficient")
    reason: str = Field(description="Reasoning for the sufficiency judgment")
    answer: str = Field(description="The answer if sufficient, 'I don't know' otherwise")


class ValidationResult(BaseModel):
    """Validation result for consensus among multiple routes."""
    judgement: str = Field(description="'Yes' or 'No' for whether consensus is reached")
    final_answer: str = Field(description="The final answer with explanation")


class DirectAnswer(BaseModel):
    """Direct answer without knowledge graph."""
    sufficient: str = Field(description="'Yes' or 'No' indicating if LLM knowledge is sufficient")
    reason: str = Field(description="Reasoning for the answer")
    answer: str = Field(description="The answer or 'I don't know'")
