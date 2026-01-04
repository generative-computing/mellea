"""Generative functions for KG Update using Mellea's @generative decorator."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from mellea.stdlib.genslot import generative


# Pydantic models for structured outputs
class ExtractedEntity(BaseModel):
    """Extracted entity from document."""
    type: str = Field(description="Entity type (e.g., Person, Movie, Organization)")
    name: str = Field(description="Entity name")
    description: str = Field(description="Brief description of the entity")
    paragraph_start: str = Field(description="First 5-30 chars of supporting paragraph")
    paragraph_end: str = Field(description="Last 5-30 chars of supporting paragraph")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class ExtractedRelation(BaseModel):
    """Extracted relation between entities."""
    source_entity: str = Field(description="Source entity name")
    relation_type: str = Field(description="Relation type (e.g., acted_in, directed)")
    target_entity: str = Field(description="Target entity name")
    description: str = Field(description="Description of the relation")
    paragraph_start: str = Field(description="First 5-30 chars of supporting paragraph")
    paragraph_end: str = Field(description="Last 5-30 chars of supporting paragraph")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class ExtractionResult(BaseModel):
    """Result of entity and relation extraction."""
    entities: List[ExtractedEntity] = Field(description="List of extracted entities")
    relations: List[ExtractedRelation] = Field(description="List of extracted relations")
    reasoning: str = Field(description="Reasoning for the extractions")


class AlignmentResult(BaseModel):
    """Result of entity alignment with existing KG."""
    aligned_entity_id: Optional[str] = Field(description="ID of matched entity in KG, or None")
    confidence: float = Field(description="Confidence score 0-1 for the alignment")
    reasoning: str = Field(description="Reasoning for the alignment decision")


class MergeDecision(BaseModel):
    """Decision on whether to merge entities."""
    should_merge: bool = Field(description="Whether entities should be merged")
    reasoning: str = Field(description="Reasoning for the merge decision")
    merged_properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Properties of merged entity if merging"
    )


@generative
async def extract_entities_and_relations(
    doc_context: str,
    domain: str,
    hints: str,
    reference: str
) -> ExtractionResult:
    """Extract entities and relations from a document context.

    You are a knowledge graph construction assistant for the {domain} domain.
    Your task is to extract entities and their relations from the given context.

    Instructions:
    1. Identify all significant entities (people, places, objects, concepts)
    2. For each entity, provide:
       - Type (e.g., Person, Movie, Organization, Location)
       - Name (canonical form)
       - Description (1-2 sentences)
       - Paragraph anchors (first/last 5-30 chars of supporting text)
       - Properties (key-value pairs like birth_date, genre, etc.)

    3. Identify relations between entities:
       - Source entity name
       - Relation type (e.g., acted_in, directed, produced)
       - Target entity name
       - Description
       - Paragraph anchors
       - Properties

    4. Use informative entity types and relation types
    5. Ensure paragraph anchors are exact substrings from the context
    6. Extract properties relevant to the domain

    Domain-specific Hints: {hints}

    Context:
    {doc_context}

    Reference: {reference}

    Return your extractions as:
    {{
        "entities": [
            {{
                "type": "...",
                "name": "...",
                "description": "...",
                "paragraph_start": "...",
                "paragraph_end": "...",
                "properties": {{...}}
            }}
        ],
        "relations": [
            {{
                "source_entity": "...",
                "relation_type": "...",
                "target_entity": "...",
                "description": "...",
                "paragraph_start": "...",
                "paragraph_end": "...",
                "properties": {{...}}
            }}
        ],
        "reasoning": "..."
    }}
    """
    pass


@generative
async def align_entity_with_kg(
    extracted_entity_name: str,
    extracted_entity_type: str,
    extracted_entity_desc: str,
    candidate_entities: str,
    domain: str
) -> AlignmentResult:
    """Align extracted entity with existing entities in the knowledge graph.

    You are given an extracted entity and a list of candidate entities from the KG.
    Determine if the extracted entity matches any existing entity.

    Task:
    - Compare the extracted entity with each candidate
    - Consider name similarity, type compatibility, and description overlap
    - Return the ID of the best match if confidence > 0.7, otherwise None
    - Provide confidence score (0-1) and reasoning

    Domain: {domain}

    Extracted Entity:
    - Name: {extracted_entity_name}
    - Type: {extracted_entity_type}
    - Description: {extracted_entity_desc}

    Candidate Entities from KG:
    {candidate_entities}

    Return your alignment decision as:
    {{
        "aligned_entity_id": "entity_id_123 or null",
        "confidence": 0.0-1.0,
        "reasoning": "..."
    }}
    """
    pass


@generative
async def decide_entity_merge(
    entity1_info: str,
    entity2_info: str,
    domain: str
) -> MergeDecision:
    """Decide whether two entities should be merged.

    You are given two entities that may refer to the same real-world object.
    Decide if they should be merged into a single entity.

    Considerations:
    - Do they refer to the same real-world entity?
    - Is there conflicting information?
    - How should properties be merged?

    Domain: {domain}

    Entity 1:
    {entity1_info}

    Entity 2:
    {entity2_info}

    Return your merge decision as:
    {{
        "should_merge": true/false,
        "reasoning": "...",
        "merged_properties": {{...}}  // only if should_merge=true
    }}
    """
    pass


@generative
async def align_relation_with_kg(
    extracted_relation: str,
    candidate_relations: str,
    domain: str
) -> AlignmentResult:
    """Align extracted relation with existing relations in the knowledge graph.

    Similar to entity alignment, but for relations between entities.

    Domain: {domain}

    Extracted Relation:
    {extracted_relation}

    Candidate Relations from KG:
    {candidate_relations}

    Return your alignment decision as:
    {{
        "aligned_entity_id": "relation_id_456 or null",
        "confidence": 0.0-1.0,
        "reasoning": "..."
    }}
    """
    pass


@generative
async def decide_relation_merge(
    relation1_info: str,
    relation2_info: str,
    domain: str
) -> MergeDecision:
    """Decide whether two relations should be merged.

    Domain: {domain}

    Relation 1:
    {relation1_info}

    Relation 2:
    {relation2_info}

    Return your merge decision as:
    {{
        "should_merge": true/false,
        "reasoning": "...",
        "merged_properties": {{...}}
    }}
    """
    pass
