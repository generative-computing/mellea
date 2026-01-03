"""KG Updater Component using Mellea patterns."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from mellea import MelleaSession
from mellea.stdlib.base import Component
from mellea.stdlib.sampling import RejectionSamplingStrategy

from kg.kg_driver import KG_Driver
from kg.kg_rep import KGEntity, KGRelation, normalize_entity, normalize_relation, entity_to_text
from kg.kg_updater_generative import (
    extract_entities_and_relations,
    align_entity_with_kg,
    decide_entity_merge,
    align_relation_with_kg,
    decide_relation_merge,
    ExtractionResult,
    AlignmentResult,
    MergeDecision,
)
from kg.kg_requirements import VALID_JSON_REQ
from mellea.stdlib.requirement import Requirement
from utils.logger import BaseProgressLogger, DefaultProgressLogger
from utils.utils import generate_embedding


# Define requirements for KG update tasks
def has_entities_or_relations(ctx) -> bool:
    """Check if output has at least one entity or relation in flat JSON format."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        # Based on PROMPTS["extraction"], output should be flat JSON with:
        # "ent_i": [...] and "rel_j": [...]
        has_entity = any(key.startswith("ent_") for key in data.keys())
        has_relation = any(key.startswith("rel_") for key in data.keys())

        return has_entity or has_relation
    except Exception:
        return True


def has_valid_entity_format(ctx) -> bool:
    """Check if entities follow the format: ["type", "name", "description", "para_start", "para_end", {props}]."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        for key, value in data.items():
            if key.startswith("ent_"):
                # Entity should be a list with at least 5 elements: [type, name, desc, para_start, para_end, props]
                if not isinstance(value, list):
                    return False
                if len(value) < 5:
                    return False
                # First 5 elements should be strings
                if not all(isinstance(value[i], str) for i in range(5)):
                    return False
                # 6th element (if present) should be a dict (properties)
                if len(value) > 5 and not isinstance(value[5], dict):
                    return False

        return True
    except Exception:
        return False


def has_valid_relation_format(ctx) -> bool:
    """Check if relations follow the format: ["source", "relation", "target", "desc", "para_start", "para_end", {props}]."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        for key, value in data.items():
            if key.startswith("rel_"):
                # Relation should be a list with at least 6 elements
                if not isinstance(value, list):
                    return False
                if len(value) < 6:
                    return False
                # First 6 elements should be strings
                if not all(isinstance(value[i], str) for i in range(6)):
                    return False
                # 7th element (if present) should be a dict (properties)
                if len(value) > 6 and not isinstance(value[6], dict):
                    return False

        return True
    except Exception:
        return False


EXTRACTION_REQS = [
    VALID_JSON_REQ,
    Requirement(
        description="Must extract at least one entity (ent_i) or relation (rel_j)",
        validation_fn=has_entities_or_relations
    ),
    Requirement(
        description="Entities must follow format: ['type', 'name', 'description', 'para_start', 'para_end', {props}]",
        validation_fn=has_valid_entity_format
    ),
    Requirement(
        description="Relations must follow format: ['source', 'relation', 'target', 'desc', 'para_start', 'para_end', {props}]",
        validation_fn=has_valid_relation_format
    )
]

def has_required_alignment_fields(ctx) -> bool:
    """Check if alignment output has required fields from PROMPTS["align_entity"]."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        # Based on PROMPTS["align_entity"], output should have:
        # {"id": <id>, "aligned_type": "...", "reason": "...", "matched_entity": "..."}
        required_fields = ["id", "aligned_type", "reason", "matched_entity"]

        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return False

        for item in data:
            if not isinstance(item, dict):
                return False
            for field in required_fields:
                if field not in item:
                    return False

        return True
    except Exception:
        return False


def has_valid_matched_entity(ctx) -> bool:
    """Check if matched_entity is either a valid entity reference or empty string."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return True  # Let other validators catch this

        for item in data:
            if not isinstance(item, dict):
                continue
            matched_entity = item.get("matched_entity", "")
            # matched_entity should be either empty string or start with "ent_"
            if matched_entity and not (isinstance(matched_entity, str) and
                                      (matched_entity == "" or matched_entity.startswith("ent_"))):
                return False

        return True
    except Exception:
        return False


ALIGNMENT_REQS = [
    VALID_JSON_REQ,
    Requirement(
        description="Must have id, aligned_type, reason, and matched_entity fields for each alignment",
        validation_fn=has_required_alignment_fields
    ),
    Requirement(
        description="matched_entity must be empty string or valid entity reference (ent_i)",
        validation_fn=has_valid_matched_entity
    )
]


def has_required_merge_fields(ctx) -> bool:
    """Check if merge output has required fields from PROMPTS["merge_entity"] and PROMPTS["merge_relation"]."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        # Based on merge prompts, output should be a list of dicts with:
        # {"id": <id>, "desc": "...", "props": {...}}
        if not isinstance(data, list):
            return False

        for item in data:
            if not isinstance(item, dict):
                return False
            # Must have id, desc, and props fields
            if "id" not in item or "desc" not in item or "props" not in item:
                return False

        return True
    except Exception:
        return False


def has_valid_merge_properties(ctx) -> bool:
    """Check if merged properties follow the format: {"key": ["val", "context"], ...}."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        if not isinstance(data, list):
            return True  # Let other validators catch this

        for item in data:
            if not isinstance(item, dict):
                continue

            props = item.get("props", {})
            if not isinstance(props, dict):
                return False

            # Each property value should be a list with 2 elements: [value, context]
            for key, value in props.items():
                if not isinstance(value, list):
                    return False
                if len(value) != 2:
                    return False
                # Both elements should be strings
                if not all(isinstance(v, str) for v in value):
                    return False

        return True
    except Exception:
        return False


MERGE_REQS = [
    VALID_JSON_REQ,
    Requirement(
        description="Must have id, desc, and props fields for each merged item",
        validation_fn=has_required_merge_fields
    ),
    Requirement(
        description="Properties must follow format: {\"key\": [\"val\", \"context\"], ...}",
        validation_fn=has_valid_merge_properties
    )
]


def has_required_relation_alignment_fields(ctx) -> bool:
    """Check if relation alignment output has required fields from PROMPTS["align_relation"]."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        # Based on PROMPTS["align_relation"], output should have:
        # {"id": <id>, "aligned_name": "...", "reason": "...", "matched_relation": "..."}
        required_fields = ["id", "aligned_name", "reason", "matched_relation"]

        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return False

        for item in data:
            if not isinstance(item, dict):
                return False
            for field in required_fields:
                if field not in item:
                    return False

        return True
    except Exception:
        return False


def has_valid_matched_relation(ctx) -> bool:
    """Check if matched_relation is either a valid relation reference or empty string."""
    try:
        output = ctx.last_assistant_message.as_str()
        import json
        data = json.loads(output)

        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return True  # Let other validators catch this

        for item in data:
            if not isinstance(item, dict):
                continue
            matched_relation = item.get("matched_relation", "")
            # matched_relation should be either empty string or start with "rel_"
            if matched_relation and not (isinstance(matched_relation, str) and
                                        (matched_relation == "" or matched_relation.startswith("rel_"))):
                return False

        return True
    except Exception:
        return False


RELATION_ALIGNMENT_REQS = [
    VALID_JSON_REQ,
    Requirement(
        description="Must have id, aligned_name, reason, and matched_relation fields for each alignment",
        validation_fn=has_required_relation_alignment_fields
    ),
    Requirement(
        description="matched_relation must be empty string or valid relation reference (rel_i)",
        validation_fn=has_valid_matched_relation
    )
]


class KGUpdaterComponent(Component):
    """Knowledge Graph Updater using Mellea patterns.

    This component extracts entities and relations from documents and updates
    the knowledge graph using @generative functions, Requirements, and
    RejectionSamplingStrategy for robustness.
    """

    def __init__(
        self,
        session: MelleaSession,
        emb_session: Any,
        kg_driver: KG_Driver,
        domain: str = "movie",
        config: Optional[Dict] = None,
        logger: Optional[BaseProgressLogger] = None,
        **kwargs,
    ):
        """Initialize KG Updater component.

        Args:
            session: Mellea session for LLM calls
            emb_session: Session for embeddings
            kg_driver: KG database driver
            domain: Knowledge domain
            config: Configuration dict
            logger: Logger for progress tracking
        """
        super().__init__()
        self.session = session
        self.emb_session = emb_session
        self.kg_driver = kg_driver
        self.domain = domain
        self.logger = logger or DefaultProgressLogger()

        # Default config
        self.config = {
            "align_entity": True,
            "merge_entity": True,
            "align_relation": True,
            "merge_relation": True,
            "extraction_loop_budget": 3,
            "alignment_loop_budget": 2,
            "align_topk": 10,
            "align_entity_batch_size": 10,
            "merge_entity_batch_size": 10,
            "align_relation_batch_size": 10,
            "merge_relation_batch_size": 10,
        }
        if config:
            self.config.update(config)

        self.logger.info(f"KGUpdaterComponent initialized with config: {self.config}")

    async def extract_from_context(
        self,
        context: str,
        reference: str,
        hints: str = ""
    ) -> ExtractionResult:
        """Extract entities and relations from context with validation.

        Uses @generative function with Requirements and RejectionSampling.

        Args:
            context: Document text
            reference: Reference/source information
            hints: Domain-specific hints

        Returns:
            ExtractionResult with entities and relations
        """
        self.logger.info("Extracting entities and relations from context")

        # Use rejection sampling for robustness
        strategy = RejectionSamplingStrategy(
            loop_budget=self.config["extraction_loop_budget"]
        )

        try:
            # Reset context before each generative call
            self.session.reset()

            result, ctx = await extract_entities_and_relations(
                self.session.ctx,
                self.session.backend,
                requirements=EXTRACTION_REQS,
                strategy=strategy,
                doc_context=context,
                domain=self.domain,
                hints=hints or f"Extract knowledge relevant to {self.domain}",
                reference=reference,
            )

            self.logger.info(
                f"Extracted {len(result.entities)} entities and "
                f"{len(result.relations)} relations"
            )

            return result

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            # Return empty result on failure
            return ExtractionResult(entities=[], relations=[], reasoning="Extraction failed")

    async def align_entity(
        self,
        entity_name: str,
        entity_type: str,
        entity_desc: str,
        context: str,
        candidate_entities: List[KGEntity],
        top_k: Optional[int] = None
    ) -> Optional[str]:
        """Align extracted entity with existing KG entities.

        Args:
            entity_name: Name of extracted entity
            entity_type: Type of extracted entity
            entity_desc: Description of extracted entity
            context: Original document text for context-aware alignment
            candidate_entities: List of candidate entities from KG
            top_k: Number of candidates to consider (default from config)

        Returns:
            ID of aligned entity, or None if no match
        """
        if not candidate_entities:
            return None

        # Use config default if not specified
        top_k = top_k or self.config.get("align_topk", 10)

        # Format candidates for LLM with configurable limit
        candidates_str = "\n\n".join([
            f"ID: {e.id}\nName: {e.name}\nType: {e.type}\nDescription: {e.description[:200]}"
            for e in candidate_entities[:top_k]
        ])

        self.logger.debug(f"Aligning entity '{entity_name}' with {len(candidate_entities[:top_k])} candidates")

        strategy = RejectionSamplingStrategy(
            loop_budget=self.config.get("alignment_loop_budget", 3)
        )

        try:
            # Reset context before each generative call
            self.session.reset()

            result, ctx = await align_entity_with_kg(
                self.session.ctx,
                self.session.backend,
                requirements=ALIGNMENT_REQS,
                strategy=strategy,
                extracted_entity_name=entity_name,
                extracted_entity_type=entity_type,
                extracted_entity_desc=entity_desc,
                candidate_entities=candidates_str,
                domain=self.domain,
                context=context[:2000] if context else "",  # Limit context to avoid token overflow
            )

            if result.confidence > 0.7 and result.aligned_entity_id:
                self.logger.info(
                    f"Aligned '{entity_name}' to '{result.aligned_entity_id}' "
                    f"(confidence: {result.confidence:.2f})"
                )
                return result.aligned_entity_id
            else:
                self.logger.debug(f"No strong alignment for '{entity_name}'")
                return None

        except Exception as e:
            self.logger.error(f"Alignment failed for '{entity_name}': {e}")
            return None

    async def merge_entities(
        self,
        entity1: KGEntity,
        entity2: KGEntity,
        context: str = ""
    ) -> Optional[KGEntity]:
        """Merge two entities using PROMPTS["merge_entity"] format.

        Args:
            entity1: Extracted entity from text document
            entity2: Existing entity from KG
            context: Original document text

        Returns:
            Merged entity with updated description and properties
        """
        # Format entity pair according to PROMPTS["merge_entity"]
        # Format: "idx: [(Type: Name, desc: "...", props: {...}), (Type: Name, desc: "...", props: {...})]"
        entity_pair = f"1: [({entity1.type}: {entity1.name}, desc: \"{entity1.description}\", props: {entity1.properties}), " \
                     f"({entity2.type}: {entity2.name}, desc: \"{entity2.description}\", props: {entity2.properties})]"

        strategy = RejectionSamplingStrategy(
            loop_budget=self.config.get("merge_loop_budget", 3)
        )

        try:
            # Reset context before each generative call
            self.session.reset()

            result, ctx = await decide_entity_merge(
                self.session.ctx,
                self.session.backend,
                requirements=MERGE_REQS,
                strategy=strategy,
                entity_pair=entity_pair,
                context=context[:2000] if context else "",  # Limit context size
                domain=self.domain,
            )

            # Parse the result - expecting format: [{"id": 1, "desc": "...", "props": {"key": ["val", "context"], ...}}]
            if result and len(result) > 0:
                merge_result = result[0] if isinstance(result, list) else result
                self.logger.info(f"Merged entities '{entity1.name}' and '{entity2.name}'")

                # Convert properties format from ["val", "context"] to standard format
                merged_properties = {}
                if "props" in merge_result and merge_result["props"]:
                    for key, val_list in merge_result["props"].items():
                        if isinstance(val_list, list) and len(val_list) >= 1:
                            # Take the value, ignore the context for now
                            merged_properties[key] = val_list[0]

                # Create merged entity
                merged = KGEntity(
                    id=entity2.id,  # Keep KG entity's ID
                    name=entity2.name,  # Keep KG entity's name
                    type=entity2.type,  # Keep KG entity's type
                    description=merge_result.get("desc", entity2.description),
                    properties=merged_properties
                )
                return merged
            else:
                return None

        except Exception as e:
            self.logger.error(f"Merge decision failed: {e}")
            return None

    async def update_kg_from_document(
        self,
        doc_id: str,
        context: str,
        reference: str,
        created_at: datetime,
    ) -> Dict[str, Any]:
        """Update KG from a single document.

        Main entry point for processing a document with Mellea patterns.

        Args:
            doc_id: Document identifier
            context: Document text
            reference: Reference/source
            created_at: Timestamp

        Returns:
            Dictionary with update statistics
        """
        self.logger.info(f"Processing document {doc_id}")

        stats = {
            "doc_id": doc_id,
            "entities_extracted": 0,
            "entities_aligned": 0,
            "entities_new": 0,
            "relations_extracted": 0,
            "relations_aligned": 0,
            "relations_new": 0,
        }

        try:
            # Step 1: Extract entities and relations
            extraction = await self.extract_from_context(
                context=context,
                reference=reference,
                hints=f"Focus on {self.domain} domain knowledge"
            )

            stats["entities_extracted"] = len(extraction.entities)
            stats["relations_extracted"] = len(extraction.relations)

            # Step 2: Process entities
            for extracted_entity in extraction.entities:
                try:
                    # Defensive check: ensure extracted_entity is an object with required attributes
                    if not hasattr(extracted_entity, 'name') or not hasattr(extracted_entity, 'type'):
                        self.logger.warning(f"Skipping malformed entity: {type(extracted_entity)}")
                        continue

                    # Normalize name
                    norm_name = normalize_entity(extracted_entity.name)

                    # Search for similar entities in KG if alignment is enabled
                    aligned_entity_id = None
                    if self.config.get("align_entity", False):
                        # Get candidate entities from KG (vector search + exact match)
                        candidate_entities = []
                        top_k = self.config.get("align_topk", 10)

                        # Try exact/fuzzy match first
                        exact_matches = self.kg_driver.get_entities(
                            type=extracted_entity.type,
                            name=norm_name,
                            top_k=top_k // 2,
                            fuzzy=True
                        )
                        if exact_matches:
                            candidate_entities.extend(exact_matches)

                        # Generate embedding and do vector search for similar entities
                        try:
                            entity_text = entity_to_text(
                                KGEntity(
                                    id="",
                                    name=norm_name,
                                    type=extracted_entity.type,
                                    description=extracted_entity.description,
                                    properties=extracted_entity.properties
                                ),
                                include_des=False
                            )
                            entity_embeddings = await generate_embedding(
                                self.emb_session,
                                [entity_text],
                                logger=self.logger
                            )

                            if entity_embeddings and len(entity_embeddings) > 0:
                                similar_matches = self.kg_driver.get_entities(
                                    embedding=entity_embeddings[0],
                                    top_k=top_k - len(candidate_entities),
                                    return_score=True
                                )
                                # Add similar matches that aren't already in candidates
                                for relevant_entity in similar_matches:
                                    if relevant_entity.entity not in candidate_entities:
                                        candidate_entities.append(relevant_entity.entity)
                        except Exception as e:
                            self.logger.warning(f"Vector search failed for '{norm_name}': {e}")

                        # Align entity with KG if candidates found
                        if candidate_entities:
                            aligned_entity_id = await self.align_entity(
                                entity_name=norm_name,
                                entity_type=extracted_entity.type,
                                entity_desc=extracted_entity.description,
                                context=context,
                                candidate_entities=candidate_entities,
                                top_k=top_k
                            )

                            if aligned_entity_id:
                                stats["entities_aligned"] += 1
                                self.logger.debug(f"Entity '{norm_name}' aligned to {aligned_entity_id}")

                    # Create or update entity in KG
                    entity = KGEntity(
                        id=aligned_entity_id or "",  # Use aligned ID or empty for new entity
                        name=norm_name,
                        type=extracted_entity.type,
                        description=extracted_entity.description,
                        properties=extracted_entity.properties,
                        created_at=created_at,
                        ref=reference
                    )

                    # Upsert to KG
                    # await self.kg_driver.upsert_entity(entity)

                    if not aligned_entity_id:
                        stats["entities_new"] += 1

                except Exception as e:
                    self.logger.error(f"Failed to process entity: {e}")
                    continue

            # Step 3: Process relations
            for extracted_relation in extraction.relations:
                try:
                    # Defensive check: ensure extracted_relation is an object with required attributes
                    if not hasattr(extracted_relation, 'source_entity') or not hasattr(extracted_relation, 'target_entity'):
                        self.logger.warning(f"Skipping malformed relation: {type(extracted_relation)}")
                        continue

                    # Similar process for relations
                    stats["relations_new"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to process relation: {e}")
                    continue

            self.logger.info(
                f"Document {doc_id} processed: "
                f"{stats['entities_new']} new entities, "
                f"{stats['relations_new']} new relations"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to process document {doc_id}: {e}")
            return stats


    async def batch_update(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update KG from multiple documents concurrently.

        Args:
            documents: List of document dicts with 'id', 'context', 'reference'

        Returns:
            List of update statistics per document
        """
        self.logger.info(f"Batch updating KG with {len(documents)} documents")

        tasks = [
            self.update_kg_from_document(
                doc_id=doc["id"],
                context=doc["context"],
                reference=doc.get("reference", ""),
                created_at=datetime.now()
            )
            for doc in documents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        stats_list = [r for r in results if isinstance(r, dict)]

        total_entities = sum(s["entities_new"] for s in stats_list)
        total_relations = sum(s["relations_new"] for s in stats_list)

        self.logger.info(
            f"Batch update complete: {total_entities} entities, "
            f"{total_relations} relations added"
        )

        return stats_list
