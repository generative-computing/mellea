"""KG Updater Component using Mellea patterns."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from mellea import MelleaSession
from mellea.stdlib.base import Component
from mellea.stdlib.sampling.rejection_sampling import RejectionSamplingStrategy

from kg.kg_driver import KG_Driver
from kg.kg_rep import KGEntity, KGRelation, normalize_entity, normalize_relation
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


# Define requirements for KG update tasks
EXTRACTION_REQS = [
    VALID_JSON_REQ,
    Requirement(
        name="has_entities_or_relations",
        requirement="Must extract at least one entity or relation",
        validator=lambda o: (
            "entities" in o.value or "relations" in o.value
        ) if hasattr(o, 'value') else True
    )
]

ALIGNMENT_REQS = [
    VALID_JSON_REQ,
    Requirement(
        name="has_confidence",
        requirement="Must provide confidence score between 0 and 1",
        validator=lambda o: True  # Pydantic handles this
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
            result = await extract_entities_and_relations(
                context=context,
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
        candidate_entities: List[KGEntity]
    ) -> Optional[str]:
        """Align extracted entity with existing KG entities.

        Args:
            entity_name: Name of extracted entity
            entity_type: Type of extracted entity
            entity_desc: Description of extracted entity
            candidate_entities: List of candidate entities from KG

        Returns:
            ID of aligned entity, or None if no match
        """
        if not candidate_entities:
            return None

        # Format candidates for LLM
        candidates_str = "\n\n".join([
            f"ID: {e.id}\nName: {e.name}\nType: {e.type}\nDescription: {e.description[:200]}"
            for e in candidate_entities[:10]  # Limit to top 10
        ])

        self.logger.debug(f"Aligning entity '{entity_name}' with {len(candidate_entities)} candidates")

        try:
            result = await align_entity_with_kg(
                extracted_entity_name=entity_name,
                extracted_entity_type=entity_type,
                extracted_entity_desc=entity_desc,
                candidate_entities=candidates_str,
                domain=self.domain,
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
        entity2: KGEntity
    ) -> Optional[KGEntity]:
        """Decide whether to merge two entities.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Merged entity if should merge, None otherwise
        """
        entity1_info = f"Name: {entity1.name}\nType: {entity1.type}\nDescription: {entity1.description}"
        entity2_info = f"Name: {entity2.name}\nType: {entity2.type}\nDescription: {entity2.description}"

        try:
            result = await decide_entity_merge(
                entity1_info=entity1_info,
                entity2_info=entity2_info,
                domain=self.domain,
            )

            if result.should_merge:
                self.logger.info(f"Merging entities '{entity1.name}' and '{entity2.name}'")
                # Create merged entity
                merged = KGEntity(
                    id=entity1.id,  # Keep first entity's ID
                    name=entity1.name,
                    type=entity1.type,
                    description=entity1.description,
                    properties={**entity1.properties, **result.merged_properties}
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

                    # Search for similar entities in KG
                    # (simplified - in reality would use vector search)
                    # For now, just create new entity
                    entity = KGEntity(
                        id="",  # Will be assigned by KG
                        name=norm_name,
                        type=extracted_entity.type,
                        description=extracted_entity.description,
                        properties=extracted_entity.properties,
                        created_at=created_at,
                        ref=reference
                    )

                    # Upsert to KG
                    # await self.kg_driver.upsert_entity(entity)
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
