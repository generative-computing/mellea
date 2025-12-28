"""Refactored Knowledge Graph Embedder following Mellea patterns.

This module provides a cleaner, more maintainable implementation of KG embedding
with:
- Pydantic models for configuration
- Better separation of concerns
- Type safety throughout
- No use of eval() or other unsafe operations
- Modern async patterns
"""

import asyncio
import os
import textwrap
from typing import List, Optional, Union, Any, Dict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from kg.kg_embed_models import (
    EmbeddingConfig,
    EmbeddingStats,
    EntityEmbedding,
    RelationEmbedding,
    SchemaEmbedding,
)
from kg.kg_driver import kg_driver
from kg.kg_rep import (
    KGEntity,
    KGRelation,
    PROP_EMBEDDING,
    TYPE_EMBEDDABLE,
    entity_to_text,
    relation_to_text,
    entity_schema_to_text,
    relation_schema_to_text,
)
from utils.utils import generate_embedding
from utils.logger import logger

# Load environment variables
load_dotenv()


class KGEmbedderBase:
    """Base class for knowledge graph embedding operations.

    Provides common functionality for generating and storing embeddings
    for entities, relations, and schemas in the knowledge graph.
    """

    def __init__(
        self,
        emb_session: Any,
        config: Optional[EmbeddingConfig] = None
    ):
        """Initialize the embedder.

        Args:
            emb_session: Embedding session (OpenAI client or SentenceTransformer)
            config: Embedding configuration (loaded from env if None)
        """
        self.emb_session = emb_session
        self.config = config or self._load_config_from_env()
        self.stats = EmbeddingStats()

    @staticmethod
    def _load_config_from_env() -> EmbeddingConfig:
        """Load configuration from environment variables."""
        return EmbeddingConfig(
            api_key=os.getenv("API_KEY", "dummy"),
            api_base=os.getenv("EMB_API_BASE"),
            model_name=os.getenv("EMB_MODEL_NAME", ""),
            timeout=int(os.getenv("EMB_TIME_OUT", "1800")),
            rits_api_key=os.getenv("RITS_API_KEY"),
            vector_dimensions=int(os.getenv("VECTOR_DIMENSIONS", "768")),
            batch_size=int(os.getenv("EMB_BATCH_SIZE", "8192")),
            concurrent_batches=int(os.getenv("EMB_CONCURRENT_BATCHES", "64")),
            storage_batch_size=int(os.getenv("EMB_STORAGE_BATCH_SIZE", "50000")),
        )

    async def generate_embeddings_batched(
        self,
        texts: List[str],
        desc: str = "Embedding"
    ) -> np.ndarray:
        """Generate embeddings for a list of texts in batches.

        Args:
            texts: List of text descriptions to embed
            desc: Description for progress bar

        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])

        async def embed_batch(start_idx: int) -> List[List[float]]:
            """Embed a single batch."""
            end_idx = start_idx + self.config.batch_size
            batch = texts[start_idx:end_idx]
            return await generate_embedding(self.emb_session, batch)

        # Process in concurrent batches
        all_embeddings: List[np.ndarray] = []
        tasks: List = []

        for i in tqdm(
            range(0, len(texts), self.config.batch_size),
            desc=desc,
            unit="batch"
        ):
            tasks.append(embed_batch(i))

            # Process in groups of concurrent_batches
            if len(tasks) >= self.config.concurrent_batches:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch embedding failed: {result}")
                        self.stats.failed_batches += 1
                    else:
                        all_embeddings.extend(np.array(result))
                        self.stats.total_batches += 1

                tasks = []

        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding failed: {result}")
                    self.stats.failed_batches += 1
                else:
                    all_embeddings.extend(np.array(result))
                    self.stats.total_batches += 1

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    async def store_embeddings_batched(
        self,
        query: str,
        embeddings_data: List[Dict[str, Any]],
        desc: str = "Storing embeddings"
    ) -> None:
        """Store embeddings in Neo4j in batches.

        Args:
            query: Cypher query for storing embeddings
            embeddings_data: List of dicts with 'id' and 'embedding' keys
            desc: Description for progress bar
        """
        batch = []

        for data in tqdm(embeddings_data, desc=desc):
            batch.append(data)

            if len(batch) >= self.config.storage_batch_size:
                await kg_driver.run_query_async(query, {"data": batch})
                batch = []

        # Store remaining items
        if batch:
            await kg_driver.run_query_async(query, {"data": batch})

    def get_all_edges_batched(self) -> List[KGRelation]:
        """Retrieve all relations from Neo4j in batches.

        Returns:
            List of KGRelation objects
        """
        skip = 0
        all_relations: List[KGRelation] = []

        while True:
            query = textwrap.dedent("""\
            MATCH (e)-[r]->(t)
            RETURN DISTINCT
                elementId(e) AS src_id, labels(e) AS src_types, e.name AS src_name,
                apoc.map.removeKey(properties(e), "_embedding") AS src_properties,
                elementId(t) AS dst_id, labels(t) AS dst_types, t.name AS dst_name,
                apoc.map.removeKey(properties(t), "_embedding") AS dst_properties,
                elementId(r) AS id, type(r) AS relation,
                apoc.map.fromPairs([key IN keys(r) WHERE key <> "_embedding" | [key, r[key]]]) AS rel_properties
            SKIP $skip
            LIMIT $limit
            """)

            results = kg_driver.run_query(
                query,
                {"skip": skip, "limit": self.config.retrieval_batch_size}
            )

            if not results:
                break

            for record in results:
                relation = KGRelation(
                    id=record["id"],
                    name=record["relation"],
                    source=self._parse_entity(record, prefix="src"),
                    target=self._parse_entity(record, prefix="dst"),
                    description=record["rel_properties"].get("description"),
                    created_at=record["rel_properties"].get("created_at"),
                    modified_at=record["rel_properties"].get("modified_at"),
                    properties={
                        k: v
                        for k, v in record["rel_properties"].items()
                        if k not in {"description", "created_at", "modified_at"}
                    }
                )
                all_relations.append(relation)

            skip += self.config.retrieval_batch_size

        return all_relations

    @staticmethod
    def _parse_entity(record: Dict[str, Any], prefix: str) -> KGEntity:
        """Parse entity from Neo4j record.

        Args:
            record: Neo4j record dictionary
            prefix: Prefix for field names (e.g., "src" or "dst")

        Returns:
            KGEntity object
        """
        properties = record[f"{prefix}_properties"]

        return KGEntity(
            id=record[f"{prefix}_id"],
            type=record[f"{prefix}_types"][0],
            name=record[f"{prefix}_name"],
            description=properties.get("description"),
            created_at=properties.get("created_at"),
            modified_at=properties.get("modified_at"),
            properties={
                k: v
                for k, v in properties.items()
                if k not in {"name", "description", "created_at", "modified_at"}
            }
        )


class KGEmbedder(KGEmbedderBase):
    """Main KG embedder implementation.

    Handles embedding generation and storage for entities, relations,
    and schemas in the knowledge graph.
    """

    async def embed_entities(self) -> None:
        """Generate and store embeddings for all entities."""
        logger.info("Loading entities...")
        entities = kg_driver.get_entities()
        self.stats.total_entities = len(entities)

        if not entities:
            logger.warning("No entities found to embed")
            return

        # Generate text descriptions
        descriptions = [entity_to_text(entity) for entity in entities]
        logger.info(f"Embedding {len(entities)} entities...")
        logger.info(f"Example entities: {descriptions[:5]}")

        # Generate embeddings
        embeddings_np = await self.generate_embeddings_batched(
            descriptions,
            desc="Entity embeddings"
        )

        # Prepare data for storage
        embeddings_data = [
            {"id": entity.id, "embedding": embedding.tolist()}
            for entity, embedding in zip(entities, embeddings_np)
        ]

        # Store embeddings
        query = f"""
        UNWIND $data AS row
        MATCH (n) WHERE elementId(n) = row.id
        CALL db.create.setNodeVectorProperty(n, '{PROP_EMBEDDING}', row.embedding)
        """

        await self.store_embeddings_batched(
            query,
            embeddings_data,
            desc="Storing entity embeddings"
        )

        self.stats.entities_embedded = len(entities)

        # Mark as embeddable
        kg_driver.run_query(f"MATCH(n) SET n:{TYPE_EMBEDDABLE}")

        # Create vector index
        self._create_entity_vector_index()

    async def embed_relations(self) -> None:
        """Generate and store embeddings for all relations."""
        logger.info("Loading relations...")
        relations = self.get_all_edges_batched()
        self.stats.total_relations = len(relations)

        if not relations:
            logger.warning("No relations found to embed")
            return

        # Generate text descriptions
        descriptions = [relation_to_text(relation) for relation in relations]
        logger.info(f"Embedding {len(relations)} relations...")
        logger.info(f"Example relations: {descriptions[:5]}")

        # Generate embeddings
        embeddings_np = await self.generate_embeddings_batched(
            descriptions,
            desc="Relation embeddings"
        )

        # Prepare data for storage
        embeddings_data = [
            {"id": relation.id, "embedding": embedding.tolist()}
            for relation, embedding in zip(relations, embeddings_np)
        ]

        # Store embeddings
        query = f"""
        UNWIND $data AS row
        MATCH ()-[r]->() WHERE elementId(r) = row.id
        CALL db.create.setRelationshipVectorProperty(r, '{PROP_EMBEDDING}', row.embedding)
        """

        await self.store_embeddings_batched(
            query,
            embeddings_data,
            desc="Storing relation embeddings"
        )

        self.stats.relations_embedded = len(relations)

    async def embed_entity_schemas(self) -> None:
        """Generate and store embeddings for entity schemas."""
        logger.info("Loading entity schemas...")
        entity_schemas = kg_driver.get_entity_schema()
        self.stats.total_entity_schemas = len(entity_schemas)

        if not entity_schemas:
            logger.warning("No entity schemas found to embed")
            return

        # Generate text descriptions
        descriptions = [entity_schema_to_text(schema) for schema in entity_schemas]
        logger.info(f"Embedding {len(entity_schemas)} entity schemas...")
        logger.info(f"Example entity types: {descriptions[:5]}")

        # Generate embeddings
        embeddings_np = await self.generate_embeddings_batched(
            descriptions,
            desc="Entity schema embeddings"
        )

        # Prepare data for storage
        embeddings_data = [
            {"name": schema, "embedding": embedding.tolist()}
            for schema, embedding in zip(entity_schemas, embeddings_np)
        ]

        # Store embeddings
        query = f"""
        UNWIND $data AS row
        MERGE (s:_EntitySchema {{name: row.name}})
        WITH s, row
        CALL db.create.setNodeVectorProperty(s, '{PROP_EMBEDDING}', row.embedding)
        """

        await self.store_embeddings_batched(
            query,
            embeddings_data,
            desc="Storing entity schema embeddings"
        )

        self.stats.schemas_embedded += len(entity_schemas)

        # Create vector index
        self._create_entity_schema_vector_index()

    async def embed_relation_schemas(self) -> None:
        """Generate and store embeddings for relation schemas."""
        logger.info("Loading relation schemas...")
        relation_schemas = kg_driver.get_relation_schema()
        self.stats.total_relation_schemas = len(relation_schemas)

        if not relation_schemas:
            logger.warning("No relation schemas found to embed")
            return

        # Generate text descriptions
        descriptions = [
            relation_schema_to_text(schema)
            for schema in relation_schemas
        ]
        logger.info(f"Embedding {len(relation_schemas)} relation schemas...")
        logger.info(f"Example relation types: {descriptions[:5]}")

        # Generate embeddings
        embeddings_np = await self.generate_embeddings_batched(
            descriptions,
            desc="Relation schema embeddings"
        )

        # Prepare data for storage
        embeddings_data = [
            {
                "source_type": schema[0],
                "name": schema[1],
                "target_type": schema[2],
                "embedding": embedding.tolist()
            }
            for schema, embedding in zip(relation_schemas, embeddings_np)
        ]

        # Store embeddings
        query = f"""
        UNWIND $data AS row
        MERGE (s:_RelationSchema {{
            name: row.name,
            source_type: row.source_type,
            target_type: row.target_type
        }})
        WITH s, row
        CALL db.create.setNodeVectorProperty(s, '{PROP_EMBEDDING}', row.embedding)
        """

        await self.store_embeddings_batched(
            query,
            embeddings_data,
            desc="Storing relation schema embeddings"
        )

        self.stats.schemas_embedded += len(relation_schemas)

        # Create vector index
        self._create_relation_schema_vector_index()

    async def embed_all(self) -> EmbeddingStats:
        """Run the complete embedding pipeline.

        Returns:
            Statistics about the embedding operation
        """
        logger.info("=" * 60)
        logger.info("Starting KG embedding pipeline")
        logger.info("=" * 60)

        # Embed entities
        await self.embed_entities()

        # Embed relations
        await self.embed_relations()

        # Embed entity schemas
        await self.embed_entity_schemas()

        # Embed relation schemas
        await self.embed_relation_schemas()

        logger.info("=" * 60)
        logger.info("Embedding pipeline completed!")
        logger.info("=" * 60)
        self._log_stats()

        return self.stats

    def _create_entity_vector_index(self) -> None:
        """Create vector index for entities."""
        query = f"""
        CREATE VECTOR INDEX entityVector IF NOT EXISTS
        FOR (n:_Embeddable)
        ON n.{PROP_EMBEDDING}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.vector_dimensions},
                `vector.similarity_function`: '{self.config.similarity_function}'
            }}
        }}
        """
        kg_driver.run_query(query)
        logger.info("Entity vector index created")

    def _create_entity_schema_vector_index(self) -> None:
        """Create vector index for entity schemas."""
        query = f"""
        CREATE VECTOR INDEX entitySchemaVector IF NOT EXISTS
        FOR (s:_EntitySchema)
        ON s.{PROP_EMBEDDING}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.vector_dimensions},
                `vector.similarity_function`: '{self.config.similarity_function}'
            }}
        }}
        """
        kg_driver.run_query(query)
        logger.info("Entity schema vector index created")

    def _create_relation_schema_vector_index(self) -> None:
        """Create vector index for relation schemas."""
        query = f"""
        CREATE VECTOR INDEX relationSchemaVector IF NOT EXISTS
        FOR (s:_RelationSchema)
        ON s.{PROP_EMBEDDING}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.vector_dimensions},
                `vector.similarity_function`: '{self.config.similarity_function}'
            }}
        }}
        """
        kg_driver.run_query(query)
        logger.info("Relation schema vector index created")

    def _log_stats(self) -> None:
        """Log embedding statistics."""
        logger.info("Embedding Statistics:")
        logger.info(f"  Entities: {self.stats.entities_embedded}/{self.stats.total_entities}")
        logger.info(f"  Relations: {self.stats.relations_embedded}/{self.stats.total_relations}")
        logger.info(f"  Entity Schemas: {self.stats.total_entity_schemas}")
        logger.info(f"  Relation Schemas: {self.stats.total_relation_schemas}")
        logger.info(f"  Total Batches: {self.stats.total_batches}")
        logger.info(f"  Failed Batches: {self.stats.failed_batches}")


# Backward compatibility alias
KG_Embedder = KGEmbedder
