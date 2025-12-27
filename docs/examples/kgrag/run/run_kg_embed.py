import asyncio
import textwrap
from tqdm import tqdm
from typing import List
from utils.logger import logger
import numpy as np
import openai

from constants import (
    API_KEY,
    EMB_API_BASE,
    EMB_MODEL_NAME,
    EMB_TIME_OUT,
    RITS_API_KEY,
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
    relation_schema_to_text
)
from utils.utils import generate_embedding


# Define a Semaphore to control concurrency (e.g., max 50 tasks at a time)
SEMAPHORE = asyncio.Semaphore(50)

class KG_Embedder():
    def __init__(self, emb_session):
        self.emb_session = emb_session

    def get_all_edges(self, batch_size=500000) -> List[KGRelation]:
        """Special batched implementation: Retrieve all relations in batches to prevent memory issues."""
        skip = 0
        all_relations = []

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
            LIMIT $limit;
            """)

            results = kg_driver.run_query(query, {"skip": skip, "limit": batch_size})
            if not results:
                break  # No more data

            all_relations.extend([
                KGRelation(
                    id=record["id"],
                    name=record["relation"],
                    source=KGEntity(
                        id=record["src_id"],
                        type=record["src_types"][0],
                        name=record["src_name"],
                        description=record["src_properties"].get("description"),
                        created_at=record["src_properties"].get("created_at"),
                        modified_at=record["src_properties"].get("modified_at"),
                        properties={k: v for k, v in record["src_properties"].items()
                                if k not in {"name", "description", "created_at", "modified_at"}}
                    ),
                    target=KGEntity(
                        id=record["dst_id"],
                        type=record["dst_types"][0],
                        name=record["dst_name"],
                        description=record["dst_properties"].get("description"),
                        created_at=record["dst_properties"].get("created_at"),
                        modified_at=record["dst_properties"].get("modified_at"),
                        properties={k: v for k, v in record["dst_properties"].items()
                                if k not in {"name", "description", "created_at", "modified_at"}}
                    ),
                    description=record["rel_properties"].get("description"),
                    created_at=record["rel_properties"].get("created_at"),
                    modified_at=record["rel_properties"].get("modified_at"),
                    properties={k: v for k, v in record["rel_properties"].items()
                            if k not in {"description", "created_at", "modified_at"}}
                ) for record in results
            ])

            skip += batch_size

        return all_relations
    
    async def get_embedding(self, description_list, batch_size=8192, concurrent_batches=64):
        # embeddings_list = []
        # for i in tqdm(range(0, len(description_list), batch_size), desc="Embedding"):
        #     batch = description_list[i : i + batch_size]
        #     embeddings = await generate_embedding(batch)
        #     embeddings_list.extend(np.array(embeddings))  # Store batch results
        
        # return np.vstack(embeddings_list)
        async def embed_batch(start_idx):
            batch = description_list[start_idx: start_idx + batch_size]
            return await generate_embedding(self.emb_session, batch)

        tasks = []
        embeddings_list = []
        for i in tqdm(range(0, len(description_list), batch_size), desc="Embedding"):
            tasks.append(embed_batch(i))
            
            # Run batches in groups of `concurrent_batches`
            if len(tasks) >= concurrent_batches:
                results = await asyncio.gather(*tasks)
                for r in results:
                    embeddings_list.extend(np.array(r))
                tasks = []

        # Finish remaining batches
        if tasks:
            results = await asyncio.gather(*tasks)
            for r in results:
                embeddings_list.extend(np.array(r))

        return np.vstack(embeddings_list)
    
    async def store_embedding(self, query, data_str, datas, embeddings_np, batch_size=50_000):
        batch = []

        for data, embedding in tqdm(zip(datas, embeddings_np), desc="Storing embedding"):
            batch.append(eval(data_str)) 

            if len(batch) == batch_size:
                # Clone and queue async task
                params = {"data": list(batch)}  # Avoid mutation
                await kg_driver.run_query_async(query, params)
                batch = []

        # Final batch
        if batch:
            params = {"data": batch}
            await kg_driver.run_query_async(query, params)
    
    async def embed(self):
        ######################## Generate entity embedding ########################
        logger.info("Loading entities...")
        all_entities = kg_driver.get_entities()
        description_list = [entity_to_text(entity) for entity in all_entities]
        logger.info(f"Embedding {len(all_entities)} entities...")
        logger.info(f"Example entities: {description_list[:5]}")
        if len(description_list):
            embeddings_np = await self.get_embedding(description_list)

            # Store embedding
            node_ids = [entity.id for entity in all_entities]  # Store Neo4j element IDs
            query = f"""
            UNWIND $data AS row
            MATCH (n) WHERE elementId(n) = row.id
            CALL db.create.setNodeVectorProperty(n, '{PROP_EMBEDDING}', row.embedding)
            """
            data_str = '{"id": data, "embedding": embedding.tolist()}'
            await self.store_embedding(query, data_str, node_ids, embeddings_np)

        kg_driver.run_query(f"MATCH(n) SET n:{TYPE_EMBEDDABLE}")

        # Vector indexing
        kg_driver.run_query(f"""CREATE VECTOR INDEX entityVector IF NOT EXISTS
            FOR (n:_Embeddable)
            ON n.{PROP_EMBEDDING}
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}}}"""
        )

        ######################## Generate relation embedding ########################
        logger.info("Loading relations...")
        all_relations = self.get_all_edges()
        description_list = [relation_to_text(relation) for relation in all_relations]

        logger.info(f"Embedding {len(all_relations)} relations...")
        logger.info(f"Example relations: {description_list[:5]}")
        if len(description_list):
            # Generate embedding
            embeddings_np = await self.get_embedding(description_list)

            # Store embedding
            edge_ids = [relation.id for relation in all_relations]  # Store Neo4j element IDs
            query = f"""
            UNWIND $data AS row
            MATCH ()-[r]->() WHERE elementId(r) = row.id
            CALL db.create.setRelationshipVectorProperty(r, '{PROP_EMBEDDING}', row.embedding)
            """
            data_str = '{"id": data, "embedding": embedding.tolist()}'
            await self.store_embedding(query, data_str, edge_ids, embeddings_np)


        ######################## Generate entity schema embedding ########################
        logger.info("Loading entity schema...")
        entity_schema = kg_driver.get_entity_schema()
        description_list = [entity_schema_to_text(schema) for schema in entity_schema]
        logger.info(f"Embedding {len(entity_schema)} entity schema...")
        logger.info(f"Example entity types: {description_list[:5]}")
        if len(description_list):
            embeddings_np = await self.get_embedding(description_list)

            # Store embedding
            query = f"""
            UNWIND $data AS row
            MERGE (s:_EntitySchema {{name: row.name}})
            WITH s, row
            CALL db.create.setNodeVectorProperty(s, '{PROP_EMBEDDING}', row.embedding)
            """
            data_str = '{"name": data, "embedding": embedding.tolist()}'
            await self.store_embedding(query, data_str, entity_schema, embeddings_np)

        kg_driver.run_query(f"""CREATE VECTOR INDEX entitySchemaVector IF NOT EXISTS
            FOR (s:_EntitySchema)
            ON s.{PROP_EMBEDDING}
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}}}"""
        )

        ######################## Generate relation schema embedding ########################
        logger.info("Loading relation schema...")
        relation_schema = kg_driver.get_relation_schema()
        description_list = [relation_schema_to_text(schema) for schema in relation_schema]
        logger.info(f"Embedding {len(relation_schema)} relation schema...")
        logger.info(f"Example relation types: {description_list[:5]}")
        if len(description_list):
            embeddings_np = await self.get_embedding(description_list)

            # Store embedding
            query = f"""
            UNWIND $data AS row
            MERGE (s:_RelationSchema {{name: row.name, source_type: row.source_type, target_type: row.target_type}})
            WITH s, row
            CALL db.create.setNodeVectorProperty(s, '{PROP_EMBEDDING}', row.embedding)
            """
            data_str = '{"source_type": data[0], "name": data[1], "target_type": data[2], "embedding": embedding.tolist()}'
            await self.store_embedding(query, data_str, relation_schema, embeddings_np)

        kg_driver.run_query(f"""CREATE VECTOR INDEX relationSchemaVector IF NOT EXISTS
            FOR (s:_RelationSchema)
            ON s.{PROP_EMBEDDING}
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}}}"""
        )
    

if __name__ == "__main__":
    # Initialize embedding session
    if EMB_API_BASE:
        emb_session = openai.AsyncOpenAI(
            base_url=EMB_API_BASE,
            api_key=API_KEY,
            timeout=EMB_TIME_OUT,
            default_headers={'RITS_API_KEY': RITS_API_KEY}
        )
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        emb_session = SentenceTransformer(
            EMB_MODEL_NAME,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            ),
        )

    embedder = KG_Embedder(emb_session)

    # Async Route
    async def main():
        await embedder.embed()

    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)  # Set it as the current loop
    loop.run_until_complete(main())

    logger.info("Neo4j KG embedding completed ✅")