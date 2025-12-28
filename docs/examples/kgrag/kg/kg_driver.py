import asyncio
import os
import textwrap
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import neo4j
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
from datetime import datetime
from utils.utils import generate_embedding
from kg.kg_rep import (
    RESERVED_KEYS,
    normalize_key,
    normalize_value,
    normalize_relation,
    entity_schema_to_text,
    relation_schema_to_text,
    relation_to_text,
    RelevantRelation,
    KGEntity,
    KGRelation,
    RelevantEntity,
    entity_to_text,
    PROP_EMBEDDING,
    PROP_DESCRIPTION,
    PROP_PARAGRAPH,
    PROP_CREATED,
    PROP_MODIFIED,
    PROP_REFERENCE,
    normalize_entity,
    normalize_entity_type,
    TYPE_EMBEDDABLE
)
from utils.utils import maybe_load_json

# Load environment variables
load_dotenv()

# Get configuration from environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


@dataclass
class TemporalConstraint:
    around: Optional[datetime] = None       # approx near this date
    start: Optional[datetime] = None        # >= this datetime
    end: Optional[datetime] = None          # < this datetime

class KG_Driver:
    _instance = None

    # Maintain a singleton driver across files
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, database=None, emb_session=None):
        if not hasattr(self, "_initialized"):
            self._initialized = True

            self.driver = neo4j.GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.async_driver = neo4j.AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.database = database
            self.emb_session = emb_session

            self.entity_schema_cache = set(self.get_entity_schema())
            self.relation_schema_cache = set(self.get_relation_schema())

    def set_emb_session(self, emb_session):
        """Set or update the embedding session."""
        self.emb_session = emb_session

    async def close(self):
        self.driver.close()
        await self.async_driver.close()

    def run_query(self, query, parameters=None):
        """Run a Cypher query in Neo4j."""
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, parameters))

    async def run_query_async(self, query, parameters=None, semaphore=None, retries=5, delay=1):
        """Runs an async query with retries"""
        async def run():
            for attempt in range(retries):
                try:
                    async with self.async_driver.session() as session:
                        result = await session.run(query, parameters)
                        # Ensure query executes
                        return [record async for record in result]
                except neo4j.exceptions.TransientError as e:
                    if "DeadlockDetected" in str(e):
                        print(
                            f"Deadlock detected. Retrying {attempt + 1}/{retries}...")
                        # Exponential backoff
                        await asyncio.sleep(delay * (2 ** attempt))
                    else:
                        raise e
            raise RuntimeError("Max retries reached for Neo4j transaction.")

        if semaphore:
            async with semaphore:
                return await run()
        else:
            return await run()

    def build_temporal_clause(self,
        constraint: TemporalConstraint,
        param_dict: dict,
        date_field: str = "node._timestamp",
        var_prefix: str = "node"
    ) -> Tuple[str, Optional[str]]:
        """
        Constructs a Cypher clause for temporal filtering and/or temporal ordering.
    
        Args:
            constraint (TemporalConstraint): The temporal constraint to apply.
            param_dict (dict): A dictionary to populate with parameters for Cypher.
            date_field (str): The property path to the date field (e.g., "node.date").
            var_prefix (str): The variable name to use for duration calculation (e.g., "node", "rel").
    
        Returns:
            Tuple[str, Optional[str]]: A WHERE clause (string) and an optional time_diff computation for sorting.
        """
        filters = []
        time_diff_expr = None

        if constraint:
            time_expr = f"datetime(replace({date_field}, ' ', 'T'))"
            if constraint.around:
                param_dict["around"] = constraint.around.isoformat()
                # Difference in total seconds between entity time and target time
                time_diff_expr = (
                    f"abs(duration.inSeconds(datetime($around), {time_expr}).seconds) AS {var_prefix}_time_diff"
                )
        
            if constraint.start:
                param_dict["start"] = constraint.start.isoformat()
                filters.append(f"datetime({time_expr}) >= datetime($start)")
            if constraint.end:
                param_dict["end"] = constraint.end.isoformat()
                filters.append(f"datetime({time_expr}) < datetime($end)")
    
        where_clause = " AND ".join(filters) if filters else ""
    
        return where_clause, time_diff_expr
        
    def get_label(self, labels: List[str]) -> str:
        for label in labels:
            if not label.startswith("_"):
                return label
        return ""
    
    def get_properties(self, properties: Dict, current_time=None):
        results = {}
        for key, value in properties.items():
            if key in RESERVED_KEYS:
                continue
            res = maybe_load_json(value, force_load=False)
            if res and isinstance(res, dict):
                try:
                    results[normalize_key(key)] = {normalize_value(
                        property_value): info for property_value, info in res.items()}
                except:
                    print(f"Property value was not properly formatted: {res}")
            else:
                results[normalize_key(key)] = {normalize_value(
                    value): {"count": 1, "context": None, "last_seen": current_time}}
        return results

    # ====================== Entity/Relation Query ==================================

    def get_node_types(self):
        """Retrieve all entity types (node labels)"""
        query = "CALL db.labels();"
        results = self.run_query(query)
        return [record["label"] for record in results if not record["label"].startswith('_')]

    def get_edge_types(self):
        """Retrieve all relationship types"""
        query = "CALL db.relationshipTypes();"
        results = self.run_query(query)
        return [record["relationshipType"] for record in results]
    
    def get_entity_schema(self):
        """
        Retrieve all distinct entity types from the KG.
        """
        query = textwrap.dedent("""\
        CALL db.labels()
        YIELD label
        WHERE NOT label STARTS WITH "_"
        RETURN label
        """)
        results = self.run_query(query)
        return [
            record["label"] for record in results
        ]

    def get_relation_schema(self):
        """
        Retrieve all distinct (source_type, relation_type, target_type) triples from the KG.
        """
        query = textwrap.dedent("""\
        MATCH (a)-[r]->(b)
        WITH type(r) AS rel_type,
             [label IN labels(a) WHERE NOT label STARTS WITH "_"][0] AS source_type,
             [label IN labels(b) WHERE NOT label STARTS WITH "_"][0] AS target_type
        WHERE source_type IS NOT NULL AND target_type IS NOT NULL
        RETURN DISTINCT source_type, rel_type, target_type
        """)
        results = self.run_query(query)
        return [
            (record["source_type"], record["rel_type"], record["target_type"])
            for record in results
        ]
    
    def check_entity_schema(self, schema):
        return schema in self.entity_schema_cache

    def check_relation_schema(self, schema):
        return schema in self.relation_schema_cache

    async def add_entity_schema(self, entities_dict: Dict[str, KGEntity]):
        description_list = []
        for entity in entities_dict.values():
            schema = entity.type
            if not self.check_entity_schema(schema):
                self.entity_schema_cache.add(schema)
                description_list.append(entity_schema_to_text(schema))
        
        embeddings = await generate_embedding(self.emb_session, description_list)

        batch = []
        for name, embedding in zip(description_list, embeddings):
            batch.append({"name": name, "embedding": embedding})
        params = {"data": batch}

        query = f"""
        UNWIND $data AS row
        MERGE (s:_EntitySchema {{name: row.name}})
        WITH s, row
        CALL db.create.setNodeVectorProperty(s, '{PROP_EMBEDDING}', row.embedding)
        """
        await kg_driver.run_query_async(query, params)

    async def add_relation_schema(self, relations_dict: Dict[str, KGRelation]):
        relation_schema = []
        description_list = []
        for relation in relations_dict.values():
            schema = (relation.source.type, relation.name, relation.target.type)
            if not self.check_relation_schema(schema):
                self.relation_schema_cache.add(schema)
                description_list.append(relation_schema_to_text(schema))
                relation_schema.append(schema)
        
        embeddings = await generate_embedding(self.emb_session, description_list)

        batch = []
        for schema, embedding in zip(relation_schema, embeddings):
            batch.append({"source_type": schema[0], "name": schema[1], "target_type": schema[2], "embedding": embedding})
        params = {"data": batch}

        query = f"""
        UNWIND $data AS row
        MERGE (s:_RelationSchema {{name: row.name, source_type: row.source_type, target_type: row.target_type}})
        WITH s, row
        CALL db.create.setNodeVectorProperty(s, '{PROP_EMBEDDING}', row.embedding)
        """
        await kg_driver.run_query_async(query, params)
    
    def vector_search_entity_schema(self, embedding: List[float],
                               top_k: int = 5):
        query = textwrap.dedent(f"""\
        CALL db.index.vector.queryNodes('entitySchemaVector', $top_k, $embedding)
        YIELD node, score
        RETURN node.name AS type, score
        """)
        params = {"embedding": embedding, "top_k": top_k}
        
        results = self.run_query(query, params)

        schema = [
            record['type'] for record in results
        ]
        return schema

    def vector_search_relation_schema(self, embedding: List[float],
                               top_k: int = 5):
        query = textwrap.dedent(f"""\
        CALL db.index.vector.queryNodes('relationSchemaVector', $top_k, $embedding)
        YIELD node, score
        RETURN node.name AS rel, node.source_type AS source, node.target_type AS target, score
        """)
        params = {"embedding": embedding, "top_k": top_k}
        
        results = self.run_query(query, params)

        schema = [
            (record['source'], record['rel'], record['target']) for record in results
        ]
        return schema
    
    def get_entities(self,
                     type: Optional[str] = None,
                     name: Optional[str] = None,
                     fuzzy: bool = False,
                     embedding: Optional[List[float]] = None,
                     constraint: Optional[TemporalConstraint] = None,
                     top_k: Optional[int] = None,
                     return_score: bool = False) -> Union[List[KGEntity], List[RelevantEntity]]:
        """
        Perform a exact match or vector-based nearest neighbor search in Neo4j to find the most similar entities.

        This function queries the Neo4j vector index `entityVector` using a given embedding 
        and retrieves the top-K closest entities based on vector similarity.

        Args:
            embedding (List[float]): The embedding representation of the query entity.
            top_k (int): The number of top results to retrieve.
            return_score (bool, optional): Whether to return similarity scores alongside entities.
                                        Defaults to False.

        Returns:
            List[KGEntity] or List[RelevantEntity]: 
                - If `return_score` is `False`, returns a list of `KGEntity` objects representing 
                the retrieved entities.
                - If `return_score` is `True`, returns a list of `RelevantEntity` objects, each 
                containing a `KGEntity` and its similarity score.

        Raises:
            neo4j.exceptions.Neo4jError: If there is an issue with the Neo4j query execution.

        Example:
            >>> query_embedding = [0.12, -0.45, 0.88, ...]  # Example embedding
            >>> results = vector_search_entity(query_embedding, top_k=5)
            >>> for entity in results:
            >>>     print(entity.name)

            # If retrieving similarity scores:
            >>> results_with_scores = vector_search_entity(query_embedding, top_k=5, return_score=True)
            >>> for rel_entity in results_with_scores:
            >>>     print(f"Entity: {rel_entity.entity.name}, Score: {rel_entity.score}")
        """
        params = {"top_k": top_k, "embedding": embedding}
        score_clause, where_clause, order_clause = "", "", ""

        label = f":{type}" if type else ""
        match_clause = f"MATCH (n{label})"
        if name:
            if not fuzzy:
                where_clause = "n.name = $name"
            else:
                score_clause = ", apoc.text.levenshteinSimilarity(n.name, $name) AS score"
                order_clause = "ORDER BY score DESC"
            params["name"] = name
        
        if embedding:
            raw_k = top_k * 20 if constraint else top_k
            params.update({"raw_k": raw_k})
            match_clause = textwrap.dedent(f"""\
                CALL db.index.vector.queryNodes('entityVector', $raw_k, $embedding)
                YIELD node AS n, score
            """)
            if constraint:
                where_clause, time_diff_expr = self.build_temporal_clause(constraint, params, "n._timestamp", "n")
                score_clause = f", {time_diff_expr}" if time_diff_expr else ""
                order_clause = f"ORDER BY n_time_diff ASC, score DESC"
            else:
                order_clause = "ORDER BY score DESC"
        
        limit_clause = f"LIMIT $top_k" if top_k else ""

        query = textwrap.dedent(f"""\
            {match_clause}
            {f"WITH n{score_clause}" if fuzzy or constraint else ''}
            {f"WHERE {where_clause}" if where_clause else ''}
            RETURN elementId(n) AS id, labels(n) AS labels, n.name AS name,
                apoc.map.removeKey(properties(n), '{PROP_EMBEDDING}') AS properties
                {", score" if embedding or fuzzy else ""}
            {order_clause}
            {limit_clause}
        """)

        results = self.run_query(query, params)

        entities = [
            KGEntity(
                id=record["id"],
                type=self.get_label(record["labels"]),
                name=record["name"],
                description=record["properties"].get(PROP_DESCRIPTION),
                paragraph=record["properties"].get(PROP_PARAGRAPH),
                created_at=record["properties"].get(PROP_CREATED),
                modified_at=record["properties"].get(PROP_MODIFIED),
                properties=self.get_properties(record["properties"]),
                ref=record["properties"].get(PROP_REFERENCE)
            ) for record in results
        ]

        if return_score and (embedding or fuzzy):
            return [RelevantEntity(entity, record["score"]) for entity, record in zip(entities, results)]
        else:
            return entities

    # def get_entities(self, type: str = None, 
    #               name: str = None, 
    #               top_k: int = None,
    #               fuzzy: bool = False) -> KGEntity:
    #     """
    #     Retrieve entities (nodes) from the KG that match the criteria and return them as KGEntity objects.

    #     Args:
    #         type (str, optional): Filter the entity type.
    #         name (str, optional): Specify the entity name.
    #         top_k (str, optional): Only return up to top-k entities.

    #     Returns:
    #         List[KGEntity]: A list of KGEntity objects.
    #     """
    #     match_clause = "WHERE n.name = $name " if not fuzzy else \
    #                    "WITH n, apoc.text.levenshteinSimilarity(n.name, $name) AS score ORDER BY score DESC "
        
    #     query = "MATCH (n" + (f":{type}" if type else "") + ") " + \
    #         (match_clause if name else "") + \
    #         f"RETURN elementId(n) AS id, labels(n) AS labels, n.name AS name, apoc.map.removeKey(properties(n), '{PROP_EMBEDDING}') AS properties" + \
    #         (f" LIMIT {top_k}" if top_k else "")

        
    #     params = {"type": type, "name": name, "top_k": top_k}

    #     results = self.run_query(query, params)

    #     return [
    #         KGEntity(
    #             id=record["id"],  # Convert Neo4j string ID to integer
    #             type=self.get_label(record["labels"]),
    #             name=record["name"],
    #             description=record["properties"].get(PROP_DESCRIPTION),
    #             created_at=record["properties"].get(PROP_CREATED),
    #             modified_at=record["properties"].get(PROP_MODIFIED),
    #             properties=self.get_properties(record["properties"]),
    #             ref=record["properties"].get(PROP_REFERENCE)
    #         ) for record in results
    #     ]
    
    # def vector_search_entity(self, embedding: List[float],
    #                          top_k: int = 5,
    #                          constraint: TemporalConstraint = None,
    #                          return_score: bool = False) -> List[KGEntity] | List[RelevantEntity]:
    #     raw_k = top_k * 20 if constraint else top_k  # fetch more candidates
    #     params = {"embedding": embedding, "raw_k": raw_k, "top_k": top_k}
    #     where_clause, time_diff_expr = self.build_temporal_clause(constraint, params)
    #     query = textwrap.dedent(f"""\
    #         CALL db.index.vector.queryNodes('entityVector', $raw_k, $embedding)
    #         YIELD node, score
    #         WITH node, score
    #         {',' + time_diff_expr if time_diff_expr else ""}
    #         {f"WHERE {where_clause}" if where_clause else ""}
    #         RETURN elementId(node) AS id, labels(node) AS labels, node.name AS name,
    #                apoc.map.removeKey(properties(node), '{PROP_EMBEDDING}') AS properties,
    #                score
    #         ORDER BY {f'node_time_diff ASC,' if time_diff_expr else ''} score DESC
    #         LIMIT $top_k
    #     """)

    #     results = self.run_query(query, params)

    #     entities = [
    #         KGEntity(
    #             id=record["id"],  # Convert Neo4j string ID to integer
    #             type=self.get_label(record["labels"]),
    #             name=record["name"],
    #             description=record["properties"].get(PROP_DESCRIPTION),
    #             created_at=record["properties"].get(PROP_CREATED),
    #             modified_at=record["properties"].get(PROP_MODIFIED),
    #             properties=self.get_properties(record["properties"]),
    #             ref=record["properties"].get(PROP_REFERENCE)
    #         ) for record in results
    #     ]
    #     if not return_score:
    #         return entities
    #     else:
    #         return [
    #             RelevantEntity(entity, record["score"]) for entity, record in zip(entities, results)
    #         ]
    
    def get_relations(self,
                      source: Optional[KGEntity] = None,
                      relation: Optional[str] = None,
                      target: Optional[KGEntity] = None,
                      source_type: Optional[str] = None,
                      target_type: Optional[str] = None,
                      unique_relation: bool = False,
                      embedding: Optional[List[float]] = None,
                      target_embedding: Optional[List[float]] = None,
                      top_k: Optional[int] = None,
                      return_score: bool = False) -> Union[List[KGRelation], List[RelevantRelation]]:
        """
        Perform an exact search or vector-based nearest neighbor search on relations in Neo4j.

        This function queries Neo4j's vector index on relationship embeddings and retrieves 
        the top-K most similar relations based on cosine similarity.

        Args:
            embedding (List[float]): The embedding representation of the query relation.
            top_k (int): The number of top results to retrieve.
            source (KGEntity, optional): The source entity to filter relations. Default is None.
            target (KGEntity, optional): The target entity to filter relations. Default is None.
            return_score (bool, optional): Whether to return similarity scores alongside relations. 
                                        Defaults to False.

        Returns:
            List[KGRelation] or List[RelevantRelation]: 
                - If `return_score` is `False`, returns a list of `KGRelation` objects.
                - If `return_score` is `True`, returns a list of `RelevantRelation` objects, 
                each containing a `KGRelation` and its similarity score.

        Raises:
            neo4j.exceptions.Neo4jError: If there is an issue with the Neo4j query execution.

        Example:
            >>> query_embedding = [0.12, -0.45, 0.88, ...]  # Example relation embedding
            >>> results = vector_search_relation(query_embedding, top_k=5)
            >>> for relation in results:
            >>>     print(f"{relation.source.name} -[{relation.name}]-> {relation.target.name}")

            # If retrieving similarity scores:
            >>> results_with_scores = vector_search_relation(query_embedding, top_k=5, return_score=True)
            >>> for rel in results_with_scores:
            >>>     print(f"Relation: {rel.relation.name}, Score: {rel.score}")
        """
        filters = []
        params = {"embedding": embedding, "tgt_embedding": target_embedding, "top_k": top_k} if embedding or target_embedding else {}

        # Labels
        src_label = f":{source_type}" if source_type else ""
        tgt_label = f":{target_type}" if target_type else ""
        rel_type = f":{relation}" if relation else ""

        if source:
            filters.append("elementId(src) = $source_id" if source.id else "src.name = $source_name")
            params.update({"source_id": source.id, "source_name": source.name})
        if target:
            filters.append("elementId(tgt) = $target_id" if target.id else "tgt.name = $target_name")
            params.update({"target_id": target.id, "target_name": target.name})
        if relation:
            filters.append("type(rel) = $relation")
            params["relation"] = relation
        if embedding:
            filters.append(f"rel.{PROP_EMBEDDING} IS NOT NULL")
        if target_embedding:
            filters.append(f"tgt.{PROP_EMBEDDING} IS NOT NULL")

        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        # Base query: two directed parts + UNION
        match_block = textwrap.dedent(f"""
            CALL() {{
                MATCH (src{src_label})-[rel{rel_type}]->(tgt{tgt_label})
                {where_clause}
                RETURN src, rel, tgt, 'forward' AS direction
                UNION
                MATCH (src{src_label})<-[rel{rel_type}]-(tgt{tgt_label})
                {where_clause}
                RETURN src, rel, tgt, 'reverse' AS direction
            }}
        """)

        return_clause = textwrap.dedent(f"""\
            WITH [label IN labels(src) WHERE not label STARTS WITH "_" ] AS src_type, type(rel) AS relation_type,
            [label IN labels(tgt) WHERE not label STARTS WITH "_" ] AS tgt_type, collect({{src: src, rel: rel, tgt: tgt}}) AS rel_set,
            direction
            RETURN 
            elementId(rel_set[0].src) AS src_id, labels(rel_set[0].src) AS src_types, rel_set[0].src.name AS src_name, 
            apoc.map.removeKey(properties(rel_set[0].src), '{PROP_EMBEDDING}') AS src_properties,
            elementId(rel_set[0].tgt) AS tgt_id, labels(rel_set[0].tgt) AS tgt_types, rel_set[0].tgt.name AS tgt_name, 
            apoc.map.removeKey(properties(rel_set[0].tgt), '{PROP_EMBEDDING}') AS tgt_properties, 
            elementId(rel_set[0].rel) AS id, relation_type AS relation, 
            apoc.map.removeKey(properties(rel_set[0].rel), '{PROP_EMBEDDING}')  AS rel_properties,
            direction
            """) if unique_relation else textwrap.dedent(f"""\
            RETURN DISTINCT 
            elementId(src) AS src_id, labels(src) AS src_types, src.name AS src_name,
            apoc.map.removeKey(properties(src), '{PROP_EMBEDDING}') AS src_properties,
            elementId(tgt) AS tgt_id, labels(tgt) AS tgt_types, tgt.name AS tgt_name,
            apoc.map.removeKey(properties(tgt), '{PROP_EMBEDDING}') AS tgt_properties,
            elementId(rel) AS id, type(rel) AS relation, 
            apoc.map.removeKey(properties(rel), '{PROP_EMBEDDING}') AS rel_properties,
            direction
            """)

        score_expr = ", vector.similarity.cosine(rel._embedding, $embedding) AS score ORDER BY score DESC" if embedding else ""
        score_expr += ", vector.similarity.cosine(tgt._embedding, $tgt_embedding) AS score ORDER BY score DESC" if target_embedding else ""
        limit_clause = f"LIMIT $top_k" if top_k else ""

        query = "\n".join([match_block, return_clause, score_expr, limit_clause])

        results = self.run_query(query, params)

        relations = [
            KGRelation(
                id=record["id"],
                name=record["relation"],
                source=KGEntity(
                    id=record["src_id"],
                    type=self.get_label(record["src_types"]),
                    name=record["src_name"],
                    description=record["src_properties"].get(PROP_DESCRIPTION),
                    paragraph=record["src_properties"].get(PROP_PARAGRAPH),
                    created_at=record["src_properties"].get(PROP_CREATED),
                    modified_at=record["src_properties"].get(PROP_MODIFIED),
                    properties=self.get_properties(record["src_properties"]),
                    ref=record["src_properties"].get(PROP_REFERENCE)
                ),
                target=KGEntity(
                    id=record["tgt_id"],
                    type=self.get_label(record["tgt_types"]),
                    name=record["tgt_name"],
                    description=record["tgt_properties"].get(PROP_DESCRIPTION),
                    paragraph=record["tgt_properties"].get(PROP_PARAGRAPH),
                    created_at=record["tgt_properties"].get(PROP_CREATED),
                    modified_at=record["tgt_properties"].get(PROP_MODIFIED),
                    properties=self.get_properties(record["tgt_properties"]),
                    ref=record["tgt_properties"].get(PROP_REFERENCE)
                ),
                description=record["rel_properties"].get(PROP_DESCRIPTION),
                paragraph=record["rel_properties"].get(PROP_PARAGRAPH),
                created_at=record["rel_properties"].get(PROP_CREATED),
                modified_at=record["rel_properties"].get(PROP_MODIFIED),
                properties=self.get_properties(record["rel_properties"]),
                direction=record.get("direction"),
                ref=record["rel_properties"].get(PROP_REFERENCE)
            ) for record in results
        ]

        return (
            [RelevantRelation(rel, record["score"]) for rel, record in zip(relations, results)]
            if (return_score and embedding) else relations
        )

    # def get_relations(self, source: Optional[KGEntity] = None,
    #                  relation: Optional[str] = None,
    #                  target: Optional[KGEntity] = None,
    #                  source_type: Optional[str] = None,
    #                  target_type: Optional[str] = None,
    #                  unique_relation: bool = False,
    #                  top_k: int = None) -> List[KGRelation]:
    #     """
    #     Query Neo4j to retrieve all relations starting from a given entity.

    #     Args:
    #         source (KGEntity, optional): The source entity from which relations originate.
    #         relation (str, optional): Specify the relationship.
    #         target (KGEntity, optional): The target entity from which relations end.
    #         source_type (str): Specify the source entity type.
    #         target_type (str): Specify the target entity type.
    #         unique_relation (bool): Only return the unique type of relations started from the entity.
    #         top_k (int): Limit the number of output to be top_k.

    #     Returns:
    #         List[KGRelation]: A list of KGRelation objects representing relationships in the KG.
    #     """
    #     params = {}
    #     filters = []
    
    #     # Labels
    #     src_label = f":{source_type}" if source_type else ""
    #     tgt_label = f":{target_type}" if target_type else ""
    #     rel_type = f":{relation}" if relation else ""
    
    #     # Filters and params
    #     if source:
    #         filters.append("elementId(src) = $source_id" if source.id else "src.name = $source_name")
    #         params.update({"source_id": source.id, "source_name": source.name})
    #     if target:
    #         filters.append("elementId(tgt) = $target_id" if target.id else "tgt.name = $target_name")
    #         params.update({"target_id": target.id, "target_name": target.name})
    #     if relation:
    #         filters.append("type(rel) = $relation" if relation else "")
    #         params.update({"relation": relation})
    
    #     where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    
    #     # Base query: two directed parts + UNION
    #     match_block = textwrap.dedent(f"""
    #         CALL() {{
    #             MATCH (src{src_label})-[rel{rel_type}]->(tgt{tgt_label})
    #             {where_clause}
    #             RETURN src, rel, tgt, 'forward' AS direction
    #             UNION
    #             MATCH (src{src_label})<-[rel{rel_type}]-(tgt{tgt_label})
    #             {where_clause}
    #             RETURN src, rel, tgt, 'reverse' AS direction
    #         }}
    #     """)

    #     limit_clause = f"WITH src, rel, tgt, direction ORDER BY rand() LIMIT {top_k}" if top_k else ""

    #     return_clause = textwrap.dedent(f"""\
    #         WITH [label IN labels(src) WHERE not label STARTS WITH "_" ] AS src_type, type(rel) AS relation_type,
    #         [label IN labels(tgt) WHERE not label STARTS WITH "_" ] AS tgt_type, collect({{src: src, rel: rel, tgt: tgt}}) AS rel_set,
    #         direction
    #         RETURN 
    #         elementId(rel_set[0].src) AS src_id, labels(rel_set[0].src) AS src_types, rel_set[0].src.name AS src_name, 
    #         apoc.map.removeKey(properties(rel_set[0].src), '{PROP_EMBEDDING}') AS src_properties,
    #         elementId(rel_set[0].tgt) AS tgt_id, labels(rel_set[0].tgt) AS tgt_types, rel_set[0].tgt.name AS tgt_name, 
    #         apoc.map.removeKey(properties(rel_set[0].tgt), '{PROP_EMBEDDING}') AS tgt_properties, 
    #         elementId(rel_set[0].rel) AS id, relation_type AS relation, 
    #         apoc.map.removeKey(properties(rel_set[0].rel), '{PROP_EMBEDDING}')  AS rel_properties,
    #         direction
    #         """) if unique_relation else textwrap.dedent(f"""\
    #         RETURN DISTINCT 
    #         elementId(src) AS src_id, labels(src) AS src_types, src.name AS src_name,
    #         apoc.map.removeKey(properties(src), '{PROP_EMBEDDING}') AS src_properties,
    #         elementId(tgt) AS tgt_id, labels(tgt) AS tgt_types, tgt.name AS tgt_name,
    #         apoc.map.removeKey(properties(tgt), '{PROP_EMBEDDING}') AS tgt_properties,
    #         elementId(rel) AS id, type(rel) AS relation, 
    #         apoc.map.removeKey(properties(rel), '{PROP_EMBEDDING}') AS rel_properties,
    #         direction
    #         """)

    #     query = "\n".join([match_block, limit_clause, return_clause])

    #     results = self.run_query(query, params)

    #     relations = [
    #         KGRelation(
    #             id=record["id"],
    #             name=record["relation"],
    #             source=KGEntity(
    #                 id=record["src_id"],
    #                 type=self.get_label(record["src_types"]),
    #                 name=record["src_name"],
    #                 description=record["src_properties"].get(PROP_DESCRIPTION),
    #                 created_at=record["src_properties"].get(PROP_CREATED),
    #                 modified_at=record["src_properties"].get(PROP_MODIFIED),
    #                 properties=self.get_properties(record["src_properties"]),
    #                 ref=record["src_properties"].get(PROP_REFERENCE)
    #             ),
    #             target=KGEntity(
    #                 id=record["tgt_id"],
    #                 type=self.get_label(record["tgt_types"]),
    #                 name=record["tgt_name"],
    #                 description=record["tgt_properties"].get(PROP_DESCRIPTION),
    #                 created_at=record["tgt_properties"].get(PROP_CREATED),
    #                 modified_at=record["tgt_properties"].get(PROP_MODIFIED),
    #                 properties=self.get_properties(record["tgt_properties"]),
    #                 ref=record["tgt_properties"].get(PROP_REFERENCE)
    #             ),
    #             description=record["rel_properties"].get(PROP_DESCRIPTION),
    #             created_at=record["rel_properties"].get(PROP_CREATED),
    #             modified_at=record["rel_properties"].get(PROP_MODIFIED),
    #             properties=self.get_properties(record["rel_properties"]),
    #             direction=record.get("direction"),
    #             ref=record["rel_properties"].get(PROP_REFERENCE)
    #         ) for record in results
    #     ]
    #     return relations
    
    # def vector_search_relation(self, embedding: List[float],
    #                            top_k: int = 5,
    #                            source: Optional[KGEntity] = None,
    #                            relation: Optional[str] = None,
    #                            target: Optional[KGEntity] = None,
    #                            return_score: bool = False):
    #     """
    #     Perform a vector-based nearest neighbor search on relations in Neo4j.

    #     This function queries Neo4j's vector index on relationship embeddings and retrieves 
    #     the top-K most similar relations based on cosine similarity.

    #     Args:
    #         embedding (List[float]): The embedding representation of the query relation.
    #         top_k (int): The number of top results to retrieve.
    #         source (KGEntity, optional): The source entity to filter relations. Default is None.
    #         target (KGEntity, optional): The target entity to filter relations. Default is None.
    #         return_score (bool, optional): Whether to return similarity scores alongside relations. 
    #                                     Defaults to False.

    #     Returns:
    #         List[KGRelation] or List[RelevantRelation]: 
    #             - If `return_score` is `False`, returns a list of `KGRelation` objects.
    #             - If `return_score` is `True`, returns a list of `RelevantRelation` objects, 
    #             each containing a `KGRelation` and its similarity score.

    #     Raises:
    #         neo4j.exceptions.Neo4jError: If there is an issue with the Neo4j query execution.

    #     Example:
    #         >>> query_embedding = [0.12, -0.45, 0.88, ...]  # Example relation embedding
    #         >>> results = vector_search_relation(query_embedding, top_k=5)
    #         >>> for relation in results:
    #         >>>     print(f"{relation.source.name} -[{relation.name}]-> {relation.target.name}")

    #         # If retrieving similarity scores:
    #         >>> results_with_scores = vector_search_relation(query_embedding, top_k=5, return_score=True)
    #         >>> for rel in results_with_scores:
    #         >>>     print(f"Relation: {rel.relation.name}, Score: {rel.score}")
    #     """

    #     query = textwrap.dedent(f"""\
    #     MATCH (src)-[rel{':' + relation if relation else ''}]-(tgt)
    #     WHERE rel.{PROP_EMBEDDING} IS NOT NULL
    #     """)
    #     params = {"embedding": embedding, "top_k": top_k}

    #     # Add filtering conditions for source and target
    #     filters = []
    #     if source:
    #         filters.append(
    #             "elementId(src) = $source_id" if source.id else "src.name = $source_name")
    #         params.update({"source_id": source.id, "source_name": source.name})
    #     if target:
    #         filters.append(
    #             "elementId(tgt) = $target_id" if target.id else "tgt.name = $target_name")
    #         params.update({"target_id": target.id, "target_name": target.name})
    #     if filters:
    #         query += " AND " + " AND ".join(filters)

    #     query += textwrap.dedent(f""" \
    #     RETURN elementId(rel) AS id, type(rel) AS relation, 
    #         elementId(src) AS src_id, labels(src) AS src_types, src.name AS src_name, 
    #         apoc.map.removeKey(properties(src), '{PROP_EMBEDDING}') AS src_properties, 
    #         elementId(tgt) AS tgt_id, labels(tgt) AS tgt_types, tgt.name AS tgt_name, 
    #         apoc.map.removeKey(properties(tgt), '{PROP_EMBEDDING}') AS tgt_properties, 
    #         apoc.map.removeKey(properties(rel), '{PROP_EMBEDDING}') AS rel_properties, 
    #         vector.similarity.cosine(rel.{PROP_EMBEDDING}, $embedding) AS score
    #     ORDER BY score DESC
    #     LIMIT $top_k
    #     """)

    #     results = self.run_query(query, params)

    #     # Convert results to KGRelation objects
    #     relations = [
    #         KGRelation(
    #             id=record["id"],
    #             name=record["relation"],
    #             source=KGEntity(
    #                 id=record["src_id"],
    #                 type=self.get_label(record["src_types"]),
    #                 name=record["src_name"],
    #                 description=record["src_properties"].get(PROP_DESCRIPTION),
    #                 created_at=record["src_properties"].get(PROP_CREATED),
    #                 modified_at=record["src_properties"].get(PROP_MODIFIED),
    #                 properties=self.get_properties(record["src_properties"]),
    #                 ref=record["src_properties"].get(PROP_REFERENCE)
    #             ),
    #             target=KGEntity(
    #                 id=record["tgt_id"],
    #                 type=self.get_label(record["tgt_types"]),
    #                 name=record["tgt_name"],
    #                 description=record["tgt_properties"].get(PROP_DESCRIPTION),
    #                 created_at=record["tgt_properties"].get(PROP_CREATED),
    #                 modified_at=record["tgt_properties"].get(PROP_MODIFIED),
    #                 properties=self.get_properties(record["tgt_properties"]),
    #                 ref=record["tgt_properties"].get(PROP_REFERENCE)
    #             ),
    #             description=record["rel_properties"].get(PROP_DESCRIPTION),
    #             created_at=record["rel_properties"].get(PROP_CREATED),
    #             modified_at=record["rel_properties"].get(PROP_MODIFIED),
    #             properties=self.get_properties(record["rel_properties"]),
    #             ref=record["rel_properties"].get(PROP_REFERENCE)
    #         ) for record in results
    #     ]

    #     if not return_score:
    #         return relations
    #     else:
    #         return [RelevantRelation(relation, record["score"]) for relation, record in zip(relations, results)]


    # ====================== Property Query ==================================

    def get_node_properties(self):
        """Retrieve unique properties for each entity type."""
        query = """
        MATCH (n)
        UNWIND [label IN labels(n) WHERE label <> "_Embeddable"] AS entity_type
        UNWIND [key IN keys(n) WHERE key <> "embedding"] AS property
        WITH entity_type, COLLECT(DISTINCT property) AS properties
        RETURN entity_type, properties
        ORDER BY entity_type;
        """
        results = self.run_query(query)
        return {record["entity_type"]: sorted(record["properties"]) for record in results}

    async def get_existing_node_properties_async(self, type, name):
        """Async query to retrieve existing properties of an entity."""
        query = f"""
        MATCH (e:{type} {{name: $name}})
        RETURN apoc.map.removeKey(properties(n), "embedding") AS props
        """
        async with self.async_driver.session(database=self.database) as session:
            result = await session.run(query, {"name": name})
            record = await result.single()
            return record["props"] if record else {}

    async def get_all_node_properties_async(self, node_properties, type, name):
        """Async query to retrieve all properties of an entity."""
        expected_props = node_properties.get(type, None)
        if expected_props is None:
            return {}
        existing_props = await self.get_existing_node_properties_async(type, name)
        all_props = {prop: existing_props.get(prop, None) for prop in expected_props if prop not in {
            "description", "modified_at", "created_at"}}
        return all_props

    async def get_node_properties_async(self, entity_list):
        """Run multiple property queries asynchronously."""
        node_properteis = self.get_node_properties()
        tasks = [self.get_all_node_properties_async(
            node_properteis, entity["type"], entity["name"]) for entity in entity_list]
        results = await asyncio.gather(*tasks)

        return [props for entity, props in zip(entity_list, results)]

    def get_edge_properties(self):
        """Retrieve unique properties for each relationship type."""
        query = """
        MATCH ()-[r]->()
        UNWIND [key IN keys(r) WHERE key <> "embedding"] AS property
        WITH type(r) AS relationship_type, COLLECT(DISTINCT property) AS properties
        RETURN relationship_type, properties
        ORDER BY relationship_type;
        """
        results = self.run_query(query)
        return {record["relationship_type"]: sorted(record["properties"]) for record in results}

    async def get_existing_edge_properties_async(self, src_name, edge_type, dst_name):
        """Async query to retrieve existing properties of an entity."""
        query = f"""
        MATCH (s {{name: $src_name}})-[r: {edge_type}]->(d {{name: $dst_name}})
        RETURN apoc.map.removeKey(properties(r), "embedding") AS props
        """
        async with self.async_driver.session(database=self.database) as session:
            result = await session.run(query, {"src_name": src_name, "dst_name": dst_name})
            record = await result.single()
            return record["props"] if record else {}

    async def get_all_edge_properties_async(self, edge_properties, src_name, edge_type, dst_name):
        """Async query to retrieve all properties of an entity."""
        expected_props = edge_properties.get(edge_type, None)
        if expected_props is None:
            return {}

        existing_props = await self.get_existing_edge_properties_async(src_name, edge_type, dst_name)
        all_props = {prop: existing_props.get(prop, None) for prop in expected_props if prop not in {
            "description", "confidence", "modified_at", "created_at"}}
        return all_props

    async def get_edge_properties_async(self, relation_list):
        """Run multiple property queries asynchronously."""
        edge_properties = self.get_edge_properties()
        tasks = [self.get_all_edge_properties_async(
            edge_properties, relation["src"], relation["relation"], relation["dst"]) for relation in relation_list]
        results = await asyncio.gather(*tasks)

        return [props for relation, props in zip(relation_list, results)]

    def create_vector_index(self):
        self.run_query("""CREATE VECTOR INDEX entityVector
        FOR (n:_Embeddable)
        ON n.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }}""")

    async def insert_vector(self):
        import numpy as np
        all_entities = self.get_all_nodes()
        description_list = [entity_to_text(entity) for entity in all_entities]
        batch_size = 4096
        embeddings = []
        for i in tqdm(range(0, len(description_list), batch_size), desc="Encoding Sentences"):
            batch = description_list[i: i + batch_size]
            # Store batch results
            embeddings.extend(np.array(await generate_embedding(self.emb_session, batch)))
        embeddings_np = np.vstack(embeddings)
        embeddings_np.shape

        # Set one additional label "_Embeddable" to each entity, so that
        # we can create a union vector index for them.
        self.run_query("MATCH (n) SET n :_Embeddable")

        # Store Neo4j element IDs
        node_ids = [entity.id for entity in all_entities]
        query = """
        UNWIND $data AS row
        MATCH (n) WHERE elementId(n) = row.id
        CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
        """

        batch_size = 1000
        batch = []
        batch_n = 0
        for node_id, embedding in zip(node_ids, embeddings_np):
            batch.append({"id": node_id, "embedding": embedding.tolist()})

            # Import when a batch of movies has embeddings ready; flush buffer
            if len(batch) == batch_size:
                params = {"data": batch}
                self.run_query(query, params)
                print(f'Processed batch {batch_n}.')

                batch = []
                batch_n += 1

        params = {"data": batch}
        self.run_query(query, params)


    # ====================== Upsert ==================================

    async def upsert_entity_async(self, entity: KGEntity, embedding: List[float] = [],
                                  return_entity: bool = False,
                                  semaphore=None):
        """Async function to insert or update an entity with all its properties in Neo4j."""

        params = {"name": normalize_entity(
            entity.name), PROP_EMBEDDING: embedding}

        # Include metadata fields (description, timestamps)
        params.update({
            key: getattr(entity, key.strip("_"))
            for key in [PROP_DESCRIPTION, PROP_PARAGRAPH, PROP_CREATED, PROP_MODIFIED, PROP_REFERENCE]
        })

        # Include additional entity properties
        params.update({normalize_key(k): normalize_value(v)
                      for k, v in entity.properties.items()})

        # Construct SET clause dynamically (excluding embedding)
        set_clause = ", ".join(f"n.{key} = ${key}" for key in params if key not in {
                               PROP_CREATED, PROP_MODIFIED, PROP_EMBEDDING})
        set_clause += (", " if set_clause else "") + \
            f"n.{PROP_MODIFIED} = datetime(${PROP_MODIFIED})"

        # Use MATCH for updates, CREATE for new insertions
        if entity.id:
            query = textwrap.dedent(f"""\
            MATCH (n:{normalize_entity_type(entity.type)}) 
            WHERE elementId(n) = $id
            SET {set_clause} WITH n
            CALL db.create.setNodeVectorProperty(n, '{PROP_EMBEDDING}', ${PROP_EMBEDDING})""") + \
                (f" RETURN elementId(n) AS id, labels(n) AS labels, n.name AS name, apoc.map.removeKey(properties(n), '{PROP_EMBEDDING}') AS properties" if return_entity else "")

            params["id"] = entity.id  # Include ID in query parameters
        else:  # The new insertions may have the same entity name but different props
            query = textwrap.dedent(f"""\
            CREATE (n:{normalize_entity_type(entity.type)}:{TYPE_EMBEDDABLE})
            SET {set_clause}, n.{PROP_CREATED} = COALESCE(n.{PROP_CREATED}, datetime(${PROP_CREATED})) WITH n
            CALL db.create.setNodeVectorProperty(n, '{PROP_EMBEDDING}', ${PROP_EMBEDDING})""") + \
                (f" RETURN elementId(n) AS id, labels(n) AS labels, n.name AS name, apoc.map.removeKey(properties(n), '{PROP_EMBEDDING}') AS properties" if return_entity else "")

        results = await self.run_query_async(query, params, semaphore)

        if return_entity and results:
            record = results[0]
            return KGEntity(
                id=record["id"],
                type=self.get_label(record["labels"]),
                name=record["name"],
                description=record["properties"].get(PROP_DESCRIPTION),
                paragraph=record["properties"].get(PROP_PARAGRAPH),
                created_at=record["properties"].get(PROP_CREATED),
                modified_at=record["properties"].get(PROP_MODIFIED),
                properties=self.get_properties(record["properties"]),
                ref=record["properties"].get(PROP_REFERENCE)
            )

    async def upsert_relation_async(self, relation: KGRelation, embedding: List[float] = [],
                                    return_relation: bool = False,
                                    semaphore=None):
        """Async function to insert or update a relationship with all its properties in Neo4j."""
        if not relation:
            return None
        assert relation.source is not None, "Source entity cannot be None!"
        assert relation.target is not None, "Target entity cannot be None!"

        params = {
            "src_name": normalize_entity(relation.source.name),
            "src_id": relation.source.id,
            "tgt_name": normalize_entity(relation.target.name),
            "tgt_id": relation.target.id,
            PROP_EMBEDDING: embedding
        }

        # Include metadata fields (description, timestamps)
        params.update({
            key: getattr(relation, key.strip("_"))
            for key in [PROP_DESCRIPTION, PROP_PARAGRAPH, PROP_CREATED, PROP_MODIFIED, PROP_REFERENCE]
        })

        # Include additional relation properties
        params.update({normalize_key(k): normalize_value(v)
                      for k, v in relation.properties.items()})

        # Construct SET clause dynamically (excluding embedding)
        set_clause = ", ".join(f"rel.{key} = ${key}" for key in params if key not in
                               {"src_name", "src_id", "tgt_name", "tgt_id", PROP_CREATED, PROP_MODIFIED, PROP_EMBEDDING})
        set_clause += ("," if set_clause else "") + \
            f"rel.{PROP_MODIFIED} = datetime(${PROP_MODIFIED})"

        # Use MATCH for updates, CREATE for new insertions
        if relation.id:
            query = textwrap.dedent(f"""\
            MATCH (src)-[rel]->(tgt) 
            WHERE elementId(rel) = $id
            SET {set_clause}, rel.{PROP_CREATED} = COALESCE(rel.{PROP_CREATED}, datetime(${PROP_CREATED})) WITH rel
            CALL db.create.setRelationshipVectorProperty(rel, '{PROP_EMBEDDING}', ${PROP_EMBEDDING})""") + \
                (f" RETURN elementId(rel) AS id, type(rel) AS relation, apoc.map.removeKey(properties(rel), '{PROP_EMBEDDING}') AS properties" if return_relation else "")

            params["id"] = relation.id  # Use ID when updating an entity
        else:  # The new insertions may have the same entity name but different props
            if not relation.name:
                return None
            
            query = textwrap.dedent(f"""\
            MATCH (src) WHERE elementId(src) = $src_id
            MATCH (tgt) WHERE elementId(tgt) = $tgt_id
            CREATE (src)-[rel:{normalize_relation(relation.name)}]->(tgt)
            SET {set_clause}, rel.{PROP_CREATED} = COALESCE(rel.{PROP_CREATED}, datetime(${PROP_CREATED})) WITH rel
            CALL db.create.setRelationshipVectorProperty(rel, '{PROP_EMBEDDING}', ${PROP_EMBEDDING})""") + \
                (f" RETURN elementId(rel) AS id, type(rel) AS relation, apoc.map.removeKey(properties(rel), '{PROP_EMBEDDING}') AS properties" if return_relation else "")
        results = await self.run_query_async(query, params, semaphore)

        if return_relation and results:
            record = results[0]
            return KGRelation(
                id=record["id"],
                name=record["relation"],
                source=relation.source,
                target=relation.target,
                description=record["properties"].get(PROP_DESCRIPTION),
                paragraph=record["properties"].get(PROP_PARAGRAPH),
                created_at=record["properties"].get(PROP_CREATED),
                modified_at=record["properties"].get(PROP_MODIFIED),
                properties=self.get_properties(record["properties"]),
                ref=record["properties"].get(PROP_REFERENCE)
            )

    async def upsert_entities(self, entities_dict: Dict[str, KGEntity]) -> Dict[str, KGEntity]:
        """
        Async function to insert a set of entities along with their embeddings into Neo4j.

        Args:
            entities_dict (Dict[str, KGEntity]): A dictionary of KG entities being inserted. The key can be anything (typically the name of each entity).

        Returns:
            Dict[str, KGEntity]: An updated dictionary of KG entities inserted.
        """
        texts = [entity_to_text(entity) for entity in entities_dict.values()]
        embeddings = await generate_embedding(self.emb_session, texts)

        semaphore = asyncio.Semaphore(50)
        # Insert Entities
        entity_tasks = [self.upsert_entity_async(entity, embedding, return_entity=True, semaphore=semaphore)
                        for entity, embedding in zip(entities_dict.values(), embeddings)]
        entities = await asyncio.gather(*entity_tasks)

        # Insert Entity Schema
        await self.add_entity_schema(entities_dict)

        return {
            key: entity for key, entity in zip(entities_dict, entities)
        }

    async def upsert_relations(self, relations_dict: Dict[str, KGRelation]) -> Dict[str, KGRelation]:
        """
        Async function to insert a set of relations along with their embeddings into Neo4j.

        Args:
            relations_dict (Dict[str, KGRelation]): A dictionary of KG relations being inserted. The key can be anything (typically the name of each entity).

        Returns:
            Dict[str, KGEntity]: An updated dictionary of KG entities inserted.
        """
        texts = [relation_to_text(relation) for relation in relations_dict.values()]
        embeddings = await generate_embedding(self.emb_session, texts)

        semaphore = asyncio.Semaphore(50)
        # Insert Entities
        relation_tasks = [self.upsert_relation_async(relation, embedding, return_relation=True, semaphore=semaphore)
                          for relation, embedding in zip(relations_dict.values(), embeddings)]
        relations = await asyncio.gather(*relation_tasks)

        # Insert Relation Schema
        await self.add_relation_schema(relations_dict)

        return {
            key: relation for key, relation in zip(relations_dict, relations)
        }
    
kg_driver = KG_Driver()