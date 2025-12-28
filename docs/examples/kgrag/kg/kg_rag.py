"""Refactored KG-RAG Component using Mellea patterns."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Dict

from mellea import MelleaSession
from mellea.stdlib.base import Component, Context, ModelOutputThunk
from mellea.stdlib.sampling.rejection_sampling import RejectionSamplingStrategy
from mellea.stdlib.requirement import Requirement

from kg.kg_driver import kg_driver
from kg.kg_rep import (
    KGEntity,
    KGRelation,
    RelevantEntity,
    RelevantRelation,
    relation_to_text,
    normalize_entity,
    entity_to_text,
)
from kg_generative import (
    break_down_question,
    extract_topic_entities,
    align_topic_entities,
    prune_relations,
    prune_triplets,
    evaluate_knowledge_sufficiency,
    validate_consensus,
    generate_direct_answer,
)
from kg_requirements import get_requirements_for_task
from utils.logger import BaseProgressLogger, DefaultProgressLogger
from utils.utils import generate_embedding
from utils.prompt_list import get_default_prompts


PROMPTS = get_default_prompts()


@dataclass
class Query:
    """Query with optional sub-objectives."""
    query: str
    query_time: datetime = None
    subqueries: Optional[List[str]] = None


class KGRagComponent(Component):
    """Knowledge Graph-Enhanced RAG component using Mellea patterns.

    This component performs multi-hop reasoning over a knowledge graph to answer
    complex questions. It uses Mellea's @generative functions, Requirements,
    and Sampling Strategies for robust, composable question answering.
    """

    def __init__(
        self,
        session: MelleaSession,
        eval_session: MelleaSession,
        emb_session: Any,
        domain: str = "movie",
        config: Optional[Dict] = None,
        logger: Optional[BaseProgressLogger] = None,
        **kwargs,
    ):
        """Initialize KG-RAG component.

        Args:
            session: Mellea session for main LLM calls
            eval_session: Mellea session for evaluation
            emb_session: Session/model for embeddings
            domain: Knowledge domain (e.g., 'movie', 'finance')
            config: Configuration dict with 'route', 'width', 'depth'
            logger: Logger for progress tracking
        """
        super().__init__()
        self.session = session
        self.eval_session = eval_session
        self.emb_session = emb_session
        self.domain = domain
        self.logger = logger or DefaultProgressLogger()

        self.config = {"route": 5, "width": 30, "depth": 3}
        if config:
            self.config.update(config)

        self.route = self.config["route"]
        self.width = self.config["width"]
        self.depth = self.config["depth"]

        self.logger.info(f"KGRagComponent initialized with config: {self.config}")

    async def break_down_question_with_requirements(
        self, query: Query
    ) -> List[Query]:
        """Break down question into solving routes with validation.

        Uses Mellea's Requirements and RejectionSampling for robustness.
        """
        hints = PROMPTS.get("domain_hints", {}).get(self.domain, "No hints available.")

        # Use the generative function with requirements
        requirements = get_requirements_for_task("break_down")
        strategy = RejectionSamplingStrategy(loop_budget=3)

        result = await break_down_question(
            query=query.query,
            query_time=str(query.query_time),
            domain=self.domain,
            route=self.route,
            hints=hints,
            # Mellea will automatically handle requirements and strategy
        )

        # Convert to Query objects
        queries = []
        for route in result.routes:
            queries.append(
                Query(
                    query=query.query, query_time=query.query_time, subqueries=route
                )
            )

        self.logger.info(f"Broke down question into {len(queries)} routes")
        return queries

    async def extract_entity_with_validation(self, query: Query) -> List[str]:
        """Extract topic entities with validation."""
        requirements = get_requirements_for_task("extract_entity")
        strategy = RejectionSamplingStrategy(loop_budget=3)

        result = await extract_topic_entities(
            query=query.query,
            query_time=str(query.query_time),
            route=query.subqueries,
            domain=self.domain,
        )

        entities_list = [normalize_entity(entity) for entity in result.entities]
        self.logger.info(f"Extracted {len(entities_list)} topic entities")
        return entities_list

    async def align_topic(
        self, query: Query, topic_entities: List[str], top_k: int = 45
    ) -> List[RelevantEntity]:
        """Align topic entities with KG entities using MIPS."""
        norm_coeff = 1 / len(topic_entities) if len(topic_entities) > 0 else 1

        # Generate embeddings
        embeddings = await generate_embedding(
            self.emb_session, topic_entities, logger=self.logger
        )

        async def align_one_topic(idx, topic):
            # Exact match
            exact_match = kg_driver.get_entities(
                name=topic, top_k=min(4, top_k // 2), fuzzy=True
            )
            top_k_entities = exact_match[: min(top_k, len(exact_match))]

            # Similarity match
            if len(top_k_entities) < top_k:
                similar_match = kg_driver.get_entities(
                    embedding=embeddings[idx],
                    top_k=top_k - len(top_k_entities),
                    return_score=True,
                )
                top_k_entities.extend(
                    [
                        relevant_entity.entity
                        for relevant_entity in similar_match
                        if relevant_entity.entity not in top_k_entities
                    ]
                )

            # Format entities
            top_k_entities_dict = {
                f"ent_{i}": entity for i, entity in enumerate(top_k_entities)
            }
            top_k_entities_str = "\n".join(
                f"{key}: {entity_to_text(entity)}"
                for key, entity in top_k_entities_dict.items()
            )

            # Use generative function with requirements
            requirements = get_requirements_for_task("align_topic")
            result = await align_topic_entities(
                query=query.query,
                query_time=str(query.query_time),
                route=query.subqueries,
                domain=self.domain,
                top_k_entities_str=top_k_entities_str,
            )

            # Convert to RelevantEntity objects
            return [
                RelevantEntity(top_k_entities_dict[ind], norm_coeff * float(score))
                for ind, score in result.relevant_entities.items()
                if (float(score) > 0) and (ind in top_k_entities_dict)
            ]

        # Run in parallel
        tasks = [
            align_one_topic(idx, topic) for idx, topic in enumerate(topic_entities)
        ]
        results = await asyncio.gather(*tasks)

        ans = []
        for result in results:
            ans.extend(result)

        self.logger.info(f"Aligned {len(ans)} relevant entities")
        return ans

    async def relation_search_prune(
        self, query: Query, entity: KGEntity
    ) -> List[RelevantRelation]:
        """Prune relations from an entity using LLM."""
        relation_list = kg_driver.get_relations(entity, unique_relation=True)
        if len(relation_list) == 0:
            return []

        entity_str = entity_to_text(entity)
        unique_relations_dict = {}
        for i, relation in enumerate(relation_list):
            relation.target = KGEntity(id="", type=relation.target.type, name="")
            relation.properties = {}
            unique_relations_dict[f"rel_{i}"] = relation

        unique_relations_str = "\n".join(
            [
                f"{key}: {relation_to_text(relation, include_des=False, include_src_des=False, include_src_prop=False, property_key_only=True)}"
                for key, relation in unique_relations_dict.items()
            ]
        )

        hints = PROMPTS.get("domain_hints", {}).get(self.domain, "No hints available.")

        # Use generative function
        result = await prune_relations(
            query=query.query,
            query_time=str(query.query_time),
            route=query.subqueries,
            domain=self.domain,
            entity_str=entity_str,
            relations_str=unique_relations_str,
            width=self.width,
            hints=hints,
        )

        return [
            RelevantRelation(unique_relations_dict[ind], float(score))
            for ind, score in result.relevant_relations.items()
            if (float(score) > 0) and (ind in unique_relations_dict)
        ]

    async def triplet_prune(
        self,
        query: Query,
        relevant_relation: RelevantRelation,
        triplet_candidates: List[KGRelation],
    ) -> List[RelevantRelation]:
        """Prune triplets using LLM."""
        triplet_dict = {
            f"rel_{i}": triplet
            for i, triplet in enumerate(
                triplet_candidates[: min(self.width, len(triplet_candidates))]
            )
        }
        source = list(triplet_dict.values())[0].source
        entity_str = entity_to_text(source)
        relations_str = "\n".join(
            [
                f"{key}: {relation_to_text(triplet, include_src_des=False, include_src_prop=False)}"
                for key, triplet in triplet_dict.items()
            ]
        )

        if len(triplet_dict) < len(triplet_candidates):
            relations_str += f"\n...({len(triplet_candidates) - len(triplet_dict)} relation(s) truncated)"

        hints = PROMPTS.get("domain_hints", {}).get(self.domain, "No hints available.")

        # Use generative function
        result = await prune_triplets(
            query=query.query,
            query_time=str(query.query_time),
            route=query.subqueries,
            domain=self.domain,
            entity_str=entity_str,
            relations_str=relations_str,
            hints=hints,
        )

        return [
            RelevantRelation(
                triplet_dict[ind], relevant_relation.score * float(score)
            )
            for ind, score in result.relevant_relations.items()
            if (float(score) > 0) and (ind in triplet_dict)
        ]

    def triplet_sort(
        self, total_relevant_triplets: List[RelevantRelation]
    ) -> tuple[bool, List[str], List[RelevantRelation]]:
        """Sort and filter triplets by relevance score."""
        total_relevant_triplets = sorted(
            total_relevant_triplets, key=lambda x: x.score, reverse=True
        )[: self.width]
        filtered_relevant_triplets = [
            triplet for triplet in total_relevant_triplets if triplet.score > 0
        ]

        cluster_chain_of_entities = [
            relation_to_text(triplet.relation)
            for triplet in filtered_relevant_triplets
        ]

        return (
            len(filtered_relevant_triplets) != 0,
            cluster_chain_of_entities,
            filtered_relevant_triplets,
        )

    async def reasoning(
        self, route: Query, topic_entities: List, cluster_chain_of_entities: List
    ) -> tuple[bool, str, str]:
        """Evaluate if knowledge is sufficient to answer."""
        entities_str = "\n".join(
            [
                f"ent_{idx}: {entity_to_text(entity)}"
                for idx, entity in enumerate(topic_entities)
            ]
        )
        entities_str = entities_str if entities_str else "None"

        idx = 0
        triplets = []
        for sublist in cluster_chain_of_entities:
            for chain in sublist:
                triplets.append(f"rel_{idx}: {chain}")
                idx += 1
        triplets_str = "\n".join(triplets)
        triplets_str = triplets_str if triplets_str else "None"

        hints = PROMPTS.get("domain_hints", {}).get(self.domain, "No hints available.")

        # Use generative function
        result = await evaluate_knowledge_sufficiency(
            query=route.query,
            query_time=str(route.query_time),
            route=route.subqueries,
            domain=self.domain,
            entities=entities_str,
            triplets=triplets_str,
            hints=hints,
        )

        return (
            result.sufficient.lower().strip().replace(" ", "") == "yes",
            result.reason,
            result.answer,
        )

    async def execute(
        self,
        query: str,
        query_time: Optional[datetime] = None,
        return_details: bool = False,
        precomputed_routes: Optional[List[Query]] = None,
    ) -> str | tuple[str, List[Dict]]:
        """Execute KG-RAG pipeline to answer a query.

        Args:
            query: The question to answer
            query_time: Optional timestamp for temporal reasoning
            return_details: Whether to return detailed route results
            precomputed_routes: Optional pre-computed solving routes

        Returns:
            The answer string, or (answer, route_details) if return_details=True
        """
        query_obj = Query(query=query, query_time=query_time)

        # Generate query embedding once
        query_embedding = (
            await generate_embedding(
                self.emb_session, [query_obj.query], logger=self.logger
            )
        )[0]

        # Break down question or use precomputed routes
        if precomputed_routes:
            queries = precomputed_routes
        else:
            queries = await self.break_down_question_with_requirements(query_obj)

        # Define route exploration logic
        async def explore_one_route(route):
            topic_entities = await self.extract_entity_with_validation(route)
            self.logger.info(f"Extracted topic entities: {topic_entities}")

            topic_entities_scores = await self.align_topic(route, topic_entities)

            ans = ""
            cluster_chain_of_entities = []
            initial_topic_entities = [
                relevant_entity.entity for relevant_entity in topic_entities_scores
            ]

            all_entities = {}
            all_relations = {}
            for relevant_entity in topic_entities_scores:
                relevant_entity.step = 0
                all_entities[relevant_entity.entity.id] = relevant_entity

            # Initial reasoning
            stop, reason, answer = await self.reasoning(
                route, initial_topic_entities, [[]]
            )
            if stop:
                self.logger.info("ToG stopped at depth 0.")
                ans = answer
            else:
                # Multi-hop traversal
                for depth in range(1, self.depth + 1):
                    # Relation search and pruning (parallel)
                    tasks = [
                        self.relation_search_prune(route, entity_score.entity)
                        for entity_score in topic_entities_scores
                        if entity_score.entity is not None
                    ]
                    results = await asyncio.gather(*tasks)

                    relevant_relations_list = []
                    for entity_score, relevant_relations in zip(
                        topic_entities_scores, results
                    ):
                        relevant_relations_list.extend(
                            [
                                RelevantRelation(
                                    relation=relevant_relation.relation,
                                    score=relevant_relation.score * entity_score.score,
                                )
                                for relevant_relation in relevant_relations
                            ]
                        )

                    # Triplet pruning (parallel)
                    tasks = []
                    for relevant_relation in relevant_relations_list:
                        triplet_candidates = kg_driver.get_relations(
                            source=relevant_relation.relation.source,
                            relation=relevant_relation.relation.name,
                            target_type=relevant_relation.relation.target.type,
                            target_embedding=query_embedding,
                        )

                        # Filter visited triplets
                        triplet_candidates = [
                            triplet
                            for triplet in triplet_candidates
                            if triplet.id not in all_relations
                        ]

                        if len(triplet_candidates) == 0:
                            continue

                        tasks.append(
                            self.triplet_prune(
                                route, relevant_relation, triplet_candidates
                            )
                        )

                    results = await asyncio.gather(*tasks)
                    total_relevant_triplets = sum(results, [])

                    flag, chain_of_entities, filtered_relevant_triplets = (
                        self.triplet_sort(total_relevant_triplets)
                    )
                    cluster_chain_of_entities.append(chain_of_entities)

                    # Update scores and prepare for next depth
                    norm_coeff = sum(
                        triplet.score for triplet in filtered_relevant_triplets
                    )
                    norm_coeff = 1 / norm_coeff if norm_coeff > 0 else 1
                    topic_entities_scores_dict = {}
                    for triplet in filtered_relevant_triplets:
                        last = topic_entities_scores_dict.setdefault(
                            triplet.relation.target.id,
                            RelevantEntity(triplet.relation.target, 0),
                        )
                        topic_entities_scores_dict[triplet.relation.target.id] = (
                            RelevantEntity(
                                triplet.relation.target,
                                triplet.score * norm_coeff + last.score,
                            )
                        )
                    topic_entities_scores = list(topic_entities_scores_dict.values())

                    # Track visited entities and relations
                    for relevant_relation in filtered_relevant_triplets:
                        relevant_relation.relation.step = depth
                        all_relations[relevant_relation.relation.id] = relevant_relation
                    for relevant_entity in topic_entities_scores:
                        relevant_entity.step = depth
                        all_entities[relevant_entity.entity.id] = relevant_entity

                    # Check if we can answer
                    if flag:
                        stop, reason, answer = await self.reasoning(
                            route, initial_topic_entities, cluster_chain_of_entities
                        )
                        if stop:
                            self.logger.info(f"ToG stopped at depth {depth}.")
                            ans = answer
                            break
                        else:
                            self.logger.info(
                                f"Depth {depth} still not sufficient to answer."
                            )
                            ans = reason
                    else:
                        self.logger.info(
                            f"No new knowledge added at depth {depth}, stopping."
                        )
                        _, _, ans = await self.reasoning(
                            route, initial_topic_entities, cluster_chain_of_entities
                        )
                        break

            # Format context
            entities_str = "\n".join(
                [
                    f"ent_{idx}: {entity_to_text(entity)}"
                    for idx, entity in enumerate(initial_topic_entities)
                ]
            )
            entities_str = entities_str if entities_str else "None"

            idx = 0
            triplets = []
            for sublist in cluster_chain_of_entities:
                for chain in sublist:
                    triplets.append(f"rel_{idx}: {chain}")
                    idx += 1
            triplets_str = "\n".join(triplets)
            triplets_str = triplets_str if triplets_str else "None"

            return {
                "query": route,
                "context": "Knowledge Entities:\n"
                + entities_str
                + "\n"
                + "Knowledge Triplets:\n"
                + triplets_str,
                "ans": f'"{ans}". {reason}',
                "entities": list(all_entities.values()),
                "relations": list(all_relations.values()),
            }

        # Run first few routes in parallel, plus direct answer
        tasks = [
            generate_direct_answer(
                query=query_obj.query,
                query_time=str(query_obj.query_time),
                domain=self.domain,
            ),
            explore_one_route(queries[0]),
            explore_one_route(queries[1]) if len(queries) > 1 else None,
        ]
        tasks = [t for t in tasks if t is not None]

        results = await asyncio.gather(*tasks)
        direct_result = results[0]
        attempt = f'"{direct_result.answer}". {direct_result.reason}'

        route_results = results[1:]

        # Explore remaining routes with validation
        stop = False
        final = ""
        for route in queries[2:]:
            route_results.append(await explore_one_route(route))
            if len(route_results) >= 2:
                # Build routes info string
                routes_info = f"\nWe have identified {len(queries)} solving route(s) below, and have {len(queries) - len(route_results)} unexplored solving route left.:\n"
                for idx in range(len(route_results)):
                    routes_info += (
                        f"Route {idx + 1}: {queries[idx].subqueries}\n"
                        + "Reference: "
                        + route_results[idx]["context"]
                        + "\n"
                        + "Answer: "
                        + route_results[idx]["ans"]
                        + "\n\n"
                    )
                for idx in range(len(route_results), len(queries)):
                    routes_info += f"Route {idx + 1}: {queries[idx].subqueries}\n\n"

                hints = PROMPTS.get("domain_hints", {}).get(
                    self.domain, "No hints available."
                )

                # Validate consensus
                validation_result = await validate_consensus(
                    query=query_obj.query,
                    query_time=str(query_obj.query_time),
                    domain=self.domain,
                    attempt=attempt,
                    routes_info=routes_info,
                    hints=hints,
                )

                stop = (
                    validation_result.judgement.lower().strip().replace(" ", "")
                    == "yes"
                )
                final = validation_result.final_answer
                if stop:
                    self.logger.info(f"Consensus reached: {final}")
                    break

        if not stop:
            final = attempt

        if return_details:
            return final, route_results
        else:
            return final
