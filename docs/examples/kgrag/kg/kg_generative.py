"""Generative functions for KG-RAG using Mellea's @generative decorator."""
import textwrap
from typing import List
from mellea.stdlib.genslot import generative
from kg_models import (
    QuestionRoutes,
    TopicEntities,
    RelevantEntities,
    RelevantRelations,
    EvaluationResult,
    ValidationResult,
    DirectAnswer,
)


@generative
async def break_down_question(
    query: str,
    query_time: str,
    domain: str,
    route: int,
    hints: str
) -> QuestionRoutes:
    """Break down a complex question into multiple solving routes.

    You are a helpful assistant who is good at answering questions in the {domain} domain
    by using knowledge from an external knowledge graph. Before answering the question,
    you need to break down the question so that you may look for the information from
    the knowledge graph in a step-wise operation.

    There can be {route} possible routes to break down the question. Order them by efficiency.
    Return your reasoning and sub-objectives as: {{"reason": "...", "routes": [[...], [...]]}}

    Domain-specific Hints: {hints}

    Question: {query}
    Query Time: {query_time}
    """
    pass


@generative
async def extract_topic_entities(
    query: str,
    query_time: str,
    route: List[str],
    domain: str
) -> TopicEntities:
    """Extract topic entities from a query for knowledge graph search.

    You are presented with a question in the {domain} domain, its query time, and a solving route.

    1) Determine the topic entities asked in the query and each step in the solving route.
    2) Extract those topic entities into a list: {{"entities": ["entity1", "entity2", ...]}}

    Consider extracting entities in an informative way, combining adjectives or surrounding
    information. Include entity types explicitly for precise search.

    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
    """
    pass


@generative
async def align_topic_entities(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    top_k_entities_str: str
) -> RelevantEntities:
    """Align extracted entities with knowledge graph entities and score relevance.

    You are presented with a question in the {domain} domain and entities from the KG.
    Score ALL POSSIBLE entities relevant to answering the question on a scale 0-1.
    The sum of scores should equal 1.

    Entities: {top_k_entities_str}

    Return: {{"reason": "...", "relevant_entities": {{"ent_0": 0.6, "ent_1": 0.4}}}}

    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
    """
    pass


@generative
async def prune_relations(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    entity_str: str,
    relations_str: str,
    width: int,
    hints: str
) -> RelevantRelations:
    """Prune and score relations from an entity for relevance to the query.

    You are given a question in the {domain} domain and a list of relations from an entity.
    Retrieve up to {width} relations that contribute to answering the question.
    Rate their relevance from 0 to 1 (sum of scores = 1).

    Entity: {entity_str}
    Relations: {relations_str}

    Domain-specific Hints: {hints}

    Return: {{"reason": "...", "relevant_relations": {{"rel_0": 0.7, "rel_1": 0.3}}}}

    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
    """
    pass


@generative
async def prune_triplets(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    entity_str: str,
    relations_str: str,
    hints: str
) -> RelevantRelations:
    """Score triplet relevance for answering the query.

    You are presented with a question in the {domain} domain and directed relations
    from a source entity. Score the relations' contribution to answering the question
    on a scale from 0 to 1 (sum = 1).

    Source Entity: {entity_str}
    Relations: {relations_str}

    Domain-specific Hints: {hints}

    Return: {{"reason": "...", "relevant_relations": {{"rel_0": 1.0}}}}

    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
    """
    pass


@generative
async def evaluate_knowledge_sufficiency(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    entities: str,
    triplets: str,
    hints: str
) -> EvaluationResult:
    """Evaluate if retrieved knowledge is sufficient to answer the question.

    You are presented with a question in the {domain} domain and retrieved entities/triplets
    from a noisy knowledge graph. Determine whether these references are sufficient to
    answer the question (Yes or No).

    - If yes, answer the question using fewer than 50 words.
    - If no, respond with 'I don't know'.

    If multiple conflicting candidates exist, use the one with stronger supporting evidence.

    Domain-specific Hints: {hints}

    Knowledge Entities: {entities}
    Knowledge Triplets: {triplets}

    Return: {{"sufficient": "Yes/No", "reason": "...", "answer": "..."}}

    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
    """
    pass


@generative
async def validate_consensus(
    query: str,
    query_time: str,
    domain: str,
    attempt: str,
    routes_info: str,
    hints: str
) -> ValidationResult:
    """Validate consensus among multiple solving routes.

    You are presented with a question in the {domain} domain and answers from multiple routes.
    Act as a rigorous judge to determine if answers reach consensus.

    Consensus = at least half of the answers (including attempt) agree on a specific answer.

    Strategy:
    1. If consensus exists → respond "Yes" and provide final answer
    2. If no consensus and unexplored routes remain → respond "No"
    3. If no consensus and all routes explored → respond "Yes" with best answer
    4. If all routes say "I don't know" → fall back to attempt

    You are rewarded for correct answers, penalized for wrong answers, no penalty for "I don't know".

    Domain-specific Hints: {hints}

    Question: {query}
    Query Time: {query_time}
    Attempt: {attempt}
    {routes_info}

    Return: {{"judgement": "Yes/No", "final_answer": "..."}}
    """
    pass


@generative
async def generate_direct_answer(
    query: str,
    query_time: str,
    domain: str
) -> DirectAnswer:
    """Generate answer directly without knowledge graph (baseline).

    You are provided with a question in the {domain} domain and its query time.
    Determine whether your knowledge is sufficient to answer the question (Yes or No).

    - If yes, answer succinctly using the fewest words possible.
    - If no, respond with 'I don't know'.

    Provide reasoning and supporting evidence from your knowledge.

    Return: {{"sufficient": "Yes/No", "reason": "...", "answer": "..."}}

    Question: {query}
    Query Time: {query_time}
    """
    pass
