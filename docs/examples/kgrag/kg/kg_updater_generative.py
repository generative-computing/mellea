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
    reference: str,
    entity_types: str = "",
    relation_types: str = ""
) -> ExtractionResult:
    """
    ## 1. Overview
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Try to capture as much information from the text as possible without sacrificing accuracy.
    Do not add any information that is not explicitly mentioned in the text. The text document will only be provided to you ONCE. After reading it, both you and we will no longer have access to it (like a closed-book exam).
    Therefore, extract all self-contained information needed to reconstruct the knowledge. Do NOT use vague pronouns like "this", "that", or "it" to refer to prior context in the text. Always use full, explicit names or phrases that can stand alone.
    - **Nodes** represent entities and concepts.
    - The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible to a vast audience.
    ## 2. Labeling Nodes
    - **Consistency**: Ensure you use available types for node labels. Ensure you use basic or elementary types for node labels.
    - For example, when you identify an entity representing a person, always label it as **'person'**. Avoid using more specific terms like 'mathematician' or 'scientist'.
    - **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
    - **Relationships** represent connections between entities or concepts. Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary type such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!
    ## 3. Coreference Resolution
    - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency. If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
    always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID. Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
    ## 4. Strict Compliance
    Adhere to the rules strictly. Non-compliance will result in termination.

    -Goal-
    Given a text document, identify all entities from the text and all relationships among the identified entities.

    -Steps-
    1. Identify all entities. For each identified entity, extract its type, name, description, and properties.
    - type: One of the following types, but not limited to: [{entity_types}]. Please refrain from creating a new entity type, always try to fit the entity to one of the provided types first.
    - name: Name of the entity, use the same language as input text. If English, capitalize the name.
    - description: Comprehensive and general description (under 50 words) of the entity.
    - supporting_paragraph: Provide two short anchors—paragraph_start and paragraph_end—taken verbatim from the same paragraph that supports the entity mention.
	• Each anchor must be 5–30 characters long.
	• Copy exactly from the source paragraph (case, punctuation, whitespace).
	• paragraph_start must be the first 5–30 chars of that paragraph; paragraph_end must be the last 5–30 chars of that paragraph.
	• Choose the most informative paragraph that (a) contains the entity's full name and (b) contributes evidence for its description or properties.
	• If the entity appears in multiple paragraphs, prefer the earliest paragraph that satisfies (a) and (b).
	• Do not include ellipses or added characters; the anchors must be direct substrings of the paragraph.
    - properties: Entity properties are key-value pairs modeling special relations where an entity has **only one valid value at any point in its lifetime**. These properties **do not change frequently**.
      - Each type of entity can have a distinct set of properties.
      - If any properties were not mentioned in the text, please skip them.
      - Only include those properties with a **valid value**.
      - Example entity properties: A person-typed entity may have a birthday and nationality. A movie-typed entity may have a release date and language. What they have in common is that they tend to have one valid value at any point in their lifetime.
    Format each entity as a list of 3 string elements and a set of key-value pairs: \
    ["type", "name", "description", "<paragraph_start>", "<paragraph_end>", {{"key": "val", ...}}], assign this list to a key named "ent_i", where i is the entity index.

    2. Among the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other and extract their description and potential properties.
    - source_entity_name: name of the source entity, *MUST BE* one of the entity names identified in step 1 (the "name").
    - relation_name: up to *three words* as a predicate describing the general relationship between the source entity and target entity, capitalized and joined with underscores (e.g., [{relation_types}]).
    - target_entity_name: name of the target entity, *MUST BE* one of the entity names identified in step 1 (the "name").
    - description: short and concise explanation as to why you think the source entity and the target entity are related to each other
    - supporting_paragraph: Provide two short anchors—paragraph_start and paragraph_end—taken verbatim from the same paragraph that supports the relationship mention.
	• Each anchor must be 5–30 characters long.
	• Copy exactly from the source paragraph (case, punctuation, whitespace).
	• paragraph_start must be the first 5–30 chars of that paragraph; paragraph_end must be the last 5–30 chars of that paragraph.
	• Choose the most informative paragraph that (a) contains the entity's full name and (b) contributes evidence for its description or properties.
	• If the entity appears in multiple paragraphs, prefer the earliest paragraph that satisfies (a) and (b).
	• Do not include ellipses or added characters; the anchors must be direct substrings of the paragraph.
    - relation_properties: Relation properties are special complement parts of relations, they store information that is not manifest by the relation name alone.
        - Each type of relation can have a distinct set of properties.
        - Example relation properties: A WORK_IN relation may have an occupation. A HAS_POPULATION relation may have the value of the population.
    Format each relationship as a list of 4 string elements and a set of key-value pairs: \
    ["source_entity_name", "relation_name", "target_entity_name", "description", "<paragraph_start>", "<paragraph_end>", {{"key": "val", ...}}], assign this list to a key named "rel_i", where i is the relation index.

    To better extract relations, please follow these two sub-steps exactly.
    a. Identify **exclusive relations that evolve over time** (time-sensitive exclusivity). These relationships should be extracted as **temporal relations** instead of properties.
    - If a relationship **can change over time but only one value is valid at any given moment**, it must be modeled as a **temporal relationship with timestamps**. Example relationships include:
     - A person works at only one company at a time: (Person: JOHN)-[WORKS_AT, props: {{valid_from: 2019-01-01, valid_until: 2021-06-01}}]->(Company: IBM).
     - A person resides in only one place at a time: (Person: LISA)-[LIVES_IN, props: {{valid_from: 2021-03-14, valid_until: None}}]->(Geo: BOSTON).
     - A geographic region has a population that changes over time: (Geo: UNITED STATES)-[HAS_POPULATION, props: {{valid_from: 2025, valid_until: None, population: 340.1 million}}]->(Geo: UNITED STATES).
    - These relationships should be formatted as a list of 4 string elements and a set of key-value pairs: ["source_entity", "relation_name", "target_entity", "relation_description", {{"valid_from": "YYYY-MM-DD", "valid_until": "YYYY-MM-DD", "key": "val", ...}}].

    b. Identify **accumulative relations** (non-exclusive relationships). These relations **do not need deprecation** and can have multiple values coexisting. Example relationships include:
    - Actors can act in multiple movies: (Person: AMY)-[ACTED_IN, props: {{character: Anna, year: 2019}}]->(Movie: A GOOD MOVIE).
    - A person can have multiple skills: (Person: AMY)-[HAS_SKILL, props: {{skill: jogging}}]->(Person: AMY).
    - A person can have multiple friends: (Person: JENNY)-[HAS_FRIEND]->(Person: AMY).
    - Format these relations as: ["source_entity", "relation_name", "target_entity", "relation_description", {{"key": "val", ...}}].

    3. Return output as a flat JSON. *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*
    **You must attempt to extract as many entities and relations as you can.** It's fine to infer entity roles and connections when strongly suggested by context or scene description.
    But it's crucial that "source_entity_name" and "target_entity_name" in the identified relations, *MUST BE* one of the identified entity names.

    Domain-specific Hints:
    {hints}

    Text: {doc_context}

    Output format (flat JSON):
    {{
      "ent_i": ["type", "name", "description", "<paragraph_start>", "<paragraph_end>", {{"key": "val", ...}}],
      "rel_j": ["source_entity_name", "relation_name", "target_entity_name", "relation_description", "<paragraph_start>", "<paragraph_end>", {{"key": "val", ...}}],
      ...
    }}
    **REMINDER**: You are rewarded for high coverage and precise reasoning. Extract as much useful information as you can.
    Output:
    """
    pass


@generative
async def align_entity_with_kg(
    extracted_entity_name: str,
    extracted_entity_type: str,
    extracted_entity_desc: str,
    candidate_entities: str,
    domain: str,
    doc_text: str = ""
) -> AlignmentResult:
    """
    -Goal-
    You are given a text document, an entity candidate (with type, name, description, and potential properties) identified from the document, and a list of similar entities extracted from a knowledge graph (KG).
    The goal is to independently align each candidate entity with those KG entities. Therefore, we can leverage the candidate entity to update or create a new entity in KG.

    -Steps-
    I. Firstly, you are presented with an ID and a candidate entity in the format of "ID idx. Candidate: (<entity_type>: <entity_name>, desc: "description", props: {{key: val, ...}})".
    You will then be provided a list of existing, possible synonyms, entity types. You are also provided a set of entities from a Knowledge Graph, which also have associated entity types.
    Determine if the candidate entity type is equivalent to or a semantic subtype of any existing synonym, entity types based on semantic similarity — *we prefer using existing entity type*.
    - If yes, output the exact synonym or more general entity type (denoted as "aligned_type").
    - If no, use the original candidate entity type as is (still, denote as "aligned_type").
    #### Example ####
    ## ID 1. Candidate: (People: JOHN DOE)
    Synonym Entity Types: [Person, Employee, Actor]
    Entities:
    ent_0: (Person: JOHN DOE, props: {{gender: Male, birthday: 1994-01-17}})
    ent_1: (Person: JOHN DAN, props: {{gender: Male}})
    ent_2: (Person: JACK DOE, props: {{gender: Male}})
    Output: {{"id": 1, "aligned_type": "Person", ...}}
    Explanation: "People" can be mapped to "Person". Similarly, "Car" can be mapped to "Vehicle", and "Job" can be mapped to "Occupation".
    ####

    II. You are provided a set of entities (with type, name, description, and potential properties) from a noisy Knowledge Graph, identified to be relevant to the entity candidate, given in the format of:
    "ent_i: (<entity_type>: <entity_name>, desc: "description", props: {{key1: val, key2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})"
    where "ent_i" is the index, the percentage is a confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of varying confidence under different contexts.

    Score these KG entities that are most similar to the given candidate, particularly paying attention to entity type and name match, and provide a short reason for your choice.
    Return the matched index (ent_i) and results in a JSON of the format:
    [{{"id": 1, "aligned_type": "...", "reason": "...", "matched_entity": "ent_0"}},
     {{"id": 2, "aligned_type": "...", "reason": "...", "matched_entity": "ent_3"}}]

    Here are some tips:
    a. If you find an exact match (where both the entity type and entity name match), evaluate the "desc" and "props" information to determine if they are suitable matches.
    #### Example ####
    ## ID 1. Candidate: (Person: JOHN DOE, desc: "A normal male", props: {{gender: Male, birth_place: US, birthday: 1994-02-10}})
    Synonym Entity Types: [Person]
    Entities:
    ent_0: (Person: JOHN DOE, props: {{gender: Male, birthday: 1994-01-17}})
    ent_1: (Person: JOHN DAN, props: {{gender: Male}})
    ent_2: (Person: JACK DOE, props: {{gender: Male}})
    Output: {{"id": 1, "aligned_type": "Person", "reason": "Candidate person John Doe may not match with ent_0: person John Doe because of different birthday.", "matched_entity": ""}}
    ####

    b. If you find there is a close match (for example, different names of the same person, like "John Doe" vs. "Joe"), please also return it. It's important to maintain entity consistency in the knowledge graph.
    #### Example ####
    ## ID 2. Candidate: (Person: JENNIFER HALLEY, desc: "Actress, producer, director, and writer", props: {{birthday: 1971-01-08}})
    Synonym Entity Types: [Person]
    Entities:
    ent_0: (Person: JOHN HALLEY)
    ent_1: (Person: JEN HALLEY, desc: Actress)
    ent_2: (Person: HEATHER HALLEY)
    Output: {{"id": 2, "aligned_type": "Person", "reason": "Candidate person Jennifer Halley refers to ent_1 Person Jen Halley.", "matched_entity": "ent_1"}}
    ####

    c. If you see names that are closely matched, but they are not pointing to the same entity (for example, books with similar titles but not the same books; different types of entities with the same name), do not return any matches or suggestions. Because the candidate shouldn't update any of them.
    #### Example ####
    ## ID 3. Candidate: (Movie: KITS THESE DAYS, desc: "TV series")
    Synonym Entity Types: [Movie]
    Entities:
    ent_0: (Movie: THESE ARE THE DAYS, props: {{budget: 0, original_language: en, release_date: 1994-01-01, rating: 0.0, original_name: These Are the Days, revenue: 0}})
    ent_1: (Movie: ONE OF THESE DAYS, props: {{budget: 5217000, original_language: en, release_date: 2021-06-17, rating: None, original_name: One of These Days, revenue: 0}})
    ent_2: (Movie: BOOK OF DAYS, props: {{budget: 0, original_language: en, release_date: 2003-01-31, rating: 6.667, original_name: Book of Days, revenue: 0}})
    Output: {{"id": 3, "aligned_type": "Movie", "reason": "Candidate movie Kits These Days doesn't match any of them", "matched_entity": ""}}

    ## ID 4. Candidate: (Movie: SPRING FESTIVAL, desc: "A movie about a Chinese holiday")
    Synonym Entity Types: [Movie]
    Entities:
    ent_0: (Event: SPRING FESTIVAL, desc: "A Chinese holiday.")
    ent_1: (Movie: SPRING IS COMING, desc: "A warm movie about Spring.")
    ent_2: (Movie: FESTIVALS IN SPRING, desc: "A movie about festivals that happen in Spring.")
    Output: {{"id": 4, "aligned_type": "Movie", "reason": "Candidate movie Spring Festival doesn't match any of them. ent_0 is a type of an event, while the candidate is a movie", "matched_entity": "", "suggested_desc": "", "suggested_merge": []}}

    ## ID 5. Candidate: (Year: 1999, desc: "The year Toy Story 2 was released")
    Synonym Entity Types: [Year]
    Entities:
    ent_0: (Movie: TOY STORY 2, props: {{release_date: 1999-10-30, rating: 7.592}})
    ent_1: (Movie: TOY BOYS, props: {{release_date: 1999-03-31, rating: 0.0}})
    ent_2: (Movie: TOY STORY 4, props: {{release_date: 2019-06-19, rating: 7.505}})
    Output: {{"id": 5, "aligned_type": "Year", "reason": "Candidate year 1999 doesn't match any of them. ent_0 is a type of a movie, while the candidate represents a year", "matched_entity": ""}}
    ####

    d. Lastly, for the candidate entity that does not have enough information to make the judgment or does not have a good match, please don't return any matches (that is, "matched_entity":"").
    #### Example ####
    ## ID 6. Candidate: (Event: SPRING FESTIVAL, desc: "A Chinese holiday")
    Synonym Entity Types: [Event]
    Entities:
    ent_0: (Event: SPRING FESTIVAL, desc: "A Chinese holiday", props: {{year: 2012}})
    ent_1: (Event: SPRING FESTIVAL, desc: "A Chinese holiday", props: {{year: 2008}})
    ent_2: (Event: SPRING FESTIVAL, desc: "A Chinese holiday", props: {{year: 2004}})
    Output: {{"id": 6, "aligned_type": "Event", "reason": "Candidate event Spring Festival has multiple matches but doesn't have enough information to match exactly any of them.", "matched_entity": ""}}
    ####

    If no entities available, just simply return:
    {{"id": <id>, "aligned_type": "", "reason": "No entities to match with.", "matched_entity": ""}}

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*

    Text: {doc_text}
    Extracted Entity:
    - Name: {extracted_entity_name}
    - Type: {extracted_entity_type}
    - Description: {extracted_entity_desc}

    Candidate Entities from KG:
    {candidate_entities}
    """
    pass


@generative
async def decide_entity_merge(
    entity_pair: str,
    doc_text: str,
    domain: str
) -> MergeDecision:
    """
    -Goal-
    You are given a text document and a list of entity pairs. In each pair, the first entity is tentatively identified from the text document, while the second entity is from a knowledge graph (KG).
    The goal is to combine information from both of them and write the merged entity back to the KG, and therefore keeping the KG with accurate up-to-date information.

    -Steps-
    1. You are provided a list of entity pairs (with type, name, description, and potential properties), given in the format of
    "idx: [(<entity_type>: <entity_name>, desc: "description", props: {{key: val, ...}}), (<entity_type>: <entity_name>, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})]"
    where idx is the index, the percentage is confidence score, ctx is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of varying confidence under different contexts.
    - If there are no properties available, the entire "props" field will be skipped.
    - Each property may have multiple correct values depending on its given context. For example, a movie may have several release dates depending on the region. These values are sorted by their confidence scores ("conf").
    - You need to decide independently for each property, given the context in the text document, if its value from the first entity can be merged with a value in the second entity or if you need to create a new value with the new context.

    2. Please merge information from both of them: phrase the entity description in a better, general way, and only retain the **single**, most accurate value for each entity property.
    If the property values from both sides are essentially the same, the merged property value always adheres to the format of the second entity.
    #### Example ####
    1: [(Nation: United States, desc: "A country", props: {{population: 340.1 million}}), (Nation: United States, desc: "Country in North America", props: {{population: 340,000,000}})]
    Output: [{{"id": 1, "desc": "A country in North America", "props": {{"population": ["340,100,000", ""]}}}}]
    Explanation: The population on both sides roughly matches, so we retain the most accurate value and adhere to the numeric format of the second entity.
    ####

    3. Return the index, merged entity description, and entity properties (key, value, and an optional context, which can be an empty string, under which this value is valid) into a FLAT JSON of the format:
    [{{"id": 1, "desc": "entity_description", "props": {{"key": ["val", "context"], ...}}}},
     {{"id": 2, "desc": "entity_description", "props": {{}}}}, ...]
     where the "props" field is an optional key-value pair that can be empty, {{}}, when no property is available.

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*

    Text: {doc_text}
    Entity Pairs to Merge:
    {entity_pair}

    Output format (a flat JSON):
    [{{"id": 1, "desc": "entity_description", "props": {{"key": ["val", "context"], ...}}}},
    {{"id": 2, "desc": "entity_description", "props": {{}}}}, ...]
    Output:
    """
    pass


@generative
async def align_relation_with_kg(
    extracted_relation: str,
    candidate_relations: str,
    synonym_relations: str,
    domain: str,
    doc_text: str = ""
) -> AlignmentResult:
    """
    -Goal-
    You are given a text document, a relation candidate (type, name, description, and potential properties) identified from the document, and a list of similar relations extracted from a knowledge graph (KG).
    The goal is to independently align each candidate relation with those KG relations. Therefore, we can leverage the candidate relation to update or create a new relation in KG.

    -Steps-
    I. Firstly, you are presented with an ID and a candidate relation in the format of "ID idx. Candidate: (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {{key: val, ...}}]->(<target_entity_type>: <target_entity_name>)".
    You will then be provided a list of existing, possible synonym, directed relation names to the candidate relation in the format of "(<source_entity_type>)-[<relation_name>]->(<target_entity_type>)".
    Determine if the candidate relation name is equivalent to or a semantic subtype of any existing synonym, directed relation names based on semantic similarity — *we prefer using existing relation name*.
    If yes, output the exact synonym or more general relation name that matches the direction (denoted as "aligned_name").
    If no, just use the original candidate relation name as is (still, denote as "aligned_name").
    #### Example ####
    ## ID 1. Candidate: (Person: JOHN DOE)-[JOIN_PARTY, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    Synonym Relations:(Person)-[JOIN]->(Event)
    (Person)-[HOST]->(Event)
    (Event)-[PLANNED_BY]->(Person)
    Output: {{"id": 1, "aligned_name": "JOIN", ...}}
    Explanation: "JOIN_PARTY" can be mapped to "JOIN". Similarly, "TAUGHT_COURSE" can be mapped to "TEACH", "COLLABORATED_WITH_IN_YEAR" can be mapped to "COLLABORATED_WITH".
    ####

    II. You are then provided a set of existing relations identified from a knowledge graph that may be relevant to the relation candidate, given in the format of
    "rel_i: (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {{key1: val, key2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(<target_entity_type>: <target_entity_name>)".
    where "rel_i" is the index, the percentage is a confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of varying confidence under different contexts.

    Score the relations that are most similar to the given candidate and provide a short reason for your scoring.
    Return the candidate ID, aligned name, and its matched relation into a flat JSON of the format:
    [{{"id": 1, "aligned_name": "...", "reason": "...", "matched_relation": "rel_0"}},
     {{"id": 2, "aligned_name": "...", reason": "...", "matched_relation": "rel_3"}}]
    Here are some tips:
    a. If you find an exact match (relation type and entity name both match), please don't hesitate to just return it. For example, "matched_relation": "rel_0".
    #### Example ####
    ## ID 2. Candidate: (Person: JOHN DOE)-[JOIN_PARTY, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    Relations:
    rel_0: (Person: JOHN DOE)-[JOIN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    rel_1: (Person: JOHN DOE)-[HOST, properties: <date: 06-20-2005, place: "stadium">]->(Event: MUSIC PARTY)
    rel_2: (Person: JOHN DOE)-[PLAN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    Output: {{"id": 2, "aligned_name": "JOIN", "reason": "'John Doe join Music Party' exact match with rel_0: 'John Doe joined Music Party on 06-20-2005'", "matched_relation": "rel_0"}}
    ####

    b. If you find there is a close match (for example, different names of the same relations, like "COLLABORATED_WITH" vs. "COLLABORATED_WITH_IN_YEAR"), please also return it. It's important to maintain entity consistency in the knowledge graph.
    #### Example ####
    ## ID 3. Candidate: (Person: JOHN DOE)-[COLLABORATED_WITH_IN_YEAR]->(Person: RICHARD)
    Relations:
    rel_0: (Person: JOHN DOE)-[IS_FRIEND_WITH]->(Person: RICHARD)
    rel_1: (Person: JOHN DOE)-[COLLABORATED_WITH, properties: <year: 2015>]->(Person: RICHARD)
    rel_2: (Person: JOHN DOE)-[HAS_KNOWN, properties: <year: 2015>]->(Person: RICHARD)
    Output: {{"id": 3, "aligned_name": "COLLABORATED_WITH", "reason": "'John Doe collaborated with Richard in year' exact match with rel_1: 'John Doe collaborated with Richard in 2015'", "matched_relation": "rel_1"}}
    ####

    c. If you see names that are closely matched, but they are not pointing to the same relations (having different properties, etc.), do not return any matches. The candidate shouldn't be merged with them. But you still need to return its aligned name:
    #### Example ####
    ## ID 4. Candidate: (Person: JOHN DOE)-[JOIN_PARTY, properties: <date: 06-20-2006>]->(Event: MUSIC PARTY)
    Relations:
    rel_0: (Person: JOHN DOE)-[JOIN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    rel_1: (Person: JOHN DOE)-[HOST, properties: <date: 06-20-2005, place: "stadium">]->(Event: MUSIC PARTY)
    rel_2: (Person: JOHN DOE)-[PLAN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    Output: {{"id": 4, "aligned_name": "JOIN", "reason": "'John Doe join Music Party on 06-20-2006' doesn't match (different year) with rel_0: 'John Doe joined Music Party on 06-20-2005'", "matched_relation": ""}}
    ####

    d. Lastly, for the candidate relations that do not have a good match, please don't return any scores (that is, "matched_relation":"").

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*

    Text: {doc_text}
    Extracted Relation:
    {extracted_relation}

    Synonym Relations:
    {synonym_relations}

    Candidate Relations from KG:
    {candidate_relations}
    """
    pass


@generative
async def decide_relation_merge(
    relation_pair: str,
    doc_text: str,
    domain: str
) -> MergeDecision:
    """
    -Goal-
    You are given a text document and a list of relationship pairs. Each relationship contains a source entity, a target entity, and a relation between them (consists of type, description, and potential properties). The properties associated with each relation depend on their relation type, but some may be missing.
    In each pair, the first relationship is tentatively identified from the text document, while the second relationship is from a knowledge graph (KG).
    The goal is to combine information from both of them and write the merged relationship back to the KG, and therefore keeping the KG with accurate up-to-date information.

    -Steps-
    1. You are provided a list of relation pairs, given in the format of
    "idx: [(<source_entity_type>: <source_entity_name>)-[<relation_type>, desc: "description", props: {{key: val, ...}}]->(<target_entity_type>: <target_entity_name>),
     (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(<target_entity_type>: <target_entity_name>)]"
    where idx is the index, the percentage is confidence score, ctx is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.
    - If there are no properties available, the entire "props" field will be skipped.
    - Each property may have multiple correct values depending on its given context. For example, a movie may have several release dates depending on the region. These values are sorted by their confidence scores (the percentage).
    - You need to decide independently for each property, given the context in the text document, if its value from the first entity can be merged with a value in the second entity or if you need to create a new value with the new context.

    2. Please merge information independently from relationships in each pair: phrase the relation description in a better, general way, and only retain the **single**, most accurate value for each relation property.
    If the property values from both sides are essentially the same, the merged property value always adheres to the format of the second relationship.
    #### Example ####
    1: [(Nation: United States)-[<HAS_POPULATION>, desc: "US has 340.1 million population", props: {{population: 340.1 million}}]->(Nation: United States), (Nation: United States)-[<HAS_POPULATION>, desc: "US has population", props: {{population: 340,000,000}}]->(Nation: United States)]
    Output: [{{"id": 1, "desc": "US has 340.1 million population", "props": {{"population": ["340,100,000", ""]}}}}]
    Explanation: The population on both sides roughly matches, so we retain the most accurate value and adhere to the numeric format of the second relationship.
    ####

    3. Return the index and merged description and relation properties (key, value, and an optional context, which can be an empty string, under which this value is valid) into a FLAT JSON of the format:
    [{{"id": 1, "desc": "relation_description", "props": {{"key": ["val", "context"], ...}}}},
     {{"id": 2, "desc": "relation_description", "props": {{}}}}, ...]
     where the "props" field is an optional key-value pair that can be empty, {{}}, when no relation property is available.

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*

    Text: {doc_text}
    Relation Pairs to Merge:
    {relation_pair}

    Output format (a flat JSON):
    [{{"id": 1, "desc": "relation_description", "props": {{"key": ["val", "context"], ...}}}},
    {{"id": 2, "desc": "relation_description", "props": {{}}}}, ...]
    Output:
    """
    pass
