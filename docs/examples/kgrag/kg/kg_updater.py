import argparse
import asyncio
from collections import OrderedDict
from copy import deepcopy
import functools
import json
import math
import textwrap
import time
from typing import Any, Dict, List, Tuple

from datetime import datetime, timezone
from kg.kg_driver import kg_driver
from kg.kg_rep import (
    RESERVED_KEYS,
    KGEntity,
    KGRelation,
    CandidateEntity,
    CandidateRelation,
    normalize_entity,
    normalize_entity_type,
    normalize_relation,
    normalize_value,
    entity_to_text,
    relation_to_text,
    entity_schema_to_text,
    relation_schema_to_text,
    update_ref,
    RelevantEntity
)
from utils.prompt_list import PROMPTS
from utils.utils import (
    generate_embedding,
    generate_response,
    llm_retry,
    maybe_load_json,
    parse_timestamp,
    token_counter
)
from utils.logger import BaseProgressLogger, DefaultProgressLogger

# Configuration constants (define defaults if not present)
ALIGN_ENTITY = True
ALIGN_RELATION = True
MERGE_ENTITY = True
MERGE_RELATION = True
SELF_REFLECTION = False
ALIGN_TOPK = 10
ALIGN_ENTITY_BATCH_SIZE = 10
MERGE_ENTITY_BATCH_SIZE = 10
ALIGN_RELATION_BATCH_SIZE = 10
MERGE_RELATION_BATCH_SIZE = 10
MAX_RETRIES = 3
MAX_GENERATION_TOKENS = 20000
STAGES = 2
EXPECTED_TIME = 600  # 10 minutes in seconds
MAX_CHUNK = 60000
MIN_CHUNK = 10000

PROMPTS["extraction"] = {
    # Formatted string
    "system": textwrap.dedent("""\
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
	• Choose the most informative paragraph that (a) contains the entity’s full name and (b) contributes evidence for its description or properties.
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
	• Choose the most informative paragraph that (a) contains the entity’s full name and (b) contributes evidence for its description or properties.
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
    **You must attempt to extract as many entities and relations as you can.** It’s fine to infer entity roles and connections when strongly suggested by context or scene description.
    But it's crucial that "source_entity_name" and "target_entity_name" in the identified relations, *MUST BE* one of the identified entity names. 

    Domain-specific Hints:
    {hints}
    
    Here are some examples:
    """),

    "examples": [
        {
            "text": "Marie Curie, a Polish-French physicist, was born in Warsaw and later became a professor at the University of Paris. She was awarded two Nobel Prizes: one in Physics (1903) and one in Chemistry (1911).",
            "output": {
                "ent_0": ["Person", "Marie Curie", "Polish-French physicist", "Marie Curie, a Poli", "and one in Chemistry (1911).", {"nationality": "Polish-French", "birth_place": "Warsaw"}],
                "ent_1": ["Geo", "Warsaw", "Marie Curie's birthplace", "Marie Curie, a Poli", "and one in Chemistry (1911).", {}],
                "ent_2": ["Organization", "University of Paris", "Where Marie Curie was a professor", "Marie Curie, a Poli", "and one in Chemistry (1911).", {}],
                "ent_3": ["Award", "Nobel Prize in Physics", "Award won by Marie Curie in 1903", "She was awarded two", "and one in Chemistry (1911).", {"year": "1903", "type": "Nobel Prize"}],
                "ent_4": ["Award", "Nobel Prize in Chemistry", "Award won by Marie Curie in 1911", "She was awarded two", "and one in Chemistry (1911).", {"year": "1911", "type": "Nobel Prize"}],
                "rel_0": ["Marie Curie", "WORKS_AT", "University of Paris", "Marie Curie worked at the University of Paris", "Marie Curie, a Poli", "and one in Chemistry (1911).", {"valid_from": "None", "valid_until": "None", "occupation": "professor"}],
                "rel_1": ["Marie Curie", "WON", "Nobel Prize in Physics", "Marie Curie won the Nobel Prize in Physics in 1903", "She was awarded two", "and one in Chemistry (1911).", {"year": "1903"}],
                "rel_2": ["Marie Curie", "WON", "Nobel Prize in Chemistry", "Marie Curie won the Nobel Prize in Chemistry in 1911", "She was awarded two", "and one in Chemistry (1911).", {"year": "1911"}]
            },
            "explanation": "The nationality and birth place is life-time exclusive, so they are Marie's properties. WORKS_AT is modeled as exclusive relations, given the valid period. WON award is an accumulative relationship, so no valid period is given."
        },
        {
            "text": "Inception, a science fiction film released in 2010 and starring Leonardo DiCaprio, was lauded for its groundbreaking visual effects. At the 83rd Academy Awards (2010), Inception received the Oscar for Best Visual Effects. Leonardo DiCaprio, born on November 11, 1974, portrayed the lead character in this visually stunning film with a production budget of $160 million.",
            "output": {
                "ent_0": ["Movie", "Inception", "A science fiction film released in 2010", "Inception, a scienc", "budget of $160 million.", {"release_year": 2010, "budget": 160000000}],
                "ent_1": ["Person", "Leonardo DiCaprio", "American actor and Hollywood A-lister", "Inception, a scienc", "budget of $160 million.", {"birthday": "November 11, 1974"}],
                "ent_2": ["Award", "Visual Effects", "An award for the best visual effects", "At the 83rd Academy", "budget of $160 million.",  {"year": 2011, "ceremony_number": 83, "type": "OSCAR"}],
                "rel_0": ["Leonardo DiCaprio", "ACTED_IN", "Inception", "Leonardo DiCaprio acted in the film Inception", "Inception, a scienc", "budget of $160 million.", {}],
                "rel_1": ["Inception", "WON", "Visual Effects", "Inception won the Best Visual Effects Oscar at the 83rd Academy Awards", "At the 83rd Academy", "budget of $160 million.", {"winner": "true", "movie": "Inception"}],
            },
            "explanation": "Although the text refers to the “83rd Academy Awards (2010)”, the extraction uses 2011 as the year property of the award entity, since the actual 83rd award event took place in 2011, consistent with the domain-specific hint that the Oscars are typically held one year after the movie’s release."
        }
    ],

    "user": textwrap.dedent("""\
    Text: {input_text}
    Domain-specific Hints:
    {hints}
    
    Output format (flat JSON):
    {{
      "ent_i": ["type", "name", "description", "<paragraph_start>", "<paragraph_end>", {{"key": "val", ...}}],
      "rel_j": ["source_entity_name", "relation_name", "target_entity_name", "relation_description", "<paragraph_start>", "<paragraph_end>", {{"key": "val", ...}}],
      ...
    }}
    **REMINDER**: You are rewarded for high coverage and precise reasoning. Extract as much useful information as you can.
    Output:"""),

    "missing_entities": textwrap.dedent("""\
    Wait, now it's a great opportunity to go over all the entities mentioned in the extracted relations, and identify those that were missing extraction.
    Currently, I found these entities are mentioned in the relations but never extracted: {missing_entities}
    
    You may either (1) add those missing entities to the output JSON before completing the task. (2) overwrite an inappropriate entity or relation by adding a new entity or relation using the same index ("ent_i" or "rel_i").
    It's totally fine that the identified entity "ent_i" and identified relation "rel_i" interleave with each other in the output JSON.
    But it's crucial that "source_entity_name" and "target_entity_name" in the identified relations, *MUST BE* one of the extracted entity names. 
    """),

    "continue_extraction": textwrap.dedent("""\
    Are these extracted entities and relations fully reflecting all the essential relationships contained in the text?
    If yes, respond with an empty JSON: {}. 
    If no, continue extracting any missing relationships from the provided text. You only need to output the new extracted entities/relations.
    
    ######################
    Output:"""),

    "self-reflection": textwrap.dedent("""\
    Reflecting again on the extracted entity and relations, do they strictly adhere to the above rules? 
    Especially pay attention to: 
    1. the lifetime exclusive rule for entity properties;
    2. exclusive relationships that evolve over time;
    3. accumulative relations (non-exclusive) relationships;
    4. Do NOT use vague pronouns like "this", "that", or "it" to refer to prior context in the text. Always use full, explicit names or phrases that can stand alone.
    
    Domain-specific Hints:
    {hints}
    
    If necessary, you can *update* any inappropriate entity or relation by outputting a new entity or relation using the exact same index (the "ent_i" or "rel_i"), in the format of:
    {{
      "ent_i": ["type", "name", "description", {{"key": "val", ...}}],
      "rel_j": ["source_entity_name", "relation_name", "target_entity_name", "relation_description", {{"key": "val", ...}}],
      ...
    }}
    
    You can also *delete* any inappropriate entity or relation by outputting an empty list using the exact same index in the format of:
    {{
      "ent_i": [],
      "rel_j": [],
      ...
    }}
    **REMINDER**: ONLY output entity or relations that require an update!"""),
}

PROMPTS["align_entity"] = {
    # Original string
    "system": textwrap.dedent("""\
    -Goal-
    You are given a text document, an entity candidate (with type, name, description, and potential properties) identified from the document, and a list of similar entities extracted from a knowledge graph (KG).
    The goal is to independently align each candidate entity with those KG entities. Therefore, we can leverage the candidate entity to update or create a new entity in KG. 
    
    -Steps-
    I. Firstly, you are presented with an ID and a candidate entity in the format of "ID idx. Candidate: (<entity_type>: <entity_name>, desc: "description", props: {key: val, ...})".
    You will then be provided a list of existing, possible synonyms, entity types. You are also provided a set of entities from a Knowledge Graph, which also have associated entity types.
    Determine if the candidate entity type is equivalent to or a semantic subtype of any existing synonym, entity types based on semantic similarity — *we prefer using existing entity type*.
    - If yes, output the exact synonym or more general entity type (denoted as "aligned_type").
    - If no, use the original candidate entity type as is (still, denote as "aligned_type").
    #### Example ####
    ## ID 1. Candidate: (People: JOHN DOE) 
    Synonym Entity Types: [Person, Employee, Actor]
    Entities:
    ent_0: (Person: JOHN DOE, props: {gender: Male, birthday: 1994-01-17})
    ent_1: (Person: JOHN DAN, props: {gender: Male})
    ent_2: (Person: JACK DOE, props: {gender: Male})
    Output: {"id": 1, "aligned_type": "Person", ...} 
    Explanation: "People" can be mapped to "Person". Similarly, "Car" can be mapped to "Vehicle", and "Job" can be mapped to "Occupation".
    ####
    
    II. You are provided a set of entities (with type, name, description, and potential properties) from a noisy Knowledge Graph, identified to be relevant to the entity candidate, given in the format of:
    "ent_i: (<entity_type>: <entity_name>, desc: "description", props: {key1: val, key2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...})"
    where "ent_i" is the index, the percentage is a confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of varying confidence under different contexts.
    
    Score these KG entities that are most similar to the given candidate, particularly paying attention to entity type and name match, and provide a short reason for your choice.
    Return the matched index (ent_i) and results in a JSON of the format:
    [{"id": 1, "aligned_type": "...", "reason": "...", "matched_entity": "ent_0"},
     {"id": 2, "aligned_type": "...", "reason": "...", "matched_entity": "ent_3"}]
     
    Here are some tips:
    a. If you find an exact match (where both the entity type and entity name match), evaluate the "desc" and "props" information to determine if they are suitable matches.
    #### Example ####
    ## ID 1. Candidate: (Person: JOHN DOE, desc: "A normal male", props: {gender: Male, birth_place: US, birthday: 1994-02-10})
    Synonym Entity Types: [Person]
    Entities:
    ent_0: (Person: JOHN DOE, props: {gender: Male, birthday: 1994-01-17})
    ent_1: (Person: JOHN DAN, props: {gender: Male})
    ent_2: (Person: JACK DOE, props: {gender: Male})
    Output: {"id": 1, "aligned_type": "Person", "reason": "Candidate person John Doe may not match with ent_0: person John Doe because of different birthday.", "matched_entity": ""}
    ####
    
    b. If you find there is a close match (for example, different names of the same person, like "John Doe" vs. "Joe"), please also return it. It's important to maintain entity consistency in the knowledge graph.
    #### Example ####
    ## ID 2. Candidate: (Person: JENNIFER HALLEY, desc: "Actress, producer, director, and writer", props: {birthday: 1971-01-08})
    Synonym Entity Types: [Person]
    Entities:
    ent_0: (Person: JOHN HALLEY)
    ent_1: (Person: JEN HALLEY, desc: Actress)
    ent_2: (Person: HEATHER HALLEY)
    Output: {"id": 2, "aligned_type": "Person", "reason": "Candidate person Jennifer Halley refers to ent_1 Person Jen Halley.", "matched_entity": "ent_1"}
    ####
    
    c. If you see names that are closely matched, but they are not pointing to the same entity (for example, books with similar titles but not the same books; different types of entities with the same name), do not return any matches or suggestions. Because the candidate shouldn't update any of them.
    #### Example ####
    ## ID 3. Candidate: (Movie: KITS THESE DAYS, desc: "TV series")
    Synonym Entity Types: [Movie]
    Entities:
    ent_0: (Movie: THESE ARE THE DAYS, props: {budget: 0, original_language: en, release_date: 1994-01-01, rating: 0.0, original_name: These Are the Days, revenue: 0})
    ent_1: (Movie: ONE OF THESE DAYS, props: {budget: 5217000, original_language: en, release_date: 2021-06-17, rating: None, original_name: One of These Days, revenue: 0})
    ent_2: (Movie: BOOK OF DAYS, props: {budget: 0, original_language: en, release_date: 2003-01-31, rating: 6.667, original_name: Book of Days, revenue: 0})
    Output: {"id": 3, "aligned_type": "Movie", "reason": "Candidate movie Kits These Days doesn't match any of them", "matched_entity": ""}

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
    ent_0: (Movie: TOY STORY 2, props: {release_date: 1999-10-30, rating: 7.592})
    ent_1: (Movie: TOY BOYS, props: {release_date: 1999-03-31, rating: 0.0})
    ent_2: (Movie: TOY STORY 4, props: {release_date: 2019-06-19, rating: 7.505})
    Output: {"id": 5, "aligned_type": "Year", "reason": "Candidate year 1999 doesn't match any of them. ent_0 is a type of a movie, while the candidate represents a year", "matched_entity": ""}
    ####
    
    d. Lastly, for the candidate entity that does not have enough information to make the judgment or does not have a good match, please don't return any matches (that is, "matched_entity":"").
    #### Example ####
    ## ID 6. Candidate: (Event: SPRING FESTIVAL, desc: "A Chinese holiday")
    Synonym Entity Types: [Event]
    Entities:
    ent_0: (Event: SPRING FESTIVAL, desc: "A Chinese holiday", props: {year: 2012})
    ent_1: (Event: SPRING FESTIVAL, desc: "A Chinese holiday", props: {year: 2008})
    ent_2: (Event: SPRING FESTIVAL, desc: "A Chinese holiday", props: {year: 2004})
    Output: {{"id": 6, "aligned_type": "Event", "reason": "Candidate event Spring Festival has multiple matches but doesn't have enough information to match exactly any of them.", "matched_entity": ""}}
    ####
                              
    If no entities available, just simply return:
    {{"id": <id>, "aligned_type": "", "reason": "No entities to match with.", "matched_entity": ""}}
    
    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*
    """), # {{"reason": "reason", "similar_entites": {{"ent_i": score_i, "ent_j": score_j}}}}.

    "user": textwrap.dedent("""\
    Text: {input_text}
    Entities to Align:
    """),

    "missing": textwrap.dedent("""\
    You may have forgotten a few entity candidates with ID: {missing_entities}, or there could be a parsing error. Please return the additional results for those missing candidates. 
    """)
}

PROMPTS["merge_entity"] = {
    # Original string
    "system": textwrap.dedent("""\
    -Goal-
    You are given a text document and a list of entity pairs. In each pair, the first entity is tentatively identified from the text document, while the second entity is from a knowledge graph (KG).
    The goal is to combine information from both of them and write the merged entity back to the KG, and therefore keeping the KG with accurate up-to-date information.
    
    -Steps-
    1. You are provided a list of entity pairs (with type, name, description, and potential properties), given in the format of
    "idx: [(<entity_type>: <entity_name>, desc: "description", props: {key: val, ...}), (<entity_type>: <entity_name>, desc: "description", props: {key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...})]"
    where idx is the index, the percentage is confidence score, ctx is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of varying confidence under different contexts.
    - If there are no properties available, the entire "props" field will be skipped.
    - Each property may have multiple correct values depending on its given context. For example, a movie may have several release dates depending on the region. These values are sorted by their confidence scores ("conf").
    - You need to decide independently for each property, given the context in the text document, if its value from the first entity can be merged with a value in the second entity or if you need to create a new value with the new context.
    
    2. Please merge information from both of them: phrase the entity description in a better, general way, and only retain the **single**, most accurate value for each entity property.
    If the property values from both sides are essentially the same, the merged property value always adheres to the format of the second entity.
    #### Example ####
    1: [(Nation: United States, desc: "A country", props: {population: 340.1 million}), (Nation: United States, desc: "Country in North America", props: {population: 340,000,000})]
    Output: [{"id": 1, "desc": "A country in North America", "props": {"population": ["340,100,000", ""]}}]
    Explanation: The population on both sides roughly matches, so we retain the most accurate value and adhere to the numeric format of the second entity.
    ####
    
    3. Return the index, merged entity description, and entity properties (key, value, and an optional context, which can be an empty string, under which this value is valid) into a FLAT JSON of the format:
    [{"id": 1, "desc": "entity_description", "props": {"key": ["val", "context"], ...}},
     {"id": 2, "desc": "entity_description", "props": {}}, ...]
     where the "props" field is an optional key-value pair that can be empty, {}, when no property is available.

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*  

    """), # {{"reason": "reason", "similar_entites": {{"ent_i": score_i, "ent_j": score_j}}}}.

    "user": textwrap.dedent("""\
    Text: {input_text}
    Entity Pairs to Merge:
    """),

    "missing": textwrap.dedent("""\
    You may have forgotten a few entity pairs with indices: {missing_entities}, or there could be a parsing error. Please return the additional results for those missing pairs. 
    """)
}

PROMPTS["align_relation"] = {
    # Original string
    "system": textwrap.dedent("""\
    -Goal-
    You are given a text document, a relation candidate (type, name, description, and potential properties) identified from the document, and a list of similar relations extracted from a knowledge graph (KG).
    The goal is to independently align each candidate relation with those KG relations. Therefore, we can leverage the candidate relation to update or create a new relation in KG. 
    
    -Steps-
    I. Firstly, you are presented with an ID and a candidate relation in the format of "ID idx. Candidate: (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {key: val, ...}]->(<target_entity_type>: <target_entity_name>)".
    You will then be provided a list of existing, possible synonym, directed relation names to the candidate relation in the format of "(<source_entity_type>)-[<relation_name>]->(<target_entity_type>)".
    Determine if the candidate relation name is equivalent to or a semantic subtype of any existing synonym, directed relation names based on semantic similarity — *we prefer using existing relation name*.
    If yes, output the exact synonym or more general relation name that matches the direction (denoted as "aligned_name").
    If no, just use the original candidate relation name as is (still, denote as "aligned_name").
    #### Example ####
    ## ID 1. Candidate: (Person: JOHN DOE)-[JOIN_PARTY, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY) 
    Synonym Relations:(Person)-[JOIN]->(Event)
    (Person)-[HOST]->(Event)
    (Event)-[PLANNED_BY]->(Person)
    Output: {"id": 1, "aligned_name": "JOIN", ...} 
    Explanation: "JOIN_PARTY" can be mapped to "JOIN". Similarly, "TAUGHT_COURSE" can be mapped to "TEACH", "COLLABORATED_WITH_IN_YEAR" can be mapped to "COLLABORATED_WITH".
    ####

    II. You are then provided a set of existing relations identified from a knowledge graph that may be relevant to the relation candidate, given in the format of
    "rel_i: (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {key1: val, key2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}]->(<target_entity_type>: <target_entity_name>)".
    where "rel_i" is the index, the percentage is a confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of varying confidence under different contexts.
    
    Score the relations that are most similar to the given candidate and provide a short reason for your scoring.
    Return the candidate ID, aligned name, and its matched relation into a flat JSON of the format:
    [{"id": 1, "aligned_name": "...", "reason": "...", "matched_relation": "rel_0"},
     {"id": 2, "aligned_name": "...", reason": "...", "matched_relation": "rel_3"}]
    Here are some tips:
    a. If you find an exact match (relation type and entity name both match), please don't hesitate to just return it. For example, "matched_relation": "rel_0".
    #### Example ####
    ## ID 2. Candidate: (Person: JOHN DOE)-[JOIN_PARTY, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    Relations:
    rel_0: (Person: JOHN DOE)-[JOIN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    rel_1: (Person: JOHN DOE)-[HOST, properties: <date: 06-20-2005, place: "stadium">]->(Event: MUSIC PARTY)
    rel_2: (Person: JOHN DOE)-[PLAN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    Output: {"id": 2, "aligned_name": "JOIN", "reason": "'John Doe join Music Party' exact match with rel_0: 'John Doe joined Music Party on 06-20-2005'", "matched_relation": "rel_0"}
    ####
    
    b. If you find there is a close match (for example, different names of the same relations, like "COLLABORATED_WITH" vs. "COLLABORATED_WITH_IN_YEAR"), please also return it. It's important to maintain entity consistency in the knowledge graph.
    #### Example ####
    ## ID 3. Candidate: (Person: JOHN DOE)-[COLLABORATED_WITH_IN_YEAR]->(Person: RICHARD)
    Relations:
    rel_0: (Person: JOHN DOE)-[IS_FRIEND_WITH]->(Person: RICHARD)
    rel_1: (Person: JOHN DOE)-[COLLABORATED_WITH, properties: <year: 2015>]->(Person: RICHARD)
    rel_2: (Person: JOHN DOE)-[HAS_KNOWN, properties: <year: 2015>]->(Person: RICHARD)
    Output: {"id": 3, "aligned_name": "COLLABORATED_WITH", "reason": "'John Doe collaborated with Richard in year' exact match with rel_1: 'John Doe collaborated with Richard in 2015'", "matched_relation": "rel_1"}
    ####
    
    c. If you see names that are closely matched, but they are not pointing to the same relations (having different properties, etc.), do not return any matches. The candidate shouldn't be merged with them. But you still need to return its aligned name:
    #### Example ####
    ## ID 4. Candidate: (Person: JOHN DOE)-[JOIN_PARTY, properties: <date: 06-20-2006>]->(Event: MUSIC PARTY)
    Relations:
    rel_0: (Person: JOHN DOE)-[JOIN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    rel_1: (Person: JOHN DOE)-[HOST, properties: <date: 06-20-2005, place: "stadium">]->(Event: MUSIC PARTY)
    rel_2: (Person: JOHN DOE)-[PLAN, properties: <date: 06-20-2005>]->(Event: MUSIC PARTY)
    Output: {"id": 4, "aligned_name": "JOIN", "reason": "'John Doe join Music Party on 06-20-2006' doesn't match (different year) with rel_0: 'John Doe joined Music Party on 06-20-2005'", "matched_relation": ""}
    ####
    
    d. Lastly, for the candidate relations that do not have a good match, please don't return any scores (that is, "matched_relation":"").
    
    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*  
    """),

    "user": textwrap.dedent("""\
    Text: {input_text}
    Relations to Align:
    """),

    "missing": textwrap.dedent("""\
    You may have forgotten a few relation candidates with ID: {missing_relations}, or there could be a parsing error. Please return the additional results for those missing candidates. 
    """)
}

PROMPTS["merge_relation"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are given a text document and a list of relationship pairs. Each relationship contains a source entity, a target entity, and a relation between them (consists of type, description, and potential properties). The properties associated with each relation depend on their relation type, but some may be missing.
    In each pair, the first relationship is tentatively identified from the text document, while the second relationship is from a knowledge graph (KG).
    The goal is to combine information from both of them and write the merged relationship back to the KG, and therefore keeping the KG with accurate up-to-date information.
    
    -Steps-
    1. You are provided a list of relation pairs, given in the format of
    "idx: [(<source_entity_type>: <source_entity_name>)-[<relation_type>, desc: "description", props: {key: val, ...}]->(<target_entity_type>: <target_entity_name>), 
     (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}]->(<target_entity_type>: <target_entity_name>)]"
    where idx is the index, the percentage is confidence score, ctx is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.
    - If there are no properties available, the entire "props" field will be skipped.
    - Each property may have multiple correct values depending on its given context. For example, a movie may have several release dates depending on the region. These values are sorted by their confidence scores (the percentage).
    - You need to decide independently for each property, given the context in the text document, if its value from the first entity can be merged with a value in the second entity or if you need to create a new value with the new context.
    
    2. Please merge information independently from relationships in each pair: phrase the relation description in a better, general way, and only retain the **single**, most accurate value for each relation property.
    If the property values from both sides are essentially the same, the merged property value always adheres to the format of the second relationship.
    #### Example ####
    1: [(Nation: United States)-[<HAS_POPULATION>, desc: "US has 340.1 million population", props: {population: 340.1 million}]->(Nation: United States), (Nation: United States)-[<HAS_POPULATION>, desc: "US has population", props: {population: 340,000,000}]->(Nation: United States)]
    Output: [{"id": 1, "desc": "US has 340.1 million population", "props": {"population": ["340,100,000", ""]}}]
    Explanation: The population on both sides roughly matches, so we retain the most accurate value and adhere to the numeric format of the second relationship.
    ####
    
    3. Return the index and merged description and relation properties (key, value, and an optional context, which can be an empty string, under which this value is valid) into a FLAT JSON of the format:
    [{"id": 1, "desc": "relation_description", "props": {"key": ["val", "context"], ...}},
     {"id": 2, "desc": "relation_description", "props": {}}, ...] 
     where the "props" field is an optional key-value pair that can be empty, {}, when no relation property is available.

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT FLAT JSON*  
    """),

    "user": textwrap.dedent("""\
    Text: {input_text}
    Relation Pairs to Merge:
    """),

    "missing": textwrap.dedent("""\
    You may have forgotten a few relation pairs with indices: {missing_relations}, or there could be a parsing error. Please return the additional results for those missing pairs. 
    """)
}

class KG_Updater:
    def __init__(
        self,
        session = None,
        emb_session = None,
        config: Dict = None,
        logger: BaseProgressLogger = DefaultProgressLogger()
    ):
        self.session = session
        self.emb_session = emb_session
        self.config = {
            "align_entity": ALIGN_ENTITY,
            "merge_entity": MERGE_ENTITY,
            "align_relation": ALIGN_RELATION,
            "merge_relation": MERGE_RELATION,
            "self_reflection": SELF_REFLECTION,
        }
        if config:
            self.config.update(config)

        self.logger = logger

        # Set embedding session on the global kg_driver singleton
        if emb_session is not None:
            kg_driver.set_emb_session(emb_session)

    @llm_retry(max_retries=MAX_RETRIES, default_output={})
    async def align_entity(
        self,
        entities: OrderedDict[CandidateEntity],
        context: str, 
        top_k: int = ALIGN_TOPK, 
        batch_size: int = ALIGN_ENTITY_BATCH_SIZE,
        max_realign: int = 5
    ) -> OrderedDict[CandidateEntity]:
        """
        Perform MIPS (Maximum Inner Product Search) in KG to align a list of extracted entities to KG entities.

        Args:
            entities (OrderedDict[CandidateEntity]): A dictionary of candidate entities.
            top_k (int): Specify the top-k entities assessed in KG.
            batch_size (int): Number of entities to process per LLM call.

        Returns:
            OrderedDict[CandidateEntity]: A list of aligned KG entities.
        """
        
        entity_names = list(entities)
        schema_description = [entity_schema_to_text(entity.extracted.type) for entity in entities.values()]
        entities_description = [entity_to_text(entity.extracted) for entity in entities.values()]
        if len(entities_description) == 0:
            return entities
        
        embeddings = await generate_embedding(self.emb_session, schema_description + entities_description, logger=self.logger)
        schema_embeddings = [embeddings[idx] for idx in range(len(schema_description))]
        entity_embeddings = [embeddings[idx] for idx in range(len(schema_description), len(schema_description) + len(entities_description))]
        # Split entities into batches
        for batch_start in range(0, len(entity_names), batch_size):
            batch = entity_names[batch_start : batch_start + batch_size]
            
            user_prompt = PROMPTS["align_entity"]["user"].format(input_text=context)
            entity_mapping = {}
            consecutive_idx = 0
            for batch_idx, entity_name in zip(range(batch_start, batch_start + len(batch)), batch):
                schema = entities[entity_name].extracted.type
                if kg_driver.check_entity_schema(schema):
                    similar_schema = [schema]
                else:
                    similar_schema = kg_driver.vector_search_entity_schema(
                        schema_embeddings[batch_idx],
                        top_k=top_k
                    )

                # Multiple entities may have exact match
                exact_match = kg_driver.get_entities(
                    type=entities[entity_name].extracted.type, 
                    name=entities[entity_name].extracted.name, 
                    top_k=top_k // 2, 
                    fuzzy=True
                )
                top_k_entities = exact_match[:min(top_k // 2, len(exact_match))]
                
                similar_match = kg_driver.get_entities(
                    embedding=entity_embeddings[batch_idx],
                    top_k=top_k - len(top_k_entities),
                    return_score=True
                )
                top_k_entities.extend([relevant_entity.entity for relevant_entity in similar_match if relevant_entity.entity not in top_k_entities])

                # Don't bother to ask LLM if there is nothing to do
                if len(top_k_entities) == 0 and kg_driver.check_entity_schema(schema):
                    continue

                entity_mapping[consecutive_idx] = {
                    "entity_id": batch_idx,
                    "top_k_entities": {f"ent_{j}": entity for j, entity in enumerate(top_k_entities)}
                }

                # Format user prompt
                similar_schema_str = f"[{','.join(entity_schema_to_text(schema) for schema in similar_schema)}]"
                top_k_entities_str = '\n'.join(f"{key}: {entity_to_text(entity)}" 
                                            for key, entity in entity_mapping[consecutive_idx]["top_k_entities"].items())
                top_k_entities_str = 'No entities.' if not top_k_entities_str else top_k_entities_str
                user_prompt += textwrap.dedent(f"""
                ## ID {consecutive_idx}. Candidate: {entities_description[batch_idx]}
                Synonym Entity Types:""") + similar_schema_str + \
                '\nEntities:' + top_k_entities_str + '\n'
                
                consecutive_idx += 1
            if consecutive_idx == 0:
                continue

            user_prompt += textwrap.dedent("""\
            Output Format (a flat JSON, you will need to escape any double quotes in the string to make the JSON valid):
            [{"id": 1, "aligned_type": "...", "reason": "...", "matched_entity": "ent_0"},
            {"id": 2, "aligned_type": "...", "reason": "...", "matched_entity": "ent_3"}]
            Output:""")

            # Prepare LLM call
            system_prompt = PROMPTS["align_entity"]["system"]
            prompts = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await generate_response(self.session, 
                prompts, 
                response_format={"type": "json_object"},
                logger=self.logger
            )
            self.logger.debug(system_prompt + "\n" + user_prompt + "\n" + response)

            # Process results
            found_ids = set()
            expected_ids = set(range(1, consecutive_idx))
            for _ in range(max_realign):
                batch_results = maybe_load_json(response)
            
                for result in batch_results:
                    try:
                        idx = result["id"]  # Convert back to index
                        entity_id = entity_mapping[idx]["entity_id"]
                        entity_name = entity_names[entity_id]
                        aligned_type = result["aligned_type"]
                        match = result["matched_entity"]
                                
                        top_k_entities_dict = entity_mapping[idx]["top_k_entities"]
                        entities[entity_name].extracted.type = aligned_type
                        entities[entity_name].aligned = top_k_entities_dict[match] \
                            if (match in top_k_entities_dict) else None
                        found_ids.add(idx)
                    except Exception as e:
                        self.logger.error("Encount error while parsing aligned entity", exc_info=True)
                    
                # Identify missing candidates
                missing_ids = list(expected_ids - found_ids)
                
                if missing_ids:
                    user_prompt = PROMPTS["align_entity"]["missing"].format(missing_entities=missing_ids)
                    prompts.extend([
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": user_prompt}
                    ])
                    response = await generate_response(self.session, 
                        prompts, 
                        response_format={"type": "json_object"},
                        logger=self.logger
                    )
                    self.logger.debug(user_prompt + "\n" + response)
                else:
                    break

        return entities

    @llm_retry(max_retries=MAX_RETRIES, default_output={})
    async def merge_entity(
        self,
        entities: OrderedDict[CandidateEntity], context: str, 
        batch_size: int = MERGE_ENTITY_BATCH_SIZE,
        max_remerge: int = 5
    ) -> OrderedDict[CandidateEntity]:
        """
        Merges extracted entities with existing KG entities to maintain up-to-date knowledge.

        Args:
            entities (OrderedDict[CandidateEntity]): A dictionary of candidate entities.
            batch_size (int): Number of entities to process per LLM call.

        Returns:
            OrderedDict[CandidateEntity]: A dictionary with merged entities.
        """

        entity_names = list(entities)

        # Process entities in batches
        for batch_start in range(0, len(entity_names), batch_size):
            batch = entity_names[batch_start : batch_start + batch_size]

            # Format batch prompt
            user_prompt = PROMPTS["merge_entity"]["user"].format(
                input_text=context
            )

            entity_mapping = {}
            for idx, entity_name in enumerate(batch):
                actual_idx = batch_start + idx
                
                extracted_entity = entities[entity_name].extracted
                kg_entity = entities[entity_name].aligned

                entity_mapping[f"{actual_idx + 1}"] = {
                    "extracted": extracted_entity,
                    "kg_entity": kg_entity
                }

                # Format entity pair string
                entity_pair_str = (
                    f"{actual_idx + 1}: [{entity_to_text(extracted_entity)}, {entity_to_text(kg_entity)}]"
                )

                user_prompt += entity_pair_str + "\n"
            user_prompt += textwrap.dedent("""\
            Output Format (a flat JSON):
            [{"id": 1, "desc": "entity_description", "props": {"key": ["val", "context"], ...}},
            {"id": 2, "desc": "entity_description", "props": {}}, ...]
            Output:""")
            
            # Prepare LLM call
            system_prompt = PROMPTS["merge_entity"]["system"]
            prompts = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM for merging
            response = await generate_response(self.session, 
                prompts, 
                max_tokens=10240,
                response_format={"type": "json_object"},
                logger=self.logger
            ) # Constraint by json schema is not stable: extra_body={"guided_json": output_schema})
            self.logger.debug(system_prompt + "\n" + user_prompt + "\n" + response)
            # logger.debug(user_prompt + "\n" + response)

            found_ids = set()
            expected_ids = set(range(batch_start + 1, batch_start + len(batch) + 1))
            for _ in range(max_remerge):
                batch_results = maybe_load_json(response)

                # Process results
                for result in batch_results:
                    try:
                        entity_id = result["id"]
                        entity_name = entity_names[int(entity_id) - 1]

                        merged_properties = {}
                        if result.get("props", None):
                            for k, v in result["props"].items():
                                if k not in RESERVED_KEYS:
                                    if isinstance(v, list):
                                        merged_properties[k] = {"v": v[0], "c": v[1]}
                                    else:
                                        merged_properties[k] = {"v": str(v), "c": None}
                        merged_entity = KGEntity(
                            id=entities[entity_name].aligned.id,
                            type=entities[entity_name].aligned.type,
                            name=entities[entity_name].aligned.name,
                            description=result["desc"],
                            properties=merged_properties,
                            ref=update_ref(entities[entity_name].aligned.ref, entities[entity_name].extracted.ref)
                        )
                        entities[entity_name].merged = merged_entity
                        
                        found_ids.add(entity_id)
                    except Exception as e:
                        self.logger.error("Encount error while parsing merged entity", exc_info=True)
                
                # Identify missing candidates
                missing_ids = list(expected_ids - found_ids)
                
                if missing_ids:
                    user_prompt = PROMPTS["merge_entity"]["missing"].format(missing_entities=missing_ids)
                    prompts.extend([
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": user_prompt}
                    ])
                    response = await generate_response(self.session, 
                        prompts, 
                        response_format={"type": "json_object"},
                        logger=self.logger
                    )
                    self.logger.debug(user_prompt + "\n" + response)
                else:
                    break

        return entities

    @llm_retry(max_retries=MAX_RETRIES, default_output={})
    async def align_relation(
        self,
        relations: OrderedDict[CandidateRelation],
        context: str,
        top_k: int = ALIGN_TOPK,
        batch_size: int = ALIGN_RELATION_BATCH_SIZE,
        max_realign: int = 5
    ) -> OrderedDict[RelevantEntity]:
        """
        Perform MIPS (Maximum Inner Product Search) in KG to align a list of extracted relations to KG relations.

        Args:
            relations (OrderedDict[CandidateRelation]): A dictionary of candidate relations.
            top_k (int): Specify the top-k entities assessed in KG.
            batch_size (int): Number of relations to process per LLM call.

        Returns:
            OrderedDict[CandidateRelation]: A list of aligned KG relations.
        """
        
        relation_names = list(relations)
        schema_description = [relation_schema_to_text((relation.extracted.source.type, relation.extracted.name, relation.extracted.target.type)) for relation in relations.values()]
        relations_description = [relation_to_text(relation.extracted, include_des=False) for relation in relations.values()]
        if len(relations_description) == 0:
            return relations
        
        embeddings = await generate_embedding(self.emb_session, schema_description + relations_description, logger=self.logger)
        schema_embeddings = [embeddings[idx] for idx in range(len(schema_description))]
        relation_embeddings = [embeddings[idx] for idx in range(len(schema_description), len(schema_description) + len(relations_description))]
        # Split entities into batches
        for batch_start in range(0, len(relation_names), batch_size):
            batch = relation_names[batch_start : batch_start + batch_size]

            user_prompt = PROMPTS["align_relation"]["user"].format(input_text=context)

            relation_mapping = {}
            consecutive_idx = 0
            for batch_idx, relation_name in zip(range(batch_start, batch_start + len(batch)), batch):
                schema = (relations[relation_name].extracted.source.type, 
                        relations[relation_name].extracted.name, 
                        relations[relation_name].extracted.target.type)

                if kg_driver.check_relation_schema(schema):
                    similar_schema = [schema]
                else:
                    similar_schema = kg_driver.vector_search_relation_schema(
                        schema_embeddings[batch_idx],
                        top_k=top_k
                    )
                
                exact_match = kg_driver.get_relations(
                    source=relations[relation_name].extracted.source, 
                    relation=relations[relation_name].extracted.name,
                    target=relations[relation_name].extracted.target
                )
                top_k_relations = exact_match[:min(top_k // 2, len(exact_match))]
                
                similar_match = kg_driver.get_relations(
                    embedding=relation_embeddings[batch_idx], 
                    top_k=top_k - len(top_k_relations), 
                    source=relations[relation_name].extracted.source,
                    target=relations[relation_name].extracted.target,
                    return_score=True
                )
                top_k_relations.extend([relevant_relation.relation for relevant_relation in similar_match if relevant_relation.relation not in top_k_relations])

                # Don't bother to ask LLM if there is nothing to do
                if len(top_k_relations) == 0 and kg_driver.check_relation_schema(schema):
                    continue

                relation_mapping[consecutive_idx] = {
                    "relation_id": batch_idx,
                    "top_k_relations": {f"rel_{j}": relation for j, relation in enumerate(top_k_relations)}
                }

                # Format user prompt
                similar_schema_str = '\n'.join(relation_schema_to_text(schema) for schema in similar_schema)
                top_k_relations_str = '\n'.join(f"{key}: {relation_to_text(relation)}" 
                                            for key, relation in relation_mapping[consecutive_idx]["top_k_relations"].items())
                top_k_relations_str = 'No relations.' if not top_k_relations_str else top_k_relations_str
                user_prompt += textwrap.dedent(f"""
                ## ID {consecutive_idx}. Candidate: {relations_description[batch_idx]}
                Synonym Relations:""") + similar_schema_str + \
                '\nRelations:' + top_k_relations_str + '\n'
                
                consecutive_idx += 1
            if consecutive_idx == 0:
                continue

            user_prompt += textwrap.dedent("""\
            Output Format (a flat JSON, you will need to escape any double quotes in the string to make the JSON valid):
            [{"id": 1, "aligned_name": "...", "reason": "...", "matched_relation": "rel_0"},
            {"id": 2, "aligned_name": "...", reason": "...", "matched_relation": "rel_3"}]'
            Output:""")

            # Prepare LLM call
            system_prompt = PROMPTS["align_relation"]["system"]
            prompts = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await generate_response(self.session, 
                prompts, 
                response_format={"type": "json_object"},
                logger=self.logger
            ) #, extra_body={"guided_json": output_schema})
            self.logger.debug(system_prompt + "\n" + user_prompt + "\n" + response)

            found_ids = set()
            expected_ids = set(range(1, consecutive_idx))
            for _ in range(max_realign):
                batch_results = maybe_load_json(response)

                # Process results
                for result in batch_results:
                    try:
                        idx = result["id"]  # Convert back to index
                        relation_id = relation_mapping[idx]["relation_id"]
                        relation_name = relation_names[relation_id]
                        aligned_name = result["aligned_name"]
                        match = result["matched_relation"]
                        top_k_relations_dict = relation_mapping[idx]["top_k_relations"]

                        relations[relation_name].extracted.name = aligned_name
                        relations[relation_name].aligned = top_k_relations_dict[match] \
                            if (match in top_k_relations_dict) else None
                        
                        found_ids.add(idx)
                    except Exception as e:
                        self.logger.error("Encount error while parsing aligned relation", exc_info=True)

                # Identify missing candidates
                missing_ids = list(expected_ids - found_ids)
                
                if missing_ids:
                    user_prompt = PROMPTS["align_relation"]["missing"].format(missing_relations=missing_ids)
                    prompts.extend([
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": user_prompt}
                    ])
                    response = await generate_response(self.session, 
                        prompts, 
                        response_format={"type": "json_object"},
                        logger=self.logger
                    )
                    self.logger.debug(user_prompt + "\n" + response)
                else:
                    break

        return relations

    @llm_retry(max_retries=MAX_RETRIES, default_output={})
    async def merge_relation(
        self,
        relations: OrderedDict[CandidateRelation], context: str,
        batch_size: int = MERGE_RELATION_BATCH_SIZE,
        max_remerge: int = 5
    ) -> OrderedDict[CandidateRelation]:
        """
        Merges extracted relations with existing KG relations to maintain up-to-date knowledge.

        Args:
            relations (OrderedDict[CandidateRelation]): A dictionary of candidate relations.
            batch_size (int): Number of relations to process per LLM call.

        Returns:
            OrderedDict[CandidateRelation]: A dictionary with merged relations.
        """
        
        relation_names = list(relations)

        # Process entities in batches
        for batch_start in range(0, len(relation_names), batch_size):
            batch = relation_names[batch_start : batch_start + batch_size]

            # Format batch prompt
            user_prompt = PROMPTS["merge_relation"]["user"].format(input_text=context)

            relation_mapping = {}
            for idx, relation_name in enumerate(batch):
                actual_idx = batch_start + idx
                
                extracted_relation = relations[relation_name].extracted
                kg_relation = relations[relation_name].aligned

                relation_mapping[f"{actual_idx + 1}"] = {
                    "extracted": extracted_relation,
                    "kg_relation": kg_relation
                }

                # Format entity pair string
                relation_pair_str = (
                    f"{actual_idx + 1}: [{
                                        relation_to_text(extracted_relation,
                                                        include_src_des=False,
                                                        include_src_prop=False,
                                                        include_dst_des=False,
                                                        include_dst_prop=False)
                                        }, {
                                        relation_to_text(kg_relation,
                                                        include_src_des=False,
                                                        include_src_prop=False,
                                                        include_dst_des=False,
                                                        include_dst_prop=False)
                                        }]"
                )

                user_prompt += relation_pair_str + "\n"
            user_prompt += textwrap.dedent("""\
            Output Format (a flat JSON):
            [{"id": 1, "desc": "relation_description", "props": {"key": ["val", "context"], ...}},
            {"id": 2, "desc": "relation_description", "props": {}}, ...]
            Output:""")

            # Prepare LLM call
            system_prompt = PROMPTS["merge_relation"]["system"]
            prompts = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            

            # Call LLM for merging
            response = await generate_response(self.session, 
                prompts,
                max_tokens=10240,
                response_format={"type": "json_object"},
                logger=self.logger
            ) # Constraint by json schema is not stable: extra_body={"guided_json": output_schema})
            self.logger.debug(system_prompt + "\n" + user_prompt + "\n" + response)
            # logger.debug(user_prompt + "\n" + response)

            found_ids = set()
            expected_ids = set(range(batch_start + 1, batch_start + len(batch) + 1))
            for _ in range(max_remerge):
                # Parse JSON response
                batch_results = maybe_load_json(response)

                # Process results
                for result in batch_results:
                    try:
                        relation_id = result["id"]
                        relation_name = relation_names[int(relation_id) - 1]

                        merged_properties = {}
                        if result["props"] is not None:
                            for k, v in result["props"].items():
                                if k not in RESERVED_KEYS:
                                    if isinstance(v, list):
                                        merged_properties[k] = {"v": v[0], "c": v[1]}
                                    else:
                                        merged_properties[k] = {"v": str(v), "c": None}
                        merged_relation = KGRelation(
                            id=relations[relation_name].aligned.id,
                            name=relations[relation_name].aligned.name,
                            source=relations[relation_name].aligned.source,
                            target=relations[relation_name].aligned.target,
                            description=result["desc"],
                            properties=merged_properties,
                            ref=update_ref(relations[relation_name].aligned.ref, relations[relation_name].extracted.ref)
                        )
                        relations[relation_name].merged = merged_relation

                        found_ids.add(relation_id)
                    except Exception as e:
                        self.logger.error("Encount error while parsing merged relation", exc_info=True)
                    
                # Identify missing candidates
                missing_ids = list(expected_ids - found_ids)
                
                if missing_ids:
                    user_prompt = PROMPTS["merge_relation"]["missing"].format(missing_relations=missing_ids)
                    prompts.extend([
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": user_prompt}
                    ])
                    response = await generate_response(self.session, 
                        prompts, 
                        response_format={"type": "json_object"},
                        logger=self.logger
                    )
                    self.logger.debug(user_prompt + "\n" + response)
                else:
                    break

        return relations

    def identify_missing_entities(
        self,
        responses: List[str]
    ) -> List[str]:
        """
        Preliminarily check if there are entities used in the extracted relations that are not extracted from the text.

        Args:
            responses (List[str]): A list of all LLM-generated JSON responses.

        Returns:
            List[str]: A list of non-extracted entities' names.
        """
        results = {}
        for response in responses:
            results.update(maybe_load_json(response))
        ext_entities_set = set()
        for key, result in results.items():
            if key.startswith("ent"):
                ext_entities_set.add(normalize_entity(result[1]))
        
        missing_entities = set()
        for key, result in results.items():
            if key.startswith("rel"):
                source = normalize_entity(result[0])
                target = normalize_entity(result[2])
                if source not in ext_entities_set:
                    missing_entities.add(source)
                if target not in ext_entities_set:
                    missing_entities.add(target)
        
        return list(missing_entities)

    def finalize_entities(
        self,
        ext_entities: dict, 
        current_time: datetime = None
    ):
        """
        Finalize the entity information from the merged entity and the aligned entity: 
            1. Update the description.
            2. Update the count, context, and last_seen of the property.

        Args:
            ext_entites (dict): The properties dictionary of the entity.
            current_time (datetime, optional): Current timestamp (defaults to UTC now).
        """
        current_time = datetime.fromisoformat(current_time) if current_time else datetime.now(timezone.utc)
            
        for entity_name in ext_entities:
            if self.config.get("merge_entity", False) and ext_entities[entity_name].aligned:
                ext_entities[entity_name].final = deepcopy(ext_entities[entity_name].aligned)
                
                extracted = ext_entities[entity_name].extracted
                aligned = ext_entities[entity_name].aligned
                merged = ext_entities[entity_name].merged
                final = ext_entities[entity_name].final

                final.ref = extracted.ref
                if not merged: continue
                final.description = merged.description
                final.ref = merged.ref
                for property_name, merged_value_dict in merged.properties.items():
                    if not property_name: continue
                    final_values_dict = final.properties.setdefault(property_name, {})
                    
                    merged_value = merged_value_dict.get('v', None)
                    # Sometimes, the LLM provide a list as value
                    if not merged_value or isinstance(merged_value, list): continue
                    merged_context = merged_value_dict.get('c', None)

                    merged_value = normalize_value(merged_value)
                    merged_context = normalize_value(merged_value)
                    final_values_dict[merged_value] = final_values_dict.get(merged_value, {})
                    final_values_dict[merged_value]["context"] = merged_context
                    
                    # Update up-vote and last-seen when 
                    # 1. property reinforced by text;
                    # 2. property appears the first time;
                    # 3. property value appears the first time.
                    if (property_name in extracted.properties) or \
                    (property_name not in aligned.properties) or \
                    (merged_value not in aligned.properties[property_name]):
                        final_values_dict[merged_value]["count"] = final_values_dict[merged_value].get("count", 0) + 1
                        final_values_dict[merged_value]["last_seen"] = current_time.isoformat()
            else:
                ext_entities[entity_name].final = ext_entities[entity_name].extracted

    def finalize_relations(
        self,
        ext_relations: dict, 
        current_time: datetime = None
    ):
        """
        Finalize the relation information from the merged relation and the aligned relation: 
            1. Update the description.
            2. Update the count, context, and last_seen of the property.

        Args:
            ext_entites (dict): The properties dictionary of the entity.
            current_time (datetime, optional): Current timestamp (defaults to UTC now).
        """
        current_time = datetime.fromisoformat(current_time) if current_time else datetime.now(timezone.utc)
            
        for relation_name in ext_relations:
            if self.config.get("merge_relation", False) and ext_relations[relation_name].aligned:
                ext_relations[relation_name].final = deepcopy(ext_relations[relation_name].aligned)
                
                extracted = ext_relations[relation_name].extracted
                aligned = ext_relations[relation_name].aligned
                merged = ext_relations[relation_name].merged
                final = ext_relations[relation_name].final

                final.name = extracted.name # We aligned this in align_relation()
                final.ref = extracted.ref
                if not merged: continue
                final.description = merged.description # We merged this in merge_relation()
                final.ref = merged.ref
                for property_name, merged_value_dict in merged.properties.items():
                    if not property_name: continue
                    final_values_dict = final.properties.setdefault(property_name, {})

                    merged_value = merged_value_dict.get('v', None)
                    # Sometimes, the LLM provide a list as value
                    if not merged_value or not isinstance(merged_value, list): continue
                    merged_context = merged_value_dict.get('c', None)

                    merged_value = normalize_value(merged_value)
                    merged_context = normalize_value(merged_value)
                    final_values_dict[merged_value] = final_values_dict.get(merged_value, {})
                    final_values_dict[merged_value]["context"] = merged_context
                    
                    # Update up-vote and last-seen when 
                    # 1. property reinforced by text;
                    # 2. property appears the first time;
                    # 3. property value appears the first time.
                    if (property_name in extracted.properties) or \
                    (property_name not in aligned.properties) or \
                    (merged_value not in aligned.properties[property_name]):
                        final_values_dict[merged_value]["count"] = final_values_dict[merged_value].get("count", 0) + 1
                        final_values_dict[merged_value]["last_seen"] = current_time.isoformat()
            else:
                ext_relations[relation_name].final = ext_relations[relation_name].extracted

    def _snapshot_token_usage(self) -> Dict[str, int]:
        """Return a snapshot of the current token usage counter."""
        return deepcopy(token_counter.get_token_usage()) if token_counter else {}

    def _compute_token_usage_delta(self, start_usage: Dict[str, int]) -> Dict[str, int]:
        """Compute token deltas relative to a start snapshot."""
        if not token_counter:
            return {}
        end_usage = token_counter.get_token_usage()
        keys = set(start_usage.keys()) | set(end_usage.keys())
        return {key: end_usage.get(key, 0) - start_usage.get(key, 0) for key in keys}

    def _normalize_newlines(self, text: str) -> str:
        # Normalize CRLF to LF for consistent boundary detection
        return text.replace('\r\n', '\n').replace('\r', '\n')

    def _find_all(self, haystack: str, needle: str, start: int = 0) -> list[int]:
        """Return all start indices of exact 'needle' occurrences in 'haystack' from 'start'."""
        if not needle:
            return []
        i, out = start, []
        while True:
            i = haystack.find(needle, i)
            if i == -1:
                break
            out.append(i)
            i += 1  # allow overlapping (safe since anchors are short)
        return out

    def extract_paragraph_by_anchors(
        self,
        raw_text: str,
        paragraph_start_anchor: str,
        paragraph_end_anchor: str,
        prefer_shortest: bool = True
    ):
        """
        Locate a paragraph using two anchors:
        - paragraph_start_anchor: a verbatim substring from the FIRST 5–30 chars of the paragraph
        - paragraph_end_anchor:   a verbatim substring from the LAST 5–30 chars of the paragraph

        Strategy:
        1) Normalize newlines.
        2) Find all occurrences of start_anchor; for each, find all occurrences of end_anchor AFTER it.
        3) Choose the (start, end) pair that yields the shortest valid span (default) or the first valid span.
        4) Expand to paragraph boundaries using two-or-more newlines.
        """
        if not paragraph_start_anchor or not paragraph_end_anchor:
            # Return empty string if anchors are missing instead of raising error
            # This allows processing to continue even if LLM fails to generate anchors
            return ""

        text = self._normalize_newlines(raw_text)

        start_positions = self._find_all(text, paragraph_start_anchor, start=0)
        if not start_positions:
            return ""

        best_pair: Optional[tuple[int, int]] = None
        best_len = float('inf')

        for s in start_positions:
            # Find end occurrences after s (must end AFTER start)
            end_positions = self._find_all(text, paragraph_end_anchor, start=s + len(paragraph_start_anchor))
            for e in end_positions:
                hi = e + len(paragraph_end_anchor)
                span_len = hi - s
                if prefer_shortest:
                    if span_len < best_len:
                        best_len = span_len
                        best_pair = (s, hi)
                else:
                    if best_pair is None:
                        best_pair = (s, hi)
                        break  # take the first valid
            if not prefer_shortest and best_pair is not None:
                break

        if best_pair is None:
            return ""

        anchor_lo, anchor_hi = best_pair

        # Expand to paragraph boundaries
        # para_lo = char after the last paragraph separator BEFORE anchor_lo
        # para_hi = char at the next paragraph separator AFTER anchor_hi (or end of text)
        # We scan using the regex split positions.
        # Faster approach: find nearest '\n\n' boundaries around the anchor span.
        # Left boundary:
        left_bound = text.rfind("\n\n", 0, anchor_lo)
        para_lo = 0 if left_bound == -1 else left_bound + 2
        # Right boundary:
        right_bound = text.find("\n\n", anchor_hi)
        para_hi = len(text) if right_bound == -1 else right_bound

        # Safety: If anchors straddle multiple paragraphs due to bad anchors, we still return the minimal
        # paragraph span given the nearest boundaries.
        paragraph_text = text[para_lo:para_hi]

        return paragraph_text

    @llm_retry(max_retries=MAX_RETRIES, default_output=(OrderedDict(), OrderedDict()))
    async def update_kg(
        self,
        context: str, 
        created_at: datetime, 
        modified_at: datetime, 
        ref: str = "",
        generation_temperature=0.6,
        generate_max_tokens=MAX_GENERATION_TOKENS, 
        stages = STAGES, 
        domain: str = None
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        Perform a multi-stage knowledge graph (KG) update by extracting entities and relations from a text document
        using an LLM, aligning them with existing KG entities/relations, and incrementally merging/upserting them.

        This function runs asynchronously and supports multi-step entity and relation comprehension via few-shot prompting,
        self-reflection, and continued reasoning. It includes logic for robust JSON parsing, retry on failure, and 
        internal logging of time and output.

        Args:
            context (str): The raw text content (up to ~60k tokens) from which to extract KG information.
            created_at (datetime): The timestamp indicating the creation time of the source text/document.
            modified_at (datetime): The timestamp of the latest update to the source text/document.
            ref (str, optional): A reference link to the text content.
            max_retries (int, optional): Maximum number of retries if LLM decoding or KG update fails. Default is 3.
            generate_max_tokens (int, optional): Maximum number of tokens for each LLM generation. Default is 20000.
            stages (int, optional): Number of multi-stage prompt continuation steps (multi-gleaning) for iterative extraction. Default is 2.
            verbose (bool, optional): Whether to print LLM prompts, responses, and diagnostic info. Default is True.

        Returns:
            Tuple[OrderedDict, OrderedDict]:
                - logs: A dictionary containing raw LLM prompts, completions, and final extracted outputs.
                - elapsed_time: A dictionary summarizing the processing time of each stage.

        Notes:
            - Uses prompt chaining with system + few-shot examples, followed by continuation and self-reflection prompts.
            - All extracted entities are aligned and merged before being upserted into the KG.
            - All extracted relations are likewise aligned and merged with KG relations (if applicable), with fallback to insertion.
            - Embeddings are used for similarity matching during upsert and alignment.
            - Temporal information is preserved using created_at and modified_at.
        """
        
        task_name = asyncio.current_task().get_name()  # Get async task name

        logs = OrderedDict()
        elapsed_time = OrderedDict()
        token_usage_start = self._snapshot_token_usage()
        self.logger.info(f"[{task_name}] Performing first-stage text comprehension...")

        entity_types = kg_driver.get_node_types()
        relation_types = kg_driver.get_edge_types()

        delta_time = EXPECTED_TIME # For example, 10 mins
        expected_finish_time = time.time() + delta_time
        ################################### Entity/Relation Extraction ###################################
        last_time = start_time = time.time()

        entity_types = entity_types if len(PROMPTS["DEFAULT_ENTITY_TYPES"]) < len(entity_types) else PROMPTS["DEFAULT_ENTITY_TYPES"]
        hints = PROMPTS["domain_hints"].get(domain, "No hints available.") if domain else ""
        system_prompt = PROMPTS["extraction"]["system"].format(
            entity_types=",".join(entity_types),
            relation_types=",".join(relation_types[:min(5, len(relation_types))]),
            hints=hints
        )
        for idx, example in enumerate(PROMPTS["extraction"]["examples"]):
            system_prompt += textwrap.dedent(f"""\
            Example {idx + 1}:
            Text: {example['text']}
            Output: {example['output']}
            Explanation: {example['explanation']}
            
            """)
        
        user_message = PROMPTS["extraction"]["user"].format(
            input_text=context,
            hints=hints
        )
        self.logger.debug(user_message)

        # Perform first-stage extraction
        prompts = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        response = await generate_response(self.session, 
            prompts, 
            max_tokens=generate_max_tokens, 
            temperature=generation_temperature,
            response_format={"type": "json_object"},
            logger=self.logger,
            expected_time=expected_finish_time
        )
        self.logger.debug(response)
        prompts.append({"role": "assistant", "content": response})
        logs['extraction_0'] = user_message + response
        elapsed_time['extraction_0'], last_time = time.time() - last_time, time.time()

        # Perform multi-stage gleaning
        for step in range(1, stages):
            self.logger.info(f"[{task_name}] Performing stage-{step} text comprehension...")

            missing_entities = self.identify_missing_entities([prompts[idx]['content'] for idx in range(2, 2 * step + 1, 2)])
            user_prompt = (PROMPTS["extraction"]["missing_entities"].format(missing_entities=missing_entities) if missing_entities else "") + \
                            PROMPTS["extraction"]["continue_extraction"]
            prompts.append({"role": "user", "content": user_prompt})
            self.logger.debug(user_prompt)

            response = await generate_response(self.session, 
                prompts, 
                max_tokens=generate_max_tokens,
                temperature=generation_temperature,
                response_format={"type": "json_object"},
                logger=self.logger,
                expected_time=expected_finish_time
            )
            self.logger.debug(response)
            prompts.append({"role": "assistant", "content": response})

            logs[f'extraction_{step}'] = user_prompt + '\n' + response
            elapsed_time[f'extraction_{step}'], last_time = time.time() - last_time, time.time()
        
        # Perform self-check
        if self.config.get("self_reflection", False):
            user_prompt = PROMPTS["extraction"]["self-reflection"].format(
                hints=hints
            )
            self.logger.debug(user_prompt)
            prompts.append({"role": "user", "content": user_prompt})
            response = await generate_response(self.session, 
                prompts, 
                max_tokens=generate_max_tokens,
                temperature=generation_temperature,
                response_format={"type": "json_object"},
                logger=self.logger
            )
            prompts.append({"role": "assistant", "content": response})
            logs['self-reflection'] = user_prompt + '\n' + response
            elapsed_time['self-reflection'], last_time = time.time() - last_time, time.time()
            self.logger.debug(response)

        ################################### Parse Extracted Entities ###################################
        self.logger.info(f"[{task_name}] Parsing extracted entities...")

        responses = [prompts[idx]['content'] for idx in range(2, len(prompts), 2)]
        results = {}
        for response in responses:
            results.update(maybe_load_json(response))

        ext_entities: Dict[str, CandidateEntity] = OrderedDict()

        for key, result in results.items():
            if key.startswith("ent") and len(result):
                candidate = CandidateEntity(
                    extracted=KGEntity(
                        id="",
                        type=normalize_entity_type(result[0]),
                        name=normalize_entity(result[1]),
                        description=result[2],
                        paragraph=self.extract_paragraph_by_anchors(context, result[3], result[4]),
                        created_at=created_at,
                        modified_at=modified_at,
                        properties=kg_driver.get_properties(result[5], created_at)  if len(result) > 5 else {},
                        ref=ref
                    )
                )
                ext_entities[candidate.extracted.name] = candidate

        if self.config.get("align_entity", False):
            ext_entities = await self.align_entity(ext_entities, context)
            elapsed_time['align_entity'], last_time = time.time() - last_time, time.time()

        if self.config.get("merge_entity", False):
            await self.merge_entity({key: value for key, value in ext_entities.items() if value.aligned is not None}, context)
            elapsed_time['merge_entity'], last_time = time.time() - last_time, time.time()

        self.finalize_entities(ext_entities, created_at)

        upsert_entities_dict = {ext_entity.extracted.name: ext_entity.final for ext_entity in ext_entities.values()}
        # New entities being inserted into KG
        upsert_entities_dict = await kg_driver.upsert_entities(upsert_entities_dict)
        for key, value in upsert_entities_dict.items():
            ext_entities[key].final = value

        output = ""
        for ext_entity in ext_entities.values():
            output += \
            f"E: {entity_to_text(ext_entity.extracted, include_id=True)}\n" + \
            f"A: {entity_to_text(ext_entity.aligned, include_id=True)}\n" + \
            f"F: {entity_to_text(ext_entity.final, include_id=True)}\n\n"
        self.logger.debug(output)
        logs[f'extracted_entities'] = output

        
        ################################### Parse Extracted Relations ###################################
        self.logger.info(f"[{task_name}] Parsing extracted relations...")

        ext_relations: Dict[str, CandidateRelation] = OrderedDict()
        for key, result in results.items():
            if key.startswith("rel") and len(result):
                source = ext_entities.get(normalize_entity(result[0]))
                target = ext_entities.get(normalize_entity(result[2]))
                if source is None or target is None:
                    continue
                candidate = CandidateRelation(
                    extracted=KGRelation(
                        id="",
                        name=normalize_relation(result[1]),
                        source=source.final,
                        target=target.final,
                        description=result[3],
                        paragraph=self.extract_paragraph_by_anchors(context, result[4], result[5]),
                        properties=kg_driver.get_properties(result[6], created_at) if len(result) > 6 else {},
                        ref=ref
                    )
                )
                relation_name = relation_to_text(candidate.extracted, 
                                                include_des=False,
                                                include_prop=False, 
                                                include_src_des=False,
                                                include_src_prop=False,
                                                include_dst_des=False,
                                                include_dst_prop=False)
                ext_relations[relation_name] = candidate
        
        if self.config.get("align_relation", False):
            ext_relations = await self.align_relation(ext_relations, context)
            elapsed_time['align_relation'], last_time = time.time() - last_time, time.time()

        if self.config.get("merge_relation", False):
            await self.merge_relation({key: value for key, value in ext_relations.items() if value.aligned is not None}, context)
            elapsed_time['merge_relation'], last_time = time.time() - last_time, time.time()

        self.finalize_relations(ext_relations, created_at)

        upsert_relations_dict = {relation_name: ext_relation.final for relation_name, ext_relation in ext_relations.items()}
        # New relations being inserted into KG
        upsert_relations_dict = await kg_driver.upsert_relations(upsert_relations_dict)
        for key, value in upsert_relations_dict.items():
            ext_relations[key].final = value

        output = ""
        for ext_relation in ext_relations.values():
            output += \
            f"E: {relation_to_text(ext_relation.extracted, include_src_prop=False, include_dst_prop=False)}\n" + \
            f"A: {relation_to_text(ext_relation.aligned, include_src_prop=False, include_dst_prop=False)}\n" + \
            f"F: {relation_to_text(ext_relation.final, include_src_prop=False, include_dst_prop=False)}\n\n"
        self.logger.debug(output)
        logs[f'extracted_relations'] = output

        last_time = time.time()
        upsert_relations_dict = await kg_driver.upsert_relations(upsert_relations_dict)
        elapsed_time['upsert'], last_time = time.time() - last_time, time.time()

        ################################### Done ###################################
        self.logger.info(f"[{task_name}] Extracted {len(ext_entities)} entities and {len(ext_relations)} relations from the text.")
        elapsed_time[f'extracted_num_ent_rel'] = f"{len(ext_entities)} entities and {len(ext_relations)} relations"

        logs['modified_at'] = modified_at
        logs['created_at'] = created_at
        token_usage_delta = self._compute_token_usage_delta(token_usage_start)
        logs['token_usage'] = token_usage_delta
        
        self.logger.info(f"[{task_name}] Finished KG update from text.")
        elapsed_time['processing_time'], last_time = time.time() - start_time, time.time()
        elapsed_time['token_usage'] = token_usage_delta
        
        return logs, elapsed_time

    # Test purpose
    # async def mock_update_kg(
    #     self,
    #     id: str = "",
    #     doc: str = "",
    #     created_at: datetime = None,
    #     modified_at: datetime = None,
    #     ref: str = None,
    #     max_chunk_size: int = 0,
    #     min_chunk_size: int = 0
    # ) -> Tuple[OrderedDict, OrderedDict]:
    #     task_name = asyncio.current_task().get_name()  # Get async task name

    #     logs = OrderedDict()
    #     elapsed_time = OrderedDict()
    #     self.logger.info(f"[{task_name}] Performing first-stage text comprehension...")

    #     last_time = start_time = time.time()

    #     await asyncio.sleep(1)

    #     logs['extraction_0'] = ""
    #     elapsed_time['extraction_0'], last_time = time.time() - last_time, time.time()
    #     logs['self-reflection'] = ""
    #     elapsed_time['self-reflection'], last_time = time.time() - last_time, time.time()

    #     self.logger.info(f"[{task_name}] Parsing extracted entities...")
    #     elapsed_time['align_entity'], last_time = time.time() - last_time, time.time()
    #     elapsed_time['merge_entity'], last_time = time.time() - last_time, time.time()

    #     self.logger.info(f"[{task_name}] Parsing extracted relations...")
    #     elapsed_time['align_relation'], last_time = time.time() - last_time, time.time()
    #     elapsed_time['merge_relation'], last_time = time.time() - last_time, time.time()

    #     self.logger.info(f"[{task_name}] Extracted {0} entities and {0} relations from the text.")
    #     elapsed_time[f'extracted_num_ent_rel'] = f"{0} entities and {0} relations"

    #     logs['modified_at'] = modified_at
    #     logs['created_at'] = created_at
    #     modified_at = parse_timestamp(modified_at)
    #     created_at = parse_timestamp(created_at)

    #     self.logger.info(f"[{task_name}] Finished KG update from text.")
    #     elapsed_time['processing_time'], last_time = time.time() - start_time, time.time()

    #     return logs, elapsed_time

    async def process_doc(
        self,
        id: str = "",
        doc: str = "",
        created_at: datetime = None,
        modified_at: datetime = None,
        ref: str = None,
        domain: str = None,
        max_chunk_size: int = MAX_CHUNK,
        min_chunk_size: int = MIN_CHUNK
    ):
        """
        Process a single document (context) and update the knowledge graph accordingly.

        Args:
            id (str): An unique identifier of the document.
            context (str): A self-contained document use to update the knowledge graph.
            created_at (datetime, optional): The timestamp the document is created at, use to set the "created_at" attribute in knowledge graph.
            modified_at (datetime, optional): The timestamp the document is modified at, use to set the "modified_at" attribute in knowledge graph.
            ref (str, optional): A reference link to the document.
            verbose (bool, optional): Whether return a log or not.
            max_chunk_size (int, optional): Target maximum chunk size.
            min_chunk_size (int, optional): Target minimum chunk size, avoid chunks that are too small

        Returns:
            logs: 
        """
        if not doc:
            return 
        
        total_length = len(doc)
        self.logger.info(f"Updating KG using doc {id}")

        start_time = time.time()
        # Find the best split count where the largest chunk is ≤ MAX_CHUNK_SIZE but ≥ MIN_CHUNK_SIZE
        best_splits = 1
        for num_splits in range(1, 100):  # Try different split counts
            avg_chunk_size = math.ceil(total_length / num_splits)
            if min_chunk_size <= avg_chunk_size <= max_chunk_size:
                best_splits = num_splits
                break  # Stop once we find a reasonable split

        # Perform equal splitting
        chunk_size = math.ceil(total_length / best_splits)
        chunks = [doc[i:i+chunk_size] for i in range(0, total_length, chunk_size)]

        self.logger.info(f"number of chunks: {chunk_size}")
        self.logger.info(f"split_time: {time.time()-start_time}")        
        logs = [] # TODO: update this to a dedicated logger
        # Process each chunk separately
        for chunk_id, chunk in enumerate(chunks):
            # try:
            #     log, elapsed_time = await asyncio.wait_for(self.update_kg(
            #             chunk, 
            #             created_at=created_at, 
            #             modified_at=modified_at, 
            #             ref=ref, 
            #             domain=domain
            #         ),
            #         timeout=10_000
            #     )
            # except asyncio.TimeoutError:
            #     self.logger.add_stat({
            #         "id": id,
            #         "chunk_id": str(round(chunk_id / len(chunks), 2)),
            #         "chunk_length": len(chunk),
            #         "extraction_0": 0,
            #         "extraction_1": 0,
            #         "align_entity": 0,
            #         "merge_entity": 0,
            #         "align_relation": 0,
            #         "merge_relation": 0,
            #         "processing_time": 0,
            #         "extracted_num_ent_rel": "time out",
            #         "finished_at": time.time()
            #     })
            # else:
            #     # log, elapsed_time = await mock_update_kg(chunk, created_at=created_at, modified_at=modified_at)
            #     logs.append(log)  # Store logs for each chunk
            
            #     self.logger.add_stat({
            #         "id": id,
            #         "chunk_id": str(round(chunk_id / len(chunks), 2)),
            #         "chunk_length": len(chunk),
            #         "extraction_0": elapsed_time.get("extraction_0", 0),
            #         "extraction_1": elapsed_time.get("extraction_1", 0),
            #         "align_entity": elapsed_time.get("align_entity", 0),
            #         "merge_entity": elapsed_time.get("merge_entity", 0),
            #         "align_relation": elapsed_time.get("align_relation", 0),
            #         "merge_relation": elapsed_time.get("merge_relation", 0),
            #         "processing_time": elapsed_time.get("processing_time", 0),
            #         "extracted_num_ent_rel": elapsed_time.get("extracted_num_ent_rel", ""),
            #         "finished_at": time.time()
            #     })
            
            log, elapsed_time = await self.update_kg(
                chunk, 
                created_at=created_at, 
                modified_at=modified_at, 
                ref=ref, 
                domain=domain
            )
            # log, elapsed_time = await mock_update_kg(chunk, created_at=created_at, modified_at=modified_at)
            logs.append(log)  # Store logs for each chunk
        
            self.logger.add_stat({
                "id": id,
                "chunk_id": str(round(chunk_id / len(chunks), 2)),
                "chunk_length": len(chunk),
                "extraction_0": elapsed_time.get("extraction_0", 0),
                "extraction_1": elapsed_time.get("extraction_1", 0),
                "align_entity": elapsed_time.get("align_entity", 0),
                "merge_entity": elapsed_time.get("merge_entity", 0),
                "align_relation": elapsed_time.get("align_relation", 0),
                "merge_relation": elapsed_time.get("merge_relation", 0),
                "processing_time": elapsed_time.get("processing_time", 0),
                "extracted_num_ent_rel": elapsed_time.get("extracted_num_ent_rel", ""),
                "token_usage": elapsed_time.get("token_usage", {}),
                "finished_at": time.time()
            })
            
            self.logger.update_progress({"last_doc_total": round(time.time() - start_time, 2)})




        # # TODO: For KG-Gen evaluation! Delete this
        # print(f"Done! Exporting KG to {int(id) + 1}.json...")
        # import json
        # async def export_kg_to_json(output_file="kg.json"):
        #     # Fetch distinct entity names
        #     entity_query = """
        #     MATCH (n)
        #     RETURN DISTINCT n.name AS name
        #     """
        #     entities_result = await kg_driver.run_query_async(entity_query)
        #     entities = [record["name"] for record in entities_result if record["name"]]

        #     # Fetch all relation triples
        #     relation_query = """
        #     MATCH (h)-[r]->(t)
        #     RETURN h.name AS head, type(r) AS rel, t.name AS tail
        #     """
        #     relations_result = await kg_driver.run_query_async(relation_query)
        #     triples = []
        #     edge_set = set()
        #     for record in relations_result:
        #         if record["head"] and record["tail"]:
        #             rel = record["rel"].replace("_", " ").lower()
        #             triples.append([record["head"], rel, record["tail"]])
        #             edge_set.add(rel)

        #     # Write to JSON
        #     with open(output_file, "w") as f:
        #         json.dump({
        #             "entities": sorted(set(entities)),
        #             "relations": triples,
        #             "edges": sorted(edge_set)
        #         }, f, indent=2)
        
        # os.makedirs(f"TopK-{ALIGN_TOPK}", exist_ok=True) 
        # await export_kg_to_json(f"TopK-{ALIGN_TOPK}/{int(id) + 1}.json")

        # print("Clean KG...")
        # clear_query = """
        # MATCH (n) detach delete n
        # """
        # await kg_driver.run_query_async(clear_query)
        # print("Export done.")