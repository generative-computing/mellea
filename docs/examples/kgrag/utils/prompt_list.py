import textwrap

def get_default_prompts():
    PROMPTS = {}

    PROMPTS["DEFAULT_LANGUAGE"] = "English"
    PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
    PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
    PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
    PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Person", "Movie", "Tv", "Award", "Geo", "Genre", "Year", "Organization", "Event"]

    # From CRAG Benchmark: https://github.com/facebookresearch/CRAG/blob/main/models/rag_knowledge_graph_baseline.py
    PROMPTS['kg_topic_entity'] = {
        "system": textwrap.dedent("""\
        You are given a Query and Query Time. Do the following: 

        1) Determine the domain the query is about. The domain should be one of the following: "sports", "movie", and "other". If none of the domain applies, use "other". Use "domain" as the key in the result json. 

        2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

        For `movie` queries, these are possible keys:
        - `movie_name`: name of the movie
        - `person`: person name related to moves
        - `year`: if the query is about movies released in a specific year, extract the year

        For `sports` queries, these are possible keys:
        - `sport_type`: one of `basketball`, `soccer`, `other`
        - `tournament`: such as NBA, World Cup, Olympic.
        - `team`: teams that user interested in.
        - `datetime`: time frame that user interested in. When datetime is not explicitly mentioned, use `Query Time` as default. 

        For `other` queries, these are possible keys:
        -  `main_entity`: extract the main entity of the query. 
                                
        Return the results in a FLAT json. 

        *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
                                
        EXAMPLE JSON OUTPUT:
        {"domain": "movie", "movie_name": "Mount Everest"}
        """),

        "user": textwrap.dedent("""\
        Question: {query}
        Query Time: {query_time}
        EXAMPLE JSON OUTPUT:
        {{"domain": "movie", "movie_name": "Mount Everest"}}
        Output:
        """)
    }

    PROMPTS["domain_hints"] = {
        "movie": textwrap.dedent("""\
        1. The movie award is usually announced one year after the movie's release. Be cautious that the references may use different conventions to represent the year information. 
        When comparing the award-holding and winning years, please ensure that you always use the year in which the event occurred (the actual award-holding year).
        """),

        "sports": "",

        "music": "",

        "open": "",

        "yearly question": "You only need to provide answer up to the granularity of year."
    }

    return PROMPTS

PROMPTS = get_default_prompts()