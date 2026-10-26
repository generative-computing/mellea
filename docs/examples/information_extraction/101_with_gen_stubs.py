# pytest: ollama, e2e
"""Named entity extraction with generative stubs.

Demonstrates using a `@generative` function with a `list[str]` return type to
extract all person names from a text passage declaratively.

Prerequisites:
    ollama pull granite4.2:3b
    ollama serve

Usage:
    uv run python docs/examples/information_extraction/101_with_gen_stubs.py
"""

from mellea import generative, start_session

m = start_session()


@generative
def extract_all_person_names(doc: str) -> list[str]:
    """Given a document, extract names of ALL mentioned persons. Return these names as list of strings."""


# ref: https://www.nytimes.com/2012/05/20/world/world-leaders-at-us-meeting-urge-growth-not-austerity.html
NYTimes_text = "CAMP DAVID, Md. — Leaders of the world's richest countries banded together on Saturday to press Germany to back more pro-growth policies to halt the deepening debt crisis in Europe, as President Obama for the first time gained widespread support for his argument that Europe, and the United States by extension, cannot afford Chancellor Angela Merkel's one-size-fits-all approach emphasizing austerity."

person_names = extract_all_person_names(m, doc=NYTimes_text)

print(f"person_names = {person_names}")
# out: person_names = ['President Obama', 'Angela Merkel']
