"""Simple Example of information extraction with Mellea using generative slots."""

from mellea import generative, start_session
from mellea.backends import model_ids

m = start_session()


@generative
def extract_all_person_names(doc: str) -> list[str]:
    """Given a document, extract names of ALL mentioned persons. Return these names as list of strings."""


# ref: https://www.nytimes.com/2012/05/20/world/world-leaders-at-us-meeting-urge-growth-not-austerity.html
NYTimes_text = "CAMP DAVID, Md. — Leaders of the world's richest countries banded together on Saturday to press Germany to back more pro-growth policies to halt the deepening debt crisis in Europe, as President Obama for the first time gained widespread support for his argument that Europe, and the United States by extension, cannot afford Chancellor Angela Merkel's one-size-fits-all approach emphasizing austerity."

person_names = extract_all_person_names(m, doc=NYTimes_text)

print(f"person_names = {person_names}")
# out: person_names = ['President Obama', 'Angela Merkel']
