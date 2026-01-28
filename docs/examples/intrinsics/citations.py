"""Example usage of the citations intrinsic for RAG applications.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/citations.py
```
"""

import json

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
context = ChatContext().add(
    Message(
        "user",
        "How does Murdoch's expansion in Australia compare to his expansion "
        "in New Zealand?",
    )
)
assistant_response = (
    "Murdoch expanded in Australia and New Zealand by acquiring and expanding local "
    "newspapers. I do not have information about his expansion in New Zealand after "
    "purchasing The Dominion."
)
documents = [
    Document(
        doc_id="1",
        text="Keith Rupert Murdoch was born on 11 March 1931 in Melbourne, Australia, "
        "the son of Sir Keith Murdoch (1885-1952) and Dame Elisabeth Murdoch (nee "
        "Greene; 1909-2012). He is of English, Irish, and Scottish ancestry. Murdoch's "
        "parents were also born in Melbourne. Keith Murdoch was a war correspondent "
        "and later a regional newspaper magnate owning two newspapers in Adelaide, "
        "South Australia, and a radio station in a faraway mining town. Following his "
        "father's death, when he was 21, Murdoch returned from Oxford to take charge "
        "of the family business News Limited, which had been established in 1923. "
        "Rupert Murdoch turned its Adelaide newspaper, The News, its main asset, into "
        "a major success. He began to direct his attention to acquisition and "
        "expansion, buying the troubled Sunday Times in Perth, Western Australia "
        "(1956) and over the next few years acquiring suburban and provincial "
        "newspapers in New South Wales, Queensland, Victoria and the Northern "
        "Territory, including the Sydney afternoon tabloid, The Daily Mirror (1960). "
        'The Economist describes Murdoch as "inventing the modern tabloid", as he '
        "developed a pattern for his newspapers, increasing sports and scandal "
        "coverage and adopting eye-catching headlines. Murdoch's first foray outside "
        "Australia involved the purchase of a controlling interest in the New Zealand "
        "daily The Dominion. In January 1964, while touring New Zealand with friends "
        "in a rented Morris Minor after sailing across the Tasman, Murdoch read of a "
        "takeover bid for the Wellington paper by the British-based Canadian newspaper "
        "magnate, Lord Thomson of Fleet. On the spur of the moment, he launched a "
        "counter-bid. A four-way battle for control ensued in which the 32-year-old "
        "Murdoch was ultimately successful. Later in 1964, Murdoch launched The "
        "Australian, Australia's first national daily newspaper, which was based "
        "first in Canberra and later in Sydney. In 1972, Murdoch acquired the Sydney "
        "morning tabloid The Daily Telegraph from Australian media mogul Sir Frank "
        "Packer, who later regretted selling it to him. In 1984, Murdoch was appointed "
        "Companion of the Order of Australia (AC) for services to publishing. In 1999, "
        "Murdoch significantly expanded his music holdings in Australia by acquiring "
        "the controlling share in a leading Australian independent label, Michael "
        "Gudinski's Mushroom Records; he merged that with Festival Records, and the "
        "result was Festival Mushroom Records (FMR). Both Festival and FMR were "
        "managed by Murdoch's son James Murdoch for several years.",
    ),
    Document(
        doc_id="2",
        text="This document has nothing to do with Rupert Murdoch. This document is "
        "two sentences long.",
    ),
]


result = rag.find_citations(assistant_response, documents, context, backend)
print(f"Result of citations intrinsic:\n{json.dumps(result, indent=2)}")
