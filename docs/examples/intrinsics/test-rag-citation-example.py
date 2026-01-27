
from typing import Any
from mellea.backends.adapters.adapter import AdapterMixin, GraniteCommonAdapter
from mellea.backends.adapters.catalog import AdapterType
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_options import ModelOption
from mellea.core.backend import Backend
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.context import ChatContext

# Combine generation with citations.

def rag_generate_with_references(
        query: str,
        docs: list[Document],
        backend: Backend,
) -> tuple[ModelOutputThunk, Any]:
    """Answers a query with an answer with citations."""

    # Make sure the backend is of the right type.
    assert isinstance(backend, AdapterMixin), "backend must support intrinsics for this function"

    # Check for the adapter before even generating.
    adapter = GraniteCommonAdapter(
        "citations", adapter_type=AdapterType.LORA, base_model_name=backend.base_model_name
    )
    try:
        if adapter.qualified_name not in backend.list_adapters():
            backend.add_adapter(adapter)
    except Exception:
        raise Exception('necessary adapter "citations" not found')

    # TODO: Check that the documents are formatted correctly / as expected?
    response, next_ctx = mfuncs.instruct(
        description=query,
        backend=backend,
        context=ChatContext(),
        grounding_context={
            str(doc.doc_id): doc for doc in docs # TODO: fix the doc_id here.
        },
        strategy=None,
        model_options={ModelOption.MAX_NEW_TOKENS: 100}
    )

    # `find_citations` expects the response to not yet be in the context.
    citation_ctx = next_ctx.previous_node
    assert isinstance(citation_ctx, ChatContext), "context must stay a chat context if sampled"


    # TODO: How to handle case where intrinsic isn't there.
    # -> sentence splitter plus relevance tester as batch of async calls

    citations = rag.find_citations(str(response.value), docs, citation_ctx, backend)
    return response, citations

if __name__ == "__main__":
    query = "How does Murdoch's expansion in Australia compare to his expansion in New Zealand?"
    docs = [
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
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    response, citations = rag_generate_with_references(query, docs, backend)
    print(response)
    print(citations)

