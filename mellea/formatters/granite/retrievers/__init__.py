# SPDX-License-Identifier: Apache-2.0

"""Support for retrieving documents from various sources."""

# First Party
from ..base.io import Retriever
from ..retrievers import util
from ..retrievers.elasticsearch import ElasticsearchRetriever
from ..retrievers.embeddings import (
    InMemoryRetriever,
    compute_embeddings,
    write_embeddings,
)

# Expose public symbols at `mellea.formatters.granite.io.retrievers` to save users from
# typing
__all__ = [
    "ElasticsearchRetriever",
    "InMemoryRetriever",
    "Retriever",
    "compute_embeddings",
    "util",
    "write_embeddings",
]
