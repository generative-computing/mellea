# SPDX-License-Identifier: Apache-2.0

"""Classes and functions that implement the ElasticsearchRetriever."""

from typing import Any


class ElasticsearchRetriever:
    """Retriever for documents hosted on an Elasticsearch server.

    Queries an Elasticsearch index using the ELSER sparse-vector model to
    retrieve the top-k matching documents for a given natural language query.

    Attributes:
        hosts (str): Full ``url:port`` connection string to the Elasticsearch
            server; stored from the ``host`` constructor argument.

    Args:
        corpus_name (str): Name of the Elasticsearch index to query.
        host (str): Full ``url:port`` connection string to the Elasticsearch
            server.
        **kwargs (Any): Additional keyword arguments forwarded to the
            ``Elasticsearch`` client constructor.
    """

    def __init__(self, corpus_name: str, host: str, **kwargs: Any):
        """Initialize ElasticsearchRetriever with index name and connection details."""
        # Third Party
        from elasticsearch import Elasticsearch  # type: ignore[import-not-found]

        self.corpus_name = corpus_name

        # Hosts is the minimum required param to init a connection to the
        # Elasticsearch server, so make it explicit here.
        self.hosts = host
        self.kwargs = kwargs

        self.es = Elasticsearch(hosts=host, **kwargs)

    def create_es_body(self, limit, query):
        """Create a query body for Elasticsearch.

        Args:
            limit (int): Maximum number of documents to retrieve.
            query (str): Natural language query string used for ELSER
                sparse-vector retrieval.
        """
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": {
                        "text_expansion": {
                            "ml.tokens": {
                                "model_id": ".elser_model_1",
                                "model_text": query,
                            }
                        }
                    }
                }
            },
        }
        return body

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Run a query against the Elasticsearch index and return top-k results.

        Args:
            query (str): Natural language query string to search for.
            top_k (int): Maximum number of documents to return. Defaults to ``5``.

        Returns:
            list[dict]: List of matching documents, each with keys ``doc_id``,
                ``text``, and ``score``.
        """
        body = self.create_es_body(top_k, query)

        retriever_results = self.es.search(index=self.corpus_name, body=body)
        hits = retriever_results["hits"]["hits"]

        # Format for the processor.
        documents = []
        for hit in hits:
            document = {
                "doc_id": hit["_id"],
                "text": hit["_source"]["text"],
                "score": str(hit["_score"]),  # Non-string values crash vLLM
            }
            documents.append(document)

        return documents
