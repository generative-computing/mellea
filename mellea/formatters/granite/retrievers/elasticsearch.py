# SPDX-License-Identifier: Apache-2.0

"""Classes and functions that implement the ElasticsearchRetriever."""


class ElasticsearchRetriever:
    """Retriever for documents hosted on an ElasticSearch server."""

    def __init__(self, corpus_name: str, host: str, **kwargs: dict[str, int]):
        """Initialize ElasticsearchRetriever.

        :param hosts: Full url:port to the Elasticsearch server.
        :param kwargs: Additional kwargs to pass to the Elasticsearch class.
        """
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

        :param limit: Max number of documents to retrieve.
        :param query: Query string for retrieving documents.
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
        """Run a query and return results."""
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
