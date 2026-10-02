from __future__ import annotations
import os
from typing import List, Dict, Any # Changed to support dictionary returns
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

class Retriever:
    """
    A retriever class that efficiently searches a dataset using pre-computed embeddings.
    It returns detailed information about each match, including its file path.
    """
    def __init__(
        self,
        dataset_name: str = "FrenzyMath/mathlib_informal_v4.19.0",
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_cache_path: str = "mathlib_embeddings.pt"
    ):
        if not os.path.exists(embedding_cache_path):
            raise FileNotFoundError(
                f"Embedding cache not found at '{embedding_cache_path}'. "
                f"Please run the 'precompute_embeddings.py' script first."
            )

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.dataset = load_dataset(dataset_name)

        print(f"Loading embeddings from cache: {embedding_cache_path}")
        self.embeddings = torch.load(embedding_cache_path, map_location=self.device)

        print(f"Loaded {self.embeddings.shape[0]} embeddings.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the most similar documents to the query.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  contains the name, score, file_path, and
                                  start_line_no of a match.
        """
        query_emb = self.model.encode(query, convert_to_tensor=True, device=self.device)
        cosine_scores = util.cos_sim(query_emb, self.embeddings)[0]

        top_results = torch.topk(cosine_scores, k=k)
        top_indices = top_results.indices.tolist()
        top_scores = top_results.values.tolist()

        results = []
        for j, i in enumerate(top_indices):
            item = self.dataset["train"][i]
            results.append(item)

        return results

    def search_queries(self, queries: List[str], k: int = 5) -> List[Dict[str, Any]]:
        results = []
        for query_obj in queries:
            results += self.search(query_obj, k)
        return results

if __name__ == "__main__":
    print("--- Initializing Retriever from pre-computed embeddings ---")
    engine = Retriever(
        dataset_name="FrenzyMath/mathlib_informal_v4.19.0",
        embedding_cache_path="mathlib_embeddings.pt"
    )
    print("\n--- Retriever Initialized. Ready for queries. ---\n")

    query = "Prove that every continuous function on [0,1] is bounded."
    print(f"Query: '{query}'")
    top_matches = engine.search(query, k=3)

    for match in top_matches:
        print(match)
        print("-" * 20)
