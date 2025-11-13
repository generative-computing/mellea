from __future__ import annotations
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

# Type aliases
Embedding = torch.Tensor
Text = str

class Retriever:
    def __init__(self, dataset_name: str, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.dataset = load_dataset(dataset_name)

        # Assume your dataset has a text column and possibly precomputed embeddings
        if "embedding" in self.dataset["train"].features:
            self.embeddings: Embedding = torch.tensor(self.dataset["train"]["embedding"])
        else:
            texts: List[Text] = self.dataset["train"]["informal_description"]
            self.embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    def search(self, query: Text, k: int = 5) -> List[Tuple[Text, float]]:
        query_emb: Embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores: Embedding = util.cos_sim(query_emb, self.embeddings)[0]

        # Get top-k results
        top_results = torch.topk(cosine_scores, k)
        top_indices = top_results.indices.tolist()
        top_scores = top_results.values.tolist()

        results = [
            (self.dataset["train"][i]["name"], float(top_scores[j]))
            for j, i in enumerate(top_indices)
        ]
        return results

# Example usage
if __name__ == "__main__":
    engine = Retriever("FrenzyMath/mathlib_informal_v4.19.0")  # hypothetical dataset
    query = "Prove that every continuous function on [0,1] is bounded."
    top_matches = engine.search(query, k=3)

    for text, score in top_matches:
        print(f"{score:.4f} | {text[:80]}...")
