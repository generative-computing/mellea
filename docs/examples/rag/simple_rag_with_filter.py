# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faiss-cpu",
#     "sentence_transformers",
#     "mellea"
# ]
# ///
from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer

from mellea import generative, start_session
from mellea.backends import model_ids

docs = [
    "The capital of France is Paris. Paris is known for its Eiffel Tower.",
    "The Amazon River is the largest river by discharge volume of water in the world.",
    "Mount Everest is the Earth's highest mountain above sea level, located in the Himalayas.",
    "The Louvre Museum in Paris houses the Mona Lisa.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, process, and generate human language.",
    "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China.",
    "The solar system consists of the Sun and everything bound to it by gravity, including the eight planets, dwarf planets, and countless small Solar System bodies.",
    "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, after Mercury.",
    "The human heart has four chambers: two atria and two ventricles.",
    "Photosynthesis is the process used by plants, algae, and cyanobacteria to convert light energy into chemical energy.",
    "The internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.",
    "Python is a high-level, general-purpose programming language.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
]


def create_index(model, ds: list[str]) -> IndexFlatIP:
    print("running encoding... ")
    embeddings = model.encode(docs)
    print("running embeddings... ")
    dimension = embeddings.shape[1]
    index = IndexFlatIP(dimension)
    index.add(embeddings)  # type:ignore
    print("done indexing.")
    return index


def query_index(model, idx: IndexFlatIP, query: str, ds: list[str], k: int = 5) -> list:
    query_embedding = model.encode([query])
    distances, indices = idx.search(query_embedding, k=k)
    return [ds[i] for i in indices[0]]


@generative
def is_answer_relevant_to_question(answer: str, question: str) -> bool:
    """For the given question, determine whether the answer is relevant or not."""


if __name__ == "__main__":
    query = "How are AI and NLP related?"

    # Create a simple embedding index
    print("loading Embedding model and index data...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = create_index(embedding_model, docs)

    # Query the index
    print("Query Embedding model...")
    results = query_index(embedding_model, index, query, docs)
    results_str = "\n".join([f"=> {r}" for r in results])
    print(f"results:\n {results_str}\n ====")
    del embedding_model  # help GC

    # Create Mellea session
    m = start_session(model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B)

    # Check for each document from retrieval if it is actually relevant
    print("running filter.. ")
    relevant_answers = []
    for doc in results:
        is_it = is_answer_relevant_to_question(m, answer=doc, question=query)
        if is_it:
            relevant_answers.append(doc)
        else:
            print(f"skipping: {doc}")

    # Run final answer generation from here
    print("running generation...")
    answer = m.instruct(
        "Provided the documents in the context, answer the question: `{{query}}`",
        user_variables={"query": query},
        grounding_context={f"doc{i}": doc for i, doc in enumerate(relevant_answers)},
    )

    # Print results answer
    print(f"== answer == \n{answer.value}\n ====")
