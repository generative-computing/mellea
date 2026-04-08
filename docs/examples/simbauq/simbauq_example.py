# pytest: openai, e2e, qualitative, skip

"""SIMBA-UQ Sampling Strategy Example.

This example demonstrates the SIMBAUQSamplingStrategy using both confidence
estimation methods:

1. **Aggregation** (data-free) - Computes pairwise similarity between all
   generated samples and aggregates them into per-sample confidence scores.
   The sample with the highest confidence is selected.

2. **Classifier** (trained) - Uses a random forest classifier trained on
   labeled examples to predict P(correct) for each sample based on its
   pairwise similarity features.

Both methods generate multiple samples across different temperature values,
compute a similarity matrix, and select the most confident response.

The example uses RITSBackend with granite-4.0-micro. To run:

    uv run python docs/examples/simbauq/simbauq_example.py

Requires:
    RITS_API_KEY environment variable or hardcoded key below.
"""

import os

import numpy as np
from dotenv import load_dotenv

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.core import SamplingResult
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling.simbauq import SIMBAUQSamplingStrategy

load_dotenv()  # Load environment variables from .env file if present

RITS_API_KEY = os.getenv("RITS_API_KEY", None)


def make_session() -> MelleaSession:
    """Create a MelleaSession with RITSBackend."""
    from mellea_ibm.rits import RITS, RITSBackend  # type: ignore[import-not-found]

    backend = RITSBackend(
        model_name=RITS.GRANITE_4_MICRO,
        api_key=RITS_API_KEY,
        model_options={ModelOption.MAX_NEW_TOKENS: 100},
    )
    return MelleaSession(backend, ctx=ChatContext())


def print_results(result: SamplingResult) -> None:
    """Print detailed results from a SIMBA-UQ sampling run."""
    meta = result.result._meta["simba_uq"]
    confidences = meta["all_confidences"]
    temperatures = meta["temperatures_used"]
    sim_matrix = np.array(meta["similarity_matrix"])

    # --- Best response ---
    print("=" * 70)
    print("BEST RESPONSE")
    print("=" * 70)
    print(f"  Index:       {result.result_index}")
    print(f"  Confidence:  {meta['confidence']:.4f}")
    print(f"  Method:      {meta['confidence_method']}")
    print(f"  Metric:      {meta['similarity_metric']}")
    print(f"  Aggregation: {meta['aggregation']}")
    print(f"  Text:\n    {result.result!s}")
    print()

    # --- All samples ---
    print("=" * 70)
    print("ALL SAMPLES")
    print("=" * 70)
    print(f"{'Idx':>4}  {'Temp':>5}  {'Conf':>8}  {'Text'}")
    print("-" * 70)
    for i, mot in enumerate(result.sample_generations):
        text = str(mot).replace("\n", " ")
        truncated = (text[:100] + "...") if len(text) > 100 else text
        marker = " <-- best" if i == result.result_index else ""
        print(
            f"{i:>4}  {temperatures[i]:>5.2f}  {confidences[i]:>8.4f}  "
            f"{truncated}{marker}"
        )
    print()

    # --- Similarity matrix ---
    n = sim_matrix.shape[0]
    print("=" * 70)
    print("SIMILARITY MATRIX")
    print("=" * 70)
    header = "      " + "".join(f"  [{i:>2}]  " for i in range(n))
    print(header)
    for i in range(n):
        row = f"[{i:>2}]  " + "".join(f"  {sim_matrix[i, j]:.3f} " for j in range(n))
        print(row)
    print()


def run_aggregation_example() -> None:
    """Run SIMBA-UQ with data-free similarity aggregation."""
    print("\n>>> AGGREGATION CONFIDENCE METHOD <<<\n")

    m = make_session()

    strategy = SIMBAUQSamplingStrategy(
        temperatures=[0.3, 0.5, 0.7, 1.0],
        n_per_temp=3,
        similarity_metric="rouge",
        confidence_method="aggregation",
        aggregation="harmonic_mean",
    )

    result: SamplingResult = m.instruct(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        strategy=strategy,
        return_sampling_results=True,
    )

    print(f"Total samples generated: {len(result.sample_generations)}")
    print_results(result)

    del m


def run_classifier_example() -> None:
    """Run SIMBA-UQ with a trained random forest classifier."""
    print("\n>>> CLASSIFIER CONFIDENCE METHOD <<<\n")

    m = make_session()

    # Synthetic training data: 3 groups of 12 samples (4 temps * 3 per temp).
    # Each group has mostly "correct" similar answers and a few outliers.
    training_samples = [
        [
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "France's capital city is Paris.",
            "Paris, the capital of France.",
            "The capital city of France is Paris.",
            "France has Paris as its capital.",
            "Paris serves as France's capital.",
            "In France, Paris is the capital.",
            "The French capital is Paris.",
            "Bananas are a yellow fruit.",
            "Dogs are loyal pets.",
            "The ocean is very deep.",
        ],
        [
            "Water boils at 100 degrees Celsius.",
            "At 100C water reaches boiling point.",
            "The boiling point of water is 100 degrees.",
            "Water boils when heated to 100C.",
            "100 degrees Celsius is water's boiling point.",
            "Boiling occurs at 100C for water.",
            "Water starts boiling at one hundred degrees.",
            "At 100 degrees water boils.",
            "The temperature for boiling water is 100C.",
            "Cats like to sleep a lot.",
            "Mountains can be very high.",
            "Stars shine in the night sky.",
        ],
        [
            "Python is a programming language.",
            "Python is a popular programming language.",
            "The Python programming language is widely used.",
            "Python is used for programming.",
            "Programming in Python is common.",
            "Python is a well-known language for coding.",
            "Many developers use Python.",
            "Python is a general-purpose language.",
            "The language Python is popular.",
            "Pizza originated in Italy.",
            "Rain falls from clouds.",
            "Books contain many pages.",
        ],
    ]
    training_labels = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    ]

    strategy = SIMBAUQSamplingStrategy(
        temperatures=[0.3, 0.5, 0.7, 1.0],
        n_per_temp=3,
        similarity_metric="rouge",
        confidence_method="classifier",
        training_samples=training_samples,
        training_labels=training_labels,
    )

    result: SamplingResult = m.instruct(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        strategy=strategy,
        return_sampling_results=True,
    )

    print(f"Total samples generated: {len(result.sample_generations)}")
    print_results(result)

    del m


def main():
    """Run both SIMBA-UQ confidence estimation examples."""
    run_aggregation_example()
    print("\n" + "=" * 70 + "\n")
    run_classifier_example()


if __name__ == "__main__":
    main()
