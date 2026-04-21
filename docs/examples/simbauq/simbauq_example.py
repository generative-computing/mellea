# pytest: ollama, llm, qualitative

"""SIMBA-UQ Sampling Strategy Example.

This example demonstrates the SIMBAUQSamplingStrategy using both confidence
estimation methods:

1. **Aggregation** (data-free) - Computes pairwise similarity between all
   generated samples and aggregates them into per-sample confidence scores.
   The sample with the highest confidence is selected.

2. **Classifier with synthetic data** - Uses a random forest classifier
   trained on hand-coded labeled examples.

3. **Classifier with HF data** - Same classifier method, but training data
   is generated live from TriviaQA (short QA) and SAMSum (dialogue
   summarization) via Ollama.

4. **Classifier with pre-trained classifier** - Trains a RandomForestClassifier
   externally using HF-generated data, then passes the fitted object directly
   to SIMBAUQSamplingStrategy via the ``classifier=`` argument.

Both methods generate multiple samples across different temperature values,
compute a similarity matrix, and select the most confident response.

The example uses OllamaModelBackend with granite4:micro. To run:

    ollama serve
    uv run python docs/examples/simbauq/simbauq_example.py
"""

import logging
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]
from tqdm import tqdm

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import FancyLogger, SamplingResult
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling.simbauq import SIMBAUQSamplingStrategy

# Number of training groups collected per dataset.
# Each group has len(temperatures) * n_per_temp samples.
# Increase for better classifier signal at the cost of more LLM calls.
N_TRAINING_GROUPS = 5


def make_session() -> MelleaSession:
    """Create a MelleaSession with OllamaModelBackend."""
    FancyLogger.get_logger().setLevel(logging.WARNING)
    backend = OllamaModelBackend(model_options={ModelOption.MAX_NEW_TOKENS: 150})
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


def generate_training_data(
    session: MelleaSession,
    temperatures: list[float],
    n_per_temp: int,
    dataset: str = "triviaqa",
    similarity_metric: Literal[
        "rouge", "jaccard", "sbert", "difflib", "levenshtein"
    ] = "rouge",
    threshold: float = 0.5,
    n_groups: int = N_TRAINING_GROUPS,
) -> tuple[list[list[str]], list[list[int]]]:
    """Generate classifier training data from a single HF dataset via Ollama.

    For each dataset item, generates one group of ``len(temperatures) *
    n_per_temp`` responses at the configured temperature schedule. Each
    response is labelled 1 if its similarity to the ground-truth reference
    meets ``threshold``, 0 otherwise. Groups where all labels are identical
    are discarded as they provide no classifier signal.

    Args:
        session: Active MelleaSession to use for generation.
        temperatures: Temperature schedule (must match inference-time schedule).
        n_per_temp: Samples per temperature (must match inference-time value).
        dataset: HF dataset to use. One of ``"triviaqa"`` (short QA) or
            ``"samsum"`` (dialogue summarization).
        similarity_metric: Metric used for labelling (should match the
            ``similarity_metric`` passed to SIMBAUQSamplingStrategy).
        threshold: Similarity score >= threshold → label 1.
        n_groups: Target number of valid groups to collect.

    Returns:
        Tuple of (training_samples, training_labels), each a list of groups
        with exactly ``len(temperatures) * n_per_temp`` entries per group.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HF training data generation. "
            "Install it with: pip install datasets"
        )

    group_size = len(temperatures) * n_per_temp
    print(f"Generating training data with {group_size} samples per group ")

    # Reused solely for _compute_similarity — no training data needed at init.
    scorer = SIMBAUQSamplingStrategy(
        similarity_metric=similarity_metric, confidence_method="aggregation"
    )

    def _load_triviaqa(n: int) -> list[dict]:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="train", streaming=True)
        items = []
        for row in ds:
            ref = row.get("answer", {}).get("value", "")
            if not ref:
                continue
            items.append(
                {
                    "prompt": f"Answer the following question briefly: {row['question']}",
                    "reference": ref,
                }
            )
            if len(items) >= n:
                break
        return items

    def _load_samsum(n: int) -> list[dict]:
        ds = load_dataset("samsum", split="train", streaming=True)
        items = []
        for row in ds:
            dialogue = row.get("dialogue", "")[:1000]
            ref = row.get("summary", "")
            if not dialogue or not ref:
                continue
            items.append(
                {
                    "prompt": f"Summarize the following dialogue in one sentence:\n\n{dialogue}",
                    "reference": ref,
                }
            )
            if len(items) >= n:
                break
        return items

    loaders = {"triviaqa": _load_triviaqa, "samsum": _load_samsum}
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(loaders)}")

    print(f"  Collecting {n_groups} training groups from {dataset}...")
    items = loaders[dataset](n_groups * 3)

    training_samples: list[list[str]] = []
    training_labels: list[list[int]] = []
    collected = 0

    with tqdm(total=n_groups, desc=f"Generating [{dataset}]", unit="group") as pbar:
        for item in items:
            if collected >= n_groups:
                break

            responses: list[str] = []
            for temp in temperatures:
                for _ in range(n_per_temp):
                    try:
                        mot = session.instruct(
                            item["prompt"],
                            model_options={
                                ModelOption.TEMPERATURE: temp,
                                ModelOption.MAX_NEW_TOKENS: 150,
                            },
                        )
                        responses.append(str(mot))
                    except Exception:
                        responses.append("")

            scores = [
                scorer._compute_similarity(r, item["reference"]) for r in responses
            ]
            labels = [1 if s >= threshold else 0 for s in scores]

            if len(set(labels)) < 2:
                pbar.set_postfix_str(
                    f"discarded (scores {min(scores):.2f}–{max(scores):.2f}, threshold={threshold})"
                )
                continue

            training_samples.append(responses)
            training_labels.append(labels)
            collected += 1
            pbar.update(1)

    return training_samples, training_labels


def train_classifier(
    training_samples: list[list[str]],
    training_labels: list[list[int]],
    similarity_metric: Literal[
        "rouge", "jaccard", "sbert", "difflib", "levenshtein"
    ] = "rouge",
    clf_max_depth: int = 4,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier on similarity features extracted from training data.

    Uses the same feature extraction as SIMBAUQSamplingStrategy internally
    (_compute_similarity_matrix + _extract_features), ensuring the feature
    space is identical at train and inference time.

    Args:
        training_samples: List of groups, each with the same number of samples
            as ``len(temperatures) * n_per_temp`` used at inference time.
        training_labels: Binary correctness labels (0/1) matching
            ``training_samples``.
        similarity_metric: Similarity metric for feature extraction.
        clf_max_depth: Maximum tree depth for the random forest.

    Returns:
        Fitted RandomForestClassifier.
    """
    extractor = SIMBAUQSamplingStrategy(
        similarity_metric=similarity_metric, confidence_method="aggregation"
    )

    x_train: list[np.ndarray] = []
    y_train: list[int] = []
    for samples, labels in zip(training_samples, training_labels):
        sim_matrix = extractor._compute_similarity_matrix(samples)
        for i, label in enumerate(labels):
            x_train.append(extractor._extract_features(sim_matrix, i))
            y_train.append(label)

    clf = RandomForestClassifier(max_depth=clf_max_depth, random_state=0)
    clf.fit(x_train, y_train)
    return clf


def run_aggregation_example(session: MelleaSession) -> None:
    """Run SIMBA-UQ with data-free similarity aggregation."""
    print("\n>>> AGGREGATION CONFIDENCE METHOD <<<\n")

    strategy = SIMBAUQSamplingStrategy(
        temperatures=[0.3, 0.5, 0.7, 1.0],
        n_per_temp=3,
        similarity_metric="sbert",
        confidence_method="aggregation",
        aggregation="mean",
    )

    result: SamplingResult = session.instruct(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        strategy=strategy,
        return_sampling_results=True,
    )

    print(f"Total samples generated: {len(result.sample_generations)}")
    print_results(result)


def run_classifier_synthetic_example(session: MelleaSession) -> None:
    """Run SIMBA-UQ classifier with hand-coded synthetic training data."""
    print("\n>>> CLASSIFIER CONFIDENCE METHOD (synthetic training data) <<<\n")

    temperatures = [0.3, 0.5, 0.7, 1.0]
    n_per_temp = 3

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
        temperatures=temperatures,
        n_per_temp=n_per_temp,
        similarity_metric="rouge",
        confidence_method="classifier",
        training_samples=training_samples,
        training_labels=training_labels,
    )

    result: SamplingResult = session.instruct(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        strategy=strategy,
        return_sampling_results=True,
    )

    print(f"Total samples generated: {len(result.sample_generations)}")
    print_results(result)


def run_classifier_hf_example(
    session: MelleaSession,
    dataset: str,
    similarity_metric: Literal["rouge", "jaccard", "sbert", "difflib", "levenshtein"],
    threshold: float,
) -> None:
    """Run SIMBA-UQ classifier with training data generated from an HF dataset."""
    print(f"\n>>> CLASSIFIER CONFIDENCE METHOD (HF / {dataset}) <<<\n")

    temperatures = [0.3, 0.5, 0.7, 1.0]
    n_per_temp = 3

    print(f"Generating training data from {dataset}...")
    training_samples, training_labels = generate_training_data(
        session,
        temperatures,
        n_per_temp,
        dataset=dataset,
        similarity_metric=similarity_metric,
        threshold=threshold,
    )
    print(
        f"Training data ready: {len(training_samples)} groups of {len(temperatures) * n_per_temp} samples each.\n"
    )

    if not training_samples:
        print(
            f"  No valid training groups collected (threshold={threshold} may be "
            "too strict or too loose for this model/dataset combination). "
            "Try adjusting --threshold."
        )
        return

    print("--- Training examples sample ---")
    for group_idx, (samples, labels) in enumerate(
        zip(training_samples, training_labels)
    ):
        correct = [s for s, lab in zip(samples, labels) if lab == 1]
        incorrect = [s for s, lab in zip(samples, labels) if lab == 0]
        print(f"  Group {group_idx}:")
        if correct:
            print(f"    [correct]   {correct[0]!r}")
        if incorrect:
            print(f"    [incorrect] {incorrect[0]!r}")
    print()

    strategy = SIMBAUQSamplingStrategy(
        temperatures=temperatures,
        n_per_temp=n_per_temp,
        similarity_metric=similarity_metric,
        confidence_method="classifier",
        training_samples=training_samples,
        training_labels=training_labels,
    )

    result: SamplingResult = session.instruct(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        strategy=strategy,
        return_sampling_results=True,
    )

    print(f"Total samples generated: {len(result.sample_generations)}")
    print_results(result)


def run_classifier_pretrained_example(
    session: MelleaSession,
    dataset: str,
    similarity_metric: Literal["rouge", "jaccard", "sbert", "difflib", "levenshtein"],
    threshold: float,
) -> None:
    """Run SIMBA-UQ classifier with a pre-trained RandomForestClassifier."""
    print(f"\n>>> CLASSIFIER CONFIDENCE METHOD (pre-trained / {dataset}) <<<\n")

    temperatures = [0.3, 0.5, 0.7, 1.0]
    n_per_temp = 3

    print(f"Generating training data from {dataset}...")
    training_samples, training_labels = generate_training_data(
        session,
        temperatures,
        n_per_temp,
        dataset=dataset,
        similarity_metric=similarity_metric,
        threshold=threshold,
    )
    print(
        f"Training data ready: {len(training_samples)} groups of {len(temperatures) * n_per_temp} samples each.\n"
    )

    if not training_samples:
        print(
            f"  No valid training groups collected (threshold={threshold} may be "
            "too strict or too loose for this model/dataset combination). "
            "Try adjusting the threshold."
        )
        return

    print("--- Training examples sample ---")
    for group_idx, (samples, labels) in enumerate(
        zip(training_samples, training_labels)
    ):
        correct = [s for s, lab in zip(samples, labels) if lab == 1]
        incorrect = [s for s, lab in zip(samples, labels) if lab == 0]
        print(f"  Group {group_idx}:")
        if correct:
            print(f"    [correct]   {correct[0]!r}")
        if incorrect:
            print(f"    [incorrect] {incorrect[0]!r}")
    print()

    clf = train_classifier(
        training_samples, training_labels, similarity_metric=similarity_metric
    )
    print(f"Classifier trained: {clf}\n")

    strategy = SIMBAUQSamplingStrategy(
        temperatures=temperatures,
        n_per_temp=n_per_temp,
        similarity_metric=similarity_metric,
        confidence_method="classifier",
        classifier=clf,
    )

    result: SamplingResult = session.instruct(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        strategy=strategy,
        return_sampling_results=True,
    )

    print(f"Total samples generated: {len(result.sample_generations)}")
    print_results(result)


def main():
    """Run all SIMBA-UQ confidence estimation examples."""
    dataset = "triviaqa"
    similarity_metric = "sbert"
    threshold = 0.2

    m = make_session()
    run_aggregation_example(m)
    print("\n" + "=" * 70 + "\n")
    run_classifier_synthetic_example(m)
    print("\n" + "=" * 70 + "\n")
    run_classifier_hf_example(
        m, dataset=dataset, similarity_metric=similarity_metric, threshold=threshold
    )
    print("\n" + "=" * 70 + "\n")
    run_classifier_pretrained_example(
        m, dataset=dataset, similarity_metric=similarity_metric, threshold=threshold
    )


if __name__ == "__main__":
    main()
