# SIMBA-UQ Sampling Strategy

Confidence-aware sample selection using the SIMBA-UQ framework
(Bhattacharjya et al., 2025). Generates multiple samples across a range of
temperatures and selects the one with the highest estimated confidence.

**Paper:** [SIMBA UQ: Similarity-Based Aggregation for Uncertainty Quantification in Large Language Models](https://arxiv.org/abs/2510.13836)

## Files

### simbauq_example.py

Complete example demonstrating all four confidence estimation variants with
Ollama and granite-4.0-micro:

1. **Aggregation** — data-free, no training data required.
2. **Classifier (synthetic)** — trained on hand-coded labeled groups.
3. **Classifier (HF data)** — training data generated live from a Hugging Face
   dataset via Ollama. Calls `generate_training_data()` which streams items
   from TriviaQA or SAMSum, generates `len(temperatures) * n_per_temp` responses
   per item at the configured temperature schedule, and labels each response by
   similarity to the ground-truth reference. Groups where all labels are
   identical are discarded. Requires `pip install datasets`.
4. **Classifier (pre-trained)** — same HF-generated training data, but the
   `RandomForestClassifier` is trained externally via `train_classifier()` and
   passed directly to `SIMBAUQSamplingStrategy` via the `classifier=` argument.
   Useful when you want to persist, inspect, or swap the classifier independently
   of the sampling strategy.

### simbauq_data.py

Standalone CLI script for generating larger training datasets offline. Supports
all 9 HF datasets (3 QA, 3 summarization, 3 generative) and writes one JSON
file per dataset to an output directory. See `--help` for options.

## Architecture

```
User Query
    |
    v
Generate N samples (across temperatures)
    |
    v
Compute pairwise similarity matrix (N x N)
    |
    +---> [Aggregation] Aggregate similarities per sample -> confidence
    |
    +---> [Classifier]  Extract features per sample -> RF predicts P(correct)
    |
    v
Select sample with highest confidence
    |
    v
Result (with confidence metadata in mot.meta["simba_uq"])
```

## Confidence Methods

### 1. Aggregation (data-free)

No training data required. For each sample, computes its similarity to every
other sample, then aggregates those values into a confidence score. Samples
that are more similar to the majority get higher confidence.

```python
from mellea.stdlib.sampling.simbauq import SIMBAUQSamplingStrategy

strategy = SIMBAUQSamplingStrategy(
    temperatures=[0.3, 0.5, 0.7, 1.0],
    n_per_temp=3,
    similarity_metric="rouge",
    confidence_method="aggregation",
    aggregation="mean",
)

result = m.instruct("Your query here", strategy=strategy, return_sampling_results=True)
```

### 2. Classifier (trained)

Uses a random forest classifier trained on labeled examples. The classifier
learns to predict P(correct) from pairwise similarity features. Provide
either training data or a pre-trained sklearn classifier.

Each training group must have exactly `len(temperatures) * n_per_temp` samples
so the feature vectors match at inference time.

**Option A — synthetic training data:**

```python
strategy = SIMBAUQSamplingStrategy(
    temperatures=[0.3, 0.5, 0.7, 1.0],
    n_per_temp=3,
    similarity_metric="rouge",
    confidence_method="classifier",
    training_samples=[
        ["correct answer 1", "correct answer 2", ..., "wrong answer"],  # group 1
        ["correct answer 1", "correct answer 2", ..., "wrong answer"],  # group 2
    ],
    training_labels=[
        [1, 1, ..., 0],  # labels for group 1
        [1, 1, ..., 0],  # labels for group 2
    ],
)
```

**Option B — HF-generated training data (requires `pip install datasets`):**

`generate_training_data()` in `simbauq_example.py` streams items from a HF
dataset, generates responses at each temperature, and labels them by similarity
to the ground-truth reference. Supported datasets: `"triviaqa"` (short QA)
and `"samsum"` (dialogue summarization).

```python
from simbauq_example import generate_training_data, make_session

m = make_session()
temperatures = [0.3, 0.5, 0.7, 1.0]
n_per_temp = 3

training_samples, training_labels = generate_training_data(
    m,
    temperatures,
    n_per_temp,
    dataset="triviaqa",       # or "samsum"
    similarity_metric="rouge",
    threshold=0.5,            # similarity >= threshold → label 1
)

strategy = SIMBAUQSamplingStrategy(
    temperatures=temperatures,
    n_per_temp=n_per_temp,
    similarity_metric="rouge",
    confidence_method="classifier",
    training_samples=training_samples,
    training_labels=training_labels,
)
```

Groups where all responses receive the same label are discarded automatically.
If no valid groups are collected, lower the `threshold` (scores are too low)
or raise it (all responses score above threshold).

**Option C — pre-trained classifier:**

Train the classifier externally with `train_classifier()`, then pass the fitted
object via `classifier=`. The feature extraction reuses
`SIMBAUQSamplingStrategy._compute_similarity_matrix` and `_extract_features`
internally, so the feature space is identical to what the strategy uses at
inference time.

```python
from simbauq_example import generate_training_data, train_classifier, make_session

m = make_session()
temperatures = [0.3, 0.5, 0.7, 1.0]
n_per_temp = 3

training_samples, training_labels = generate_training_data(
    m, temperatures, n_per_temp, dataset="triviaqa", similarity_metric="rouge", threshold=0.5
)

clf = train_classifier(training_samples, training_labels, similarity_metric="rouge")

strategy = SIMBAUQSamplingStrategy(
    temperatures=temperatures,
    n_per_temp=n_per_temp,
    similarity_metric="rouge",
    confidence_method="classifier",
    classifier=clf,
)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperatures` | `list[float]` | `[0.3, 0.5, 0.7, 1.0]` | Temperature values to sample at |
| `n_per_temp` | `int` | `4` | Number of samples per temperature |
| `similarity_metric` | `"rouge"`, `"jaccard"`, `"sbert"`, `"difflib"`, `"levenshtein"` | `"rouge"` | Pairwise similarity metric |
| `confidence_method` | `"aggregation"`, `"classifier"` | `"aggregation"` | Confidence estimation method |
| `aggregation` | `"mean"`, `"geometric_mean"`, `"harmonic_mean"`, `"median"`, `"max"`, `"min"` | `"mean"` | Aggregation function (for `aggregation` method) |
| `classifier` | sklearn classifier | `None` | Pre-trained classifier with `predict_proba` |
| `training_samples` | `list[list[str]]` | `None` | Training data for classifier |
| `training_labels` | `list[list[int]]` | `None` | Binary correctness labels (0/1) |
| `clf_max_depth` | `int` | `4` | Max tree depth for random forest |
| `rouge_type` | `str` | `"rougeL"` | Rouge variant |
| `sbert_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-BERT model name |
| `requirements` | `list[Requirement]` | `None` | Requirements to validate the selected sample |

## Similarity Metrics

- **rouge** (default): RougeL F-measure. Good general-purpose text similarity.
  No extra dependencies beyond `rouge-score` (already in Mellea).
- **jaccard**: Word-level set overlap (intersection / union). Fast, no
  external dependencies, works well for short structured answers.
- **sbert**: Cosine similarity of Sentence-BERT embeddings. Best semantic
  similarity but requires `sentence-transformers`.
- **difflib**: `difflib.SequenceMatcher` ratio. Character-level similarity
  from the Python standard library; no extra dependencies.
- **levenshtein**: Normalized Levenshtein edit distance (`1 - dist / max_len`).
  Exact character-level metric; no extra dependencies.

## Inspecting Results

The selected sample's `ModelOutputThunk` stores confidence metadata:

```python
result = m.instruct(..., strategy=strategy, return_sampling_results=True)

# Best sample
best_mot = result.result
meta = best_mot._meta["simba_uq"]

meta["confidence"]        # float: confidence of the selected sample
meta["all_confidences"]   # list[float]: confidence for every sample
meta["similarity_matrix"] # list[list[float]]: N x N pairwise similarity matrix
meta["temperatures_used"] # list[float]: temperature used for each sample
meta["confidence_method"] # "aggregation" or "classifier"
meta["similarity_metric"] # "rouge", "jaccard", "sbert", "difflib", or "levenshtein"
meta["aggregation"]       # aggregation function name

# All generated samples
for i, mot in enumerate(result.sample_generations):
    print(f"Sample {i}: {mot.value}")
```

## Related Files

- `mellea/stdlib/sampling/simbauq.py` -- Strategy implementation
- `test/stdlib/sampling/test_simbauq.py` -- Unit and integration tests
- `docs/examples/simbauq/simbauq_data.py` -- CLI tool for large-scale offline training data generation
