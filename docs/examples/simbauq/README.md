# SIMBA-UQ Sampling Strategy

Confidence-aware sample selection using the SIMBA-UQ framework
(Bhattacharjya et al., 2025). Generates multiple samples across a range of
temperatures and selects the one with the highest estimated confidence.

**Paper:** [SIMBA UQ: Similarity-Based Aggregation for Uncertainty Quantification in Large Language Models](https://arxiv.org/abs/2510.13836)

## Files

### simbauq_example.py

Complete example demonstrating both confidence estimation methods with
ollama and granite-4.0-micro.

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

**With training data:**

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

Each training group must have exactly `len(temperatures) * n_per_temp` samples
so the feature vectors match at inference time.

**With pre-trained classifier:**

```python
strategy = SIMBAUQSamplingStrategy(
    temperatures=[0.3, 0.5, 0.7, 1.0],
    n_per_temp=3,
    confidence_method="classifier",
    classifier=my_pretrained_sklearn_clf,
)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperatures` | `list[float]` | `[0.3, 0.5, 0.7, 1.0]` | Temperature values to sample at |
| `n_per_temp` | `int` | `4` | Number of samples per temperature |
| `similarity_metric` | `"rouge"`, `"jaccard"`, `"sbert"` | `"rouge"` | Pairwise similarity metric |
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
  No extra dependencies beyond `rouge-score` (already in mellea).
- **jaccard**: Word-level set overlap (intersection / union). Fast, no
  external dependencies, works well for short structured answers.
- **sbert**: Cosine similarity of Sentence-BERT embeddings. Best semantic
  similarity but requires `sentence-transformers` (`pip install
  mellea[granite_retriever]`).

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
meta["similarity_metric"] # "rouge", "jaccard", or "sbert"
meta["aggregation"]       # aggregation function name

# All generated samples
for i, mot in enumerate(result.sample_generations):
    print(f"Sample {i}: {mot.value}")
```

## Related Files

- `mellea/stdlib/sampling/simbauq.py` -- Strategy implementation
- `test/stdlib/sampling/test_simbauq.py` -- Unit and integration tests
