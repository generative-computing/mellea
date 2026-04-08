"""SIMBA-UQ Sampling Strategy.

Implements confidence-aware sample selection using the SIMBA-UQ framework
(Bhattacharjya et al., 2025). Generates multiple samples across a range of
temperatures and selects the most confident one.

Two confidence estimation methods are supported:

* **aggregation** (data-free) — computes pairwise similarity between all
  samples, then aggregates per-sample similarities into a confidence score.
* **classifier** — extracts pairwise similarity features and feeds them into
  a trained probabilistic classifier (e.g. random forest) that predicts
  P(correct) for each sample.

Reference:
    Bhattacharjya et al. (2025), "SIMBA UQ: Similarity-Based Aggregation for
    Uncertainty Quantification in Large Language Models", https://arxiv.org/abs/2510.13836
"""

import asyncio
from copy import deepcopy
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from rouge_score.rouge_scorer import RougeScorer  # codespell:ignore

from ...core import (
    Backend,
    BaseModelSubclass,
    Component,
    Context,
    Requirement,
    S,
    SamplingResult,
    SamplingStrategy,
    ValidationResult,
)
from .. import functional as mfuncs


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    """Protocol for sklearn-compatible probabilistic classifiers."""

    def predict_proba(self, X: list[np.ndarray]) -> np.ndarray:
        """Return class probability estimates for the given samples."""
        ...


class SIMBAUQSamplingStrategy(SamplingStrategy):
    """Sampling strategy that selects the most confident sample using SIMBA-UQ.

    Generates ``len(temperatures) * n_per_temp`` samples across a range of
    temperature values, computes pairwise similarity between all samples, and
    uses either similarity aggregation or a trained classifier to estimate
    per-sample confidence. The sample with the highest confidence is returned.

    Confidence metadata is stored on the selected ``ModelOutputThunk`` in
    ``mot.meta['simba_uq']``.

    Args:
        temperatures (list[float]): Temperature values to sample at.
        n_per_temp (int): Number of samples to generate per temperature value.
        similarity_metric (Literal['rouge', 'jaccard', 'sbert']): Pairwise
            similarity metric. ``'rouge'`` uses RougeL F-measure; ``'jaccard'``
            uses word-level Jaccard index; ``'sbert'`` uses cosine similarity of
            Sentence-BERT embeddings (requires ``sentence-transformers``).
        confidence_method (Literal['aggregation', 'classifier']): How to
            compute confidence from the similarity matrix. ``'aggregation'``
            uses a data-free aggregation function; ``'classifier'`` uses a
            trained probabilistic classifier.
        aggregation (Literal['mean', 'geometric_mean', 'harmonic_mean',
            'median', 'max', 'min']): Aggregation function used when
            ``confidence_method='aggregation'``.
        classifier (ProbabilisticClassifier | None): Pre-trained
            sklearn-compatible probabilistic classifier (any estimator with a
            ``predict_proba`` method). Used when
            ``confidence_method='classifier'``. If not provided, a random
            forest is trained from ``training_samples`` and
            ``training_labels``.
        training_samples (list[list[str]] | None): Training data for the
            classifier — a list of query groups, each containing sample
            strings. Each group must have the same number of samples as
            ``len(temperatures) * n_per_temp``.
        training_labels (list[list[int]] | None): Binary correctness labels
            (0/1) matching ``training_samples``.
        clf_max_depth (int): Maximum tree depth for the random forest when
            training from data.
        rouge_type (str): Rouge variant when ``similarity_metric='rouge'``.
        sbert_model (str): Sentence-BERT model name when
            ``similarity_metric='sbert'``.
        requirements (list[Requirement] | None): Optional global requirements
            to validate the selected sample against.
    """

    _CLF_EPS = 1e-6

    def __init__(
        self,
        *,
        temperatures: list[float] | None = None,
        n_per_temp: int = 4,
        similarity_metric: Literal["rouge", "jaccard", "sbert"] = "rouge",
        confidence_method: Literal["aggregation", "classifier"] = "aggregation",
        aggregation: Literal[
            "mean", "geometric_mean", "harmonic_mean", "median", "max", "min"
        ] = "mean",
        classifier: ProbabilisticClassifier | None = None,
        training_samples: list[list[str]] | None = None,
        training_labels: list[list[int]] | None = None,
        clf_max_depth: int = 4,
        rouge_type: str = "rougeL",
        sbert_model: str = "all-MiniLM-L6-v2",
        requirements: list[Requirement] | None = None,
    ) -> None:
        """Initialize SIMBAUQSamplingStrategy with temperature schedule and confidence parameters."""
        if temperatures is None:
            temperatures = [0.3, 0.5, 0.7, 1.0]

        if len(temperatures) == 0:
            raise ValueError("Temperatures must be a non-empty list")
        if n_per_temp <= 0:
            raise ValueError("n_per_temp must be > 0")

        self.temperatures = temperatures
        self.n_per_temp = n_per_temp
        self.similarity_metric = similarity_metric
        self.confidence_method = confidence_method
        self.aggregation = aggregation
        self.clf_max_depth = clf_max_depth
        self.rouge_type = rouge_type
        self.sbert_model = sbert_model
        self.requirements = requirements

        # --- Similarity metric initialization ---
        if similarity_metric == "rouge":
            self._rouge_scorer = RougeScorer([rouge_type], use_stemmer=True)
        elif similarity_metric == "sbert":
            try:
                import sentence_transformers  # type: ignore[import-not-found]
            except ImportError:
                msg = (
                    "sentence-transformers is required for sbert similarity. "
                    "Please install with `pip install sentence-transformers`."
                )
                raise ImportError(msg)
            self._sbert_model_obj = sentence_transformers.SentenceTransformer(
                sbert_model
            )

        # --- Classifier initialization ---
        self._classifier: ProbabilisticClassifier | None = None
        if confidence_method == "classifier":
            if classifier is not None:
                self._classifier = classifier
            elif training_samples is not None and training_labels is not None:
                n_samples = len(temperatures) * n_per_temp
                for i, group in enumerate(training_samples):
                    msg = (
                        f"Training group {i} has {len(group)} samples, "
                        f"expected {n_samples} "
                        f"(len(temperatures) * n_per_temp)"
                    )
                    if len(group) != n_samples:
                        raise ValueError(msg)

                    msg = (
                        f"Training labels group {i} has "
                        f"{len(training_labels[i])} labels, "
                        f"expected {n_samples}"
                    )
                    if len(training_labels[i]) != n_samples:
                        raise ValueError(msg)

                self._classifier = self._train_classifier(
                    training_samples, training_labels
                )
            else:
                msg = (
                    "confidence_method='classifier' requires either a "
                    "'classifier' or both 'training_samples' and "
                    "'training_labels'"
                )
                raise ValueError(msg)

    async def sample(
        self,
        action: Component[S],
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]:
        """Sample across temperatures and select the most confident result.

        Args:
            action: The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            backend: The backend used for generating samples.
            requirements: List of requirements to test against (merged with
                global requirements).
            validation_ctx: Optional context to use for validation.
            format: Output format for structured outputs.
            model_options: Model options to pass to the backend during
                generation.
            tool_calls: True if tool calls should be used during this sampling
                strategy.

        Returns:
            SamplingResult with the most confident sample selected.
        """
        if model_options is None:
            model_options = {}

        # Merge requirements: global requirements override local.
        reqs = self._merge_requirements(requirements)

        # --- Phase 1: Generate samples across temperatures ---
        generation_tasks: list[asyncio.Task] = []
        temp_assignments: list[float] = []

        for temp in self.temperatures:
            for _ in range(self.n_per_temp):
                opts = {**model_options, "temperature": temp}
                task = asyncio.create_task(
                    backend.generate_from_context(
                        deepcopy(action),
                        ctx=context,
                        format=format,
                        model_options=opts,
                        tool_calls=tool_calls,
                    )
                )
                generation_tasks.append(task)
                temp_assignments.append(temp)

        generation_results = await asyncio.gather(*generation_tasks)

        # Resolve all thunks and parse.
        all_mots = []
        all_contexts = []
        all_actions = []
        for result_mot, result_ctx in generation_results:
            await result_mot.avalue()
            result_mot.parsed_repr = action.parse(result_mot)
            all_mots.append(result_mot)
            all_contexts.append(result_ctx)
            all_actions.append(action)

        # --- Phase 2: Compute SIMBA-UQ confidence scores ---
        sample_strings = [str(mot) for mot in all_mots]
        n = len(sample_strings)
        if n == 1:
            sim_matrix = np.ones((1, 1))
            confidences = np.array([0.5])
        else:
            sim_matrix = self._compute_similarity_matrix(sample_strings)
            if self.confidence_method == "classifier":
                confidences = self._compute_confidences_classifier(sim_matrix, n)
            else:
                confidences = self._compute_confidences(sim_matrix)

        # Select the sample with the highest confidence.
        best_index = int(np.argmax(confidences))
        best_confidence = float(confidences[best_index])

        # Store confidence metadata in the selected MOT's meta dict.
        best_mot = all_mots[best_index]
        if best_mot._meta is None:
            best_mot._meta = {}
        best_mot._meta["simba_uq"] = {
            "confidence": best_confidence,
            "all_confidences": confidences.tolist(),
            "similarity_matrix": sim_matrix.tolist(),
            "temperatures_used": temp_assignments,
            "confidence_method": self.confidence_method,
            "similarity_metric": self.similarity_metric,
            "aggregation": self.aggregation,
        }

        # Mark as final result.
        if best_mot._generate_log is not None:
            best_mot._generate_log.is_final_result = True

        # --- Phase 3: Validate selected sample (if requirements exist) ---
        success = True
        all_validations: list[list[tuple[Requirement, ValidationResult]]] = [
            [] for _ in all_mots
        ]

        if reqs:
            val_results = await mfuncs.avalidate(
                reqs=reqs,
                context=all_contexts[best_index],
                backend=backend,
                output=best_mot,
                format=None,
                model_options=model_options,
            )
            scored = list(zip(reqs, val_results))
            all_validations[best_index] = scored
            success = all(vr.as_bool() for vr in val_results)

        return SamplingResult(
            result_index=best_index,
            success=success,
            sample_generations=all_mots,
            sample_validations=all_validations,
            sample_actions=all_actions,
            sample_contexts=all_contexts,
        )

    def _merge_requirements(self, local: list[Requirement] | None) -> list[Requirement]:
        """Merge global and local requirements, deduplicating by identity."""
        combined: list[Requirement] = []
        seen: set[int] = set()
        for req_list in (self.requirements, local):
            if req_list is None:
                continue
            for req in req_list:
                if id(req) not in seen:
                    combined.append(req)
                    seen.add(id(req))
        return combined

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute pairwise similarity between two text strings.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        if self.similarity_metric == "rouge":
            scores = self._rouge_scorer.score(text1, text2)
            return scores[self.rouge_type].fmeasure

        if self.similarity_metric == "sbert":
            try:
                from sklearn.metrics.pairwise import (
                    cosine_similarity,  # type: ignore[import-not-found]
                )
            except ImportError:
                msg = (
                    "sklearn.metrics.pairwise.cosine_similarity is required for sbert similarity. "
                    "Please install with `pip install sklearn`."
                )
                raise ImportError(msg)

            embs = self._sbert_model_obj.encode([text1, text2])
            return float(cosine_similarity([embs[0]], [embs[1]])[0, 0])

        # Jaccard: word-level set overlap.
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        union = len(words1 | words2)
        return len(words1 & words2) / union if union > 0 else 0.0

    def _compute_similarity_matrix(self, samples: list[str]) -> np.ndarray:
        """Build a symmetric pairwise similarity matrix.

        For ``sbert``, batch-encodes all samples once and computes cosine
        similarity in a single matrix operation. For ``rouge`` and ``jaccard``,
        computes pairwise similarities individually (upper triangle, mirrored).

        Args:
            samples: List of sample strings.

        Returns:
            Symmetric (N, N) matrix with self-similarity = 1.0.
        """
        if self.similarity_metric == "sbert":
            try:
                from sklearn.metrics.pairwise import (
                    cosine_similarity,  # type: ignore[import-not-found]
                )
            except ImportError:
                msg = (
                    "sklearn.metrics.pairwise.cosine_similarity is required for sbert similarity. "
                    "Please install with `pip install sklearn`."
                )
                raise ImportError(msg)

            embeddings = self._sbert_model_obj.encode(samples)
            matrix = cosine_similarity(embeddings)
            np.fill_diagonal(matrix, 1.0)
            return matrix

        n = len(samples)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_similarity(samples[i], samples[j])
                matrix[i, j] = sim
                matrix[j, i] = sim
        return matrix

    def _aggregate(self, similarities: np.ndarray) -> float:
        """Aggregate a vector of similarity scores into a single confidence value.

        Args:
            similarities: 1-D array of similarity scores.

        Returns:
            Aggregated confidence score.
        """
        epsilon = 1e-10
        if len(similarities) == 0:
            return 0.0

        if self.aggregation == "mean":
            return float(np.mean(similarities))

        if self.aggregation == "geometric_mean":
            log_sims = np.log(similarities + epsilon)
            return float(np.exp(np.mean(log_sims)))

        if self.aggregation == "harmonic_mean":
            try:
                from scipy import stats as scipy_stats
            except ImportError:
                msg = (
                    "scipy is required for harmonic mean aggregation. "
                    "Please install with `pip install scipy`."
                )
                raise ImportError(msg)

            return float(scipy_stats.hmean(similarities + epsilon))

        if self.aggregation == "median":
            return float(np.median(similarities))

        if self.aggregation == "max":
            return float(np.max(similarities))

        if self.aggregation == "min":
            return float(np.min(similarities))

        msg = f"Unknown aggregation method: {self.aggregation}"
        raise ValueError(msg)

    def _extract_features(
        self, sim_matrix: np.ndarray, sample_index: int
    ) -> np.ndarray:
        """Extract pairwise similarity features for a single sample.

        Returns the similarity row with self-similarity removed and values
        clipped to ``(eps, 1 - eps)`` for numerical stability.

        Args:
            sim_matrix: Symmetric (N, N) similarity matrix.
            sample_index: Index of the sample to extract features for.

        Returns:
            1-D feature array of length ``N - 1``.
        """
        row = np.delete(sim_matrix[sample_index, :], sample_index)
        return np.clip(row, self._CLF_EPS, 1.0 - self._CLF_EPS)

    def _train_classifier(
        self, training_samples: list[list[str]], training_labels: list[list[int]]
    ) -> ProbabilisticClassifier:
        """Train a random forest classifier on similarity features.

        Args:
            training_samples: List of query groups, each a list of sample
                strings with the same length as the inference-time sample
                count.
            training_labels: Binary correctness labels (0/1) matching
                ``training_samples``.

        Returns:
            Trained ``RandomForestClassifier``.
        """
        try:
            from sklearn.ensemble import (
                RandomForestClassifier,  # type: ignore[import-not-found]
            )
        except ImportError:
            msg = (
                "sklearn is required for training a Random Forest classifier. "
                "Please install with `pip install sklearn`."
            )
            raise ImportError(msg)

        x_train: list[np.ndarray] = []
        y_train: list[int] = []
        for samples, labels in zip(training_samples, training_labels):
            sim_matrix = self._compute_similarity_matrix(samples)
            for i, label in enumerate(labels):
                x_train.append(self._extract_features(sim_matrix, i))
                y_train.append(label)
        clf = RandomForestClassifier(max_depth=self.clf_max_depth, random_state=0)
        clf.fit(x_train, y_train)
        return clf

    def _compute_confidences_classifier(
        self, sim_matrix: np.ndarray, n: int
    ) -> np.ndarray:
        """Compute per-sample confidence using the trained classifier.

        Args:
            sim_matrix: Pre-computed (N, N) similarity matrix.
            n: Number of samples.

        Returns:
            Array of P(correct) confidence scores with shape ``(n,)``.
        """
        x_test = [self._extract_features(sim_matrix, i) for i in range(n)]
        assert self._classifier is not None
        probs = self._classifier.predict_proba(x_test)
        return probs[:, 1]

    def _compute_confidences(self, sim_matrix: np.ndarray) -> np.ndarray:
        """Compute per-sample confidence using similarity-based aggregation.

        For each sample, aggregates its similarities to every other sample
        into a single confidence score.

        Args:
            sim_matrix: Symmetric (N, N) pairwise similarity matrix.

        Returns:
            Array of confidence scores with shape ``(N,)``.
        """
        n = sim_matrix.shape[0]
        if n == 1:
            return np.array([0.5])

        confidences = np.zeros(n)
        for i in range(n):
            others = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i + 1 :]])
            confidences[i] = self._aggregate(others)
        return confidences
