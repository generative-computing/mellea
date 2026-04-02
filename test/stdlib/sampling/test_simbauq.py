"""Tests for SIMBAUQSamplingStrategy."""

import numpy as np
import pytest

from mellea.stdlib.sampling.simbauq import SIMBAUQSamplingStrategy

# --- Unit tests (no LLM required) ---


class TestComputeSimilarity:
    def test_rouge_identical(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="rouge")
        assert strategy._compute_similarity(
            "hello world", "hello world"
        ) == pytest.approx(1.0)

    def test_rouge_different(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="rouge")
        score = strategy._compute_similarity("the cat sat on the mat", "dogs run fast")
        assert 0.0 <= score < 0.5

    def test_jaccard_identical(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="jaccard")
        assert strategy._compute_similarity(
            "hello world", "hello world"
        ) == pytest.approx(1.0)

    def test_jaccard_partial_overlap(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="jaccard")
        score = strategy._compute_similarity("hello world foo", "hello world bar")
        # intersection = {"hello", "world"}, union = {"hello", "world", "foo", "bar"}
        assert score == pytest.approx(2.0 / 4.0)

    def test_jaccard_no_overlap(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="jaccard")
        score = strategy._compute_similarity("alpha beta", "gamma delta")
        assert score == pytest.approx(0.0)

    def test_jaccard_empty_strings(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="jaccard")
        assert strategy._compute_similarity("", "") == pytest.approx(1.0)


class TestAggregate:
    def setup_method(self):
        self.sims = np.array([0.8, 0.6, 0.4])

    def test_mean(self):
        strategy = SIMBAUQSamplingStrategy(aggregation="mean")
        assert strategy._aggregate(self.sims) == pytest.approx(0.6)

    def test_median(self):
        strategy = SIMBAUQSamplingStrategy(aggregation="median")
        assert strategy._aggregate(self.sims) == pytest.approx(0.6)

    def test_max(self):
        strategy = SIMBAUQSamplingStrategy(aggregation="max")
        assert strategy._aggregate(self.sims) == pytest.approx(0.8)

    def test_min(self):
        strategy = SIMBAUQSamplingStrategy(aggregation="min")
        assert strategy._aggregate(self.sims) == pytest.approx(0.4)

    def test_geometric_mean(self):
        strategy = SIMBAUQSamplingStrategy(aggregation="geometric_mean")
        expected = (0.8 * 0.6 * 0.4) ** (1.0 / 3.0)
        assert strategy._aggregate(self.sims) == pytest.approx(expected, abs=1e-3)

    def test_harmonic_mean(self):
        strategy = SIMBAUQSamplingStrategy(aggregation="harmonic_mean")
        expected = 3.0 / (1.0 / 0.8 + 1.0 / 0.6 + 1.0 / 0.4)
        assert strategy._aggregate(self.sims) == pytest.approx(expected, abs=1e-3)

    def test_empty(self):
        strategy = SIMBAUQSamplingStrategy(aggregation="mean")
        assert strategy._aggregate(np.array([])) == 0.0


class TestComputeConfidences:
    def test_single_sample(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="jaccard")
        confs = strategy._compute_confidences(["hello world"])
        assert len(confs) == 1
        assert confs[0] == pytest.approx(0.5)

    def test_identical_samples_high_confidence(self):
        strategy = SIMBAUQSamplingStrategy(
            similarity_metric="jaccard", aggregation="mean"
        )
        samples = ["the cat sat on the mat"] * 5
        confs = strategy._compute_confidences(samples)
        assert len(confs) == 5
        for c in confs:
            assert c == pytest.approx(1.0)

    def test_outlier_has_lower_confidence(self):
        strategy = SIMBAUQSamplingStrategy(
            similarity_metric="jaccard", aggregation="mean"
        )
        samples = [
            "the capital of france is paris",
            "paris is the capital of france",
            "france capital is paris",
            "bananas are yellow fruit",  # outlier
        ]
        confs = strategy._compute_confidences(samples)
        assert len(confs) == 4
        # The outlier (index 3) should have the lowest confidence.
        assert confs[3] < confs[0]
        assert confs[3] < confs[1]
        assert confs[3] < confs[2]

    def test_similarity_matrix_symmetric(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="rouge")
        samples = ["hello world", "world hello", "foo bar"]
        matrix = strategy._compute_similarity_matrix(samples)
        assert matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), [1.0, 1.0, 1.0])


class TestSBERTSimilarity:
    @pytest.fixture(autouse=True)
    def _require_sbert(self):
        pytest.importorskip("sentence_transformers")

    def test_sbert_identical(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="sbert")
        score = strategy._compute_similarity("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_sbert_similar(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="sbert")
        score = strategy._compute_similarity(
            "The capital of France is Paris.", "Paris is the capital city of France."
        )
        assert score > 0.7

    def test_sbert_different(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="sbert")
        score = strategy._compute_similarity(
            "The capital of France is Paris.", "Bananas are a yellow tropical fruit."
        )
        assert score < 0.4

    def test_sbert_matrix_symmetric(self):
        strategy = SIMBAUQSamplingStrategy(similarity_metric="sbert")
        samples = ["hello world", "world hello", "foo bar baz"]
        matrix = strategy._compute_similarity_matrix(samples)
        assert matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(matrix, matrix.T)
        np.testing.assert_array_almost_equal(
            np.diag(matrix), [1.0, 1.0, 1.0], decimal=2
        )

    def test_sbert_outlier_confidence(self):
        strategy = SIMBAUQSamplingStrategy(
            similarity_metric="sbert", aggregation="mean"
        )
        samples = [
            "The capital of France is Paris.",
            "Paris is the capital of France.",
            "France has Paris as its capital.",
            "Bananas are a yellow tropical fruit.",  # outlier
        ]
        confs = strategy._compute_confidences(samples)
        assert len(confs) == 4
        assert confs[3] < confs[0]
        assert confs[3] < confs[1]
        assert confs[3] < confs[2]


class TestInit:
    def test_default_temperatures(self):
        strategy = SIMBAUQSamplingStrategy()
        assert strategy.temperatures == [0.3, 0.5, 0.7, 1.0]

    def test_custom_temperatures(self):
        strategy = SIMBAUQSamplingStrategy(temperatures=[0.1, 0.9])
        assert strategy.temperatures == [0.1, 0.9]

    def test_empty_temperatures_raises(self):
        with pytest.raises(AssertionError):
            SIMBAUQSamplingStrategy(temperatures=[])

    def test_zero_n_per_temp_raises(self):
        with pytest.raises(AssertionError):
            SIMBAUQSamplingStrategy(n_per_temp=0)


# --- Integration test (requires Ollama) ---


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
class TestSIMBAUQIntegration:
    def test_simbauq_sampling(self):
        from mellea import MelleaSession, start_session
        from mellea.backends import ModelOption
        from mellea.core import SamplingResult

        m: MelleaSession = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 30})

        result: SamplingResult = m.instruct(
            "What is the capital of France?",
            strategy=SIMBAUQSamplingStrategy(
                temperatures=[0.3, 0.7],
                n_per_temp=2,
                similarity_metric="rouge",
                aggregation="mean",
            ),
            return_sampling_results=True,
        )

        assert isinstance(result, SamplingResult)
        assert result.success is True
        assert len(result.sample_generations) == 4  # 2 temps * 2 per temp

        # Check that the selected MOT has confidence metadata.
        best_mot = result.result
        assert best_mot._meta is not None
        simba_meta = best_mot._meta["simba_uq"]
        assert "confidence" in simba_meta
        assert 0.0 <= simba_meta["confidence"] <= 1.0
        assert len(simba_meta["all_confidences"]) == 4
        assert simba_meta["similarity_metric"] == "rouge"
        assert simba_meta["aggregation"] == "mean"

        output = str(best_mot)
        print(f"Best output (confidence={simba_meta['confidence']:.3f}): {output}")
        assert output

        del m


if __name__ == "__main__":
    pytest.main(["-s", __file__])
