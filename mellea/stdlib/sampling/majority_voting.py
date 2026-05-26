"""Sampling Strategies for Minimum Bayes Risk Decoding (MBRD)."""

import abc
import asyncio

import numpy as np
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from rouge_score.rouge_scorer import RougeScorer  # codespell:ignore

from ...core import (
    Backend,
    BaseModelSubclass,
    Component,
    Context,
    Requirement,
    S,
    SamplingResult,
)
from .base import RejectionSamplingStrategy


class BaseMBRDSampling(RejectionSamplingStrategy):
    """Abstract Minimum Bayes Risk Decoding (MBRD) Sampling Strategy.

    Args:
        number_of_samples (int): Number of samples to generate and use for
            majority voting. Defaults to `8`.
        weighted (bool): Not yet implemented. If `True`, weights scores
            before majority vote.
        loop_budget (int): Inner rejection-sampling loop count. Must be > 0.
        requirements (list[Requirement] | None): Requirements to validate
            against. If `None`, uses per-call requirements.

    Attributes:
        symmetric (bool): Whether the similarity metric is symmetric, allowing
            the upper-triangle score matrix to be mirrored; always `True` for
            this base class.
    """

    number_of_samples: int
    weighted: bool
    symmetric: bool

    def __init__(
        self,
        *,
        number_of_samples: int = 8,
        weighted: bool = False,
        loop_budget: int = 1,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize BaseMBRDSampling with the number of samples, weighting flag, and inner loop budget.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(loop_budget=loop_budget, requirements=requirements)
        self.number_of_samples = number_of_samples
        self.weighted = weighted
        self.symmetric = True

    @abc.abstractmethod
    def compare_strings(self, ref: str, pred: str) -> float:
        """Compute a similarity score between a reference and a predicted string.

        Subclasses must implement this to define the MBRD similarity metric.

        Args:
            ref (str): The reference string to compare against.
            pred (str): The predicted string to evaluate.

        Returns:
            float: A similarity score, typically in `[0.0, 1.0]` where `1.0`
            indicates a perfect match.
        """

    def maybe_apply_weighted(self, scr: np.ndarray) -> np.ndarray:
        """Apply per-sample weights to the score vector if `self.weighted` is `True`.

        Currently not implemented; the input array is returned unchanged when
        `self.weighted` is `True`.

        Args:
            scr (np.ndarray): 1-D array of aggregated similarity scores, one
                entry per candidate sample.

        Returns:
            np.ndarray: The (possibly weighted) score array.
        """
        # TODO: not implemented yet
        if self.weighted:
            weights = np.asarray([1.0 for _ in range(len(scr))])
            scr = scr * weights

        return scr

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
        show_progress: bool = True,
    ) -> SamplingResult[S]:
        """Samples using majority voting.

        Args:
            action : The action object to be sampled.
            context: The context to be passed to the sampling strategy.
            backend: The backend used for generating samples.
            requirements: List of requirements to test against (merged with global requirements).
            validation_ctx: Optional context to use for validation. If None, validation_ctx = ctx.
            format: output format for structured outputs; ignored for this sampling strategy.
            model_options: model options to pass to the backend during generation / validation.
            tool_calls: True if tool calls should be used during this sampling strategy.
            show_progress: if true, a tqdm progress bar is used. Otherwise, messages will still be sent to flog.

        Returns:
            SamplingResult: A result object indicating the success or failure of the sampling process.
        """
        # execute sampling concurrently
        tasks: list[asyncio.Task[SamplingResult]] = []
        for i in range(self.number_of_samples):
            task = asyncio.create_task(
                super().sample(
                    action,
                    context,
                    backend,
                    requirements,
                    validation_ctx=validation_ctx,
                    model_options=model_options,
                    tool_calls=tool_calls,
                    show_progress=show_progress,
                )
            )
            tasks.append(task)

        sampling_results = await asyncio.gather(*tasks)

        # collect results
        results: list[tuple[str, SamplingResult]] = []
        for result in sampling_results:
            output = str(result.result)
            results.append((output, result))
        assert len(results) > 0

        # Create an array of len(results) x len(results) initialized to 0.0.
        scr = np.asarray(
            [[0.0 for _ in range(len(results))] for _ in range(len(results))]
        )
        for i in range(len(results)):
            for j in range(len(results)):
                if j == i:
                    scr[i][j] = 1.0  # self voting is 1.
                    continue

                # upper triangle
                # For sample i compute votes against all j references
                if j > i:
                    scr[i][j] = float(
                        self.compare_strings(results[j][0], results[i][0])
                    )
                    continue

                else:
                    if self.symmetric:
                        scr[i][j] = scr[j][i]
                    else:
                        scr[i][j] = float(
                            self.compare_strings(results[j][0], results[i][0])
                        )
                    continue

        # count votes
        summed_scr: np.ndarray = scr.sum(axis=0)

        # Apply weights
        weighed_scr = self.maybe_apply_weighted(summed_scr)

        maxR = int(weighed_scr.argmax())

        return results[maxR][1]  # return one of the MV answers


class MajorityVotingStrategyForMath(BaseMBRDSampling):
    """MajorityVoting Sampling Strategy for Math Expressions.

    Args:
        number_of_samples (int): Number of samples to generate. Defaults to `8`.
        float_rounding (int): Decimal places for float comparison. Defaults to `6`.
        strict (bool): Enforce strict comparison mode. Defaults to `True`.
        allow_set_relation_comp (bool): Allow set-relation comparisons. Defaults
            to `False`.
        weighted (bool): Not yet implemented. Defaults to `False`.
        loop_budget (int): Rejection-sampling loop count. Defaults to `1`.
        requirements (list[Requirement] | None): Requirements to validate against.

    Attributes:
        match_types (list[str]): Extraction target types used for parsing math
            expressions; always `["latex", "axpr"]`, computed at init.
        symmetric (bool): Inherited from `BaseMBRDSampling`; always `True`
            for this strategy (set explicitly at init).
    """

    number_of_samples: int
    match_types: list[str]
    float_rounding: int
    strict: bool
    allow_set_relation_comp: bool

    def __init__(
        self,
        *,
        number_of_samples: int = 8,
        float_rounding: int = 6,
        strict: bool = True,
        allow_set_relation_comp: bool = False,
        weighted: bool = False,
        loop_budget: int = 1,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize MajorityVotingStrategyForMath with math-comparison settings and sampling parameters.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(
            number_of_samples=number_of_samples,
            weighted=weighted,
            loop_budget=loop_budget,
            requirements=requirements,
        )
        self.number_of_samples = number_of_samples
        # match_type: type of match latex, expr (match only so far)
        #     -  For math use "latex" or "expr" or both
        #     -  For general text similarity use "rougel"
        MATCH_TYPES = ["latex", "axpr"]
        self.match_types = MATCH_TYPES
        self.float_rounding = float_rounding
        self.strict = strict
        self.allow_set_relation_comp = allow_set_relation_comp

        # Note: symmetry is not implied for certain expressions, see: https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/README.md?plain=1#L183
        self.symmetric = True

    # https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/tests/test_all.py#L36
    def compare_strings(self, ref: str, pred: str) -> float:
        """Compare two strings using math-aware extraction and verification.

        Parses both strings into mathematical expressions using the configured
        `match_types` (latex and/or expr), then verifies equivalence via
        `math_verify.verify`.

        Args:
            ref (str): The reference (gold) string containing a math expression.
            pred (str): The predicted string to compare against the reference.

        Returns:
            float: `1.0` if the expressions are considered equivalent,
            `0.0` otherwise.
        """
        # Convert string match_types to ExtractionTarget objects
        extraction_targets = []
        for match_type in self.match_types:
            if match_type == "latex":
                extraction_targets.append(LatexExtractionConfig(boxed_match_priority=0))
            elif match_type == "expr":
                extraction_targets.append(ExprExtractionConfig())

        # NOTE: Math-Verify parse and verify functions don't support threaded environment due to usage of signal.alarm() in timeout mechanism. If you need to run in multithreaded environment it's recommended to set the parsing_timeout=None
        gold_parsed = parse(ref, extraction_targets, parsing_timeout=None)  # type: ignore
        pred_parsed = parse(pred, extraction_targets, parsing_timeout=None)  # type: ignore
        return float(
            verify(
                gold_parsed,
                pred_parsed,
                float_rounding=self.float_rounding,
                strict=self.strict,
                allow_set_relation_comp=self.allow_set_relation_comp,
                timeout_seconds=None,
            )
        )


class MBRDRougeLStrategy(BaseMBRDSampling):
    """Sampling Strategy that uses RougeL to compute symbol-level distances for majority voting.

    Args:
        number_of_samples (int): Number of samples to generate. Defaults to `8`.
        weighted (bool): Not yet implemented. Defaults to `False`.
        loop_budget (int): Rejection-sampling loop count. Defaults to `1`.
        requirements (list[Requirement] | None): Requirements to validate against.

    Attributes:
        match_types (list[str]): Rouge metric names used for scoring (`["rougeL"]`).
        scorer (RougeScorer): Pre-configured `RougeScorer` instance used for
            pairwise string comparison.
        symmetric (bool): Inherited from `BaseMBRDSampling`; always `True` for
            RougeL (the score is symmetric by construction).
    """

    match_types: list[str]
    scorer: RougeScorer

    def __init__(
        self,
        *,
        number_of_samples: int = 8,
        weighted: bool = False,
        loop_budget: int = 1,
        requirements: list[Requirement] | None = None,
    ):
        """Initialize MBRDRougeLStrategy with RougeL scoring and sampling parameters.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(
            number_of_samples=number_of_samples,
            weighted=weighted,
            loop_budget=loop_budget,
            requirements=requirements,
        )
        self.match_types = ["rougeL"]
        self.symmetric = True
        self.scorer = RougeScorer(self.match_types, use_stemmer=True)

    # https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/tests/test_all.py#L36
    def compare_strings(self, ref: str, pred: str) -> float:
        """Compare two strings using the RougeL F-measure.

        Args:
            ref (str): The reference string to score against.
            pred (str): The predicted string to evaluate.

        Returns:
            float: RougeL F-measure score in the range `[0.0, 1.0]`.
        """
        scr: float = self.scorer.score(ref, pred)[self.match_types[-1]].fmeasure
        return scr
