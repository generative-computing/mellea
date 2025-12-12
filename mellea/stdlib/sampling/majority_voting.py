"""Sampling Strategies for Minimum Bayes Risk Decoding (MBRD)."""

import abc
import asyncio
from collections.abc import Sequence

import numpy as np
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from rouge_score.rouge_scorer import RougeScorer  # codespell:ignore

from mellea.backends import Backend, BaseModelSubclass
from mellea.stdlib.requirement import Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy, SamplingResult
from mellea.stdlib.sampling.base import Component, Context


class BaseMBRDSampling(RejectionSamplingStrategy):
    """Abstract Minimum Bayes Risk Decoding (MBRD) Sampling Strategy."""

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
        """Initialize a new abstract Minimum Bayes Risk Decoding (MBRD) Sampling Strategy with default parameters.

        Inherits from RejectionSamplingStrategy. Will generate up to loop_budget x number_of_samples requests. If no
        requirements are provided here or in sample(...), will only generate number_of_samples requests.

        Classes that inherit from this must implement the `compare_strings` function.

        Args:
            number_of_samples: Number of samples to generate and use for majority voting
            weighted: Not Implemented. If True, weights the score before getting the final majority vote
            loop_budget: Inner rejection sampling number of times to iterate through the process. Must be greater than 0.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

        Raises:
            AssertionError: If loop_budget is not greater than 0.
        """
        super().__init__(loop_budget=loop_budget, requirements=requirements)
        self.number_of_samples = number_of_samples
        self.weighted = weighted
        self.symmetric = True

    @abc.abstractmethod
    def compare_strings(self, ref: str, pred: str) -> float:
        """This method is the abstract method for MBRD similarity metric."""

    def maybe_apply_weighted(self, scr: np.ndarray):
        """Applies weights if self.weighted is True. Not Implemented."""
        # TODO: not implemented yet
        if self.weighted:
            weights = np.asarray([1.0 for _ in range(len(scr))])
            scr = scr * weights

        return scr

    async def sample(
        self,
        action: Component,
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        show_progress: bool = True,
        labels: Sequence[str] | None = None,
    ) -> SamplingResult:
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
            labels: if provided, restrict generation to context nodes with matching types.

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
                    labels=labels,
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
    """MajorityVoting Sampling Strategy for Math Expressions."""

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
        """Initialize a new instance of MajorityVoting Sampling Strategy for Math with default parameters.

        Will generate up to loop_budget x number_of_samples requests. If no
        requirements are provided here or in sample(...), will only generate number_of_samples requests.

        Args:
            number_of_samples: Number of samples to generate and use for majority voting
            float_rounding: Number of decimal places to round floats to. Defaults to 6.
            strict: Whether to enforce strict comparison mode. Defaults to True.
                - In strict mode: Variables matter and sets are not comparable with tuples
                - In non-strict mode: Variables are matched by position and sets can be compared with tuples
            allow_set_relation_comp: Whether to allow set - relation (e.g 1 < x < 2 and (1, 2)) comparison. Defaults to False.
                - If True, set - relation comparison will be allowed in all cases.
                - If False, set - relation comparison will be allowed only if the prediction is a set.
            weighted: Not Implemented. If True, weights the score before getting the final majority vote
            loop_budget: Inner rejection sampling number of times to iterate through the process. Must be greater than 0.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

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
        """Helper function to compare strings using the math extraction metrics."""
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
    """Sampling Strategy that uses RougeL to compute symbol-level distances for majority voting."""

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
        """Initialize a new instance of MBRDRougeL Sampling Strategy with default parameters.

        Will generate up to loop_budget x number_of_samples requests. If no
        requirements are provided here or in sample(...), will only generate number_of_samples requests.

        Args:
            number_of_samples: Number of samples to generate and use for majority voting
            weighted: Not Implemented. If True, weights the score before getting the final majority vote
            loop_budget: Inner rejection sampling number of times to iterate through the process. Must be greater than 0.
            requirements: List of requirements to test against. If None, test all requirements attached to the given instruction.

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
        """Helper function to compare strings using the math extraction metrics."""
        scr: float = self.scorer.score(ref, pred)[self.match_types[-1]].fmeasure
        return scr
