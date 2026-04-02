"""Standalone demo of SIMBAUQSamplingStrategy with RITSBackend.

Run from VS Code or the command line:

    uv run python test/stdlib/sampling/test_simbauq_dummy.py

Requires environment variables:
    RITS_API_KEY  — RITS API key
"""

import numpy as np
import pytest

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.core import SamplingResult
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling.simbauq import SIMBAUQSamplingStrategy

RITS_API_KEY = "eefb7c6fe528efb691cbf3be97fc387f"

pytestmark = [
    pytest.mark.openai,
    pytest.mark.llm,
    pytest.mark.qualitative,
    pytest.mark.skipif(not RITS_API_KEY, reason="RITS_API_KEY must be set"),
]


def _make_session() -> MelleaSession:
    from mellea_ibm.rits import RITS, RITSBackend  # type: ignore[import-not-found]

    backend = RITSBackend(
        model_name=RITS.GRANITE_4_MICRO,
        api_key=RITS_API_KEY,
        model_options={ModelOption.MAX_NEW_TOKENS: 100},
    )
    return MelleaSession(backend, ctx=ChatContext())


def _print_results(result: SamplingResult) -> None:
    meta = result.result._meta["simba_uq"]
    confidences = meta["all_confidences"]
    temperatures = meta["temperatures_used"]
    sim_matrix = np.array(meta["similarity_matrix"])

    # --- Best response ---
    print("=" * 70)
    print("BEST RESPONSE")
    print("=" * 70)
    print(f"  Index:      {result.result_index}")
    print(f"  Confidence: {meta['confidence']:.4f}")
    print(f"  Metric:     {meta['similarity_metric']}")
    print(f"  Aggregation:{meta['aggregation']}")
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
        truncated = (text[:60] + "...") if len(text) > 60 else text
        marker = " <-- best" if i == result.result_index else ""
        print(
            f"{i:>4}  {temperatures[i]:>5.2f}  {confidences[i]:>8.4f}  {truncated}{marker}"
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


def test_simbauq_rits_dummy():
    """Run SIMBAUQSamplingStrategy with RITSBackend and display results."""
    m = _make_session()

    strategy = SIMBAUQSamplingStrategy(
        temperatures=[0.3, 0.5, 0.7, 1.0],
        n_per_temp=3,
        similarity_metric="sbert",
        aggregation="mean",
    )

    result: SamplingResult = m.instruct(
        "Which magazine was started first Arthur's Magazine or First for Women?",
        strategy=strategy,
        return_sampling_results=True,
    )

    assert isinstance(result, SamplingResult)
    assert len(result.sample_generations) == 12  # 4 temps * 3 per temp

    _print_results(result)

    del m


if __name__ == "__main__":
    test_simbauq_rits_dummy()
