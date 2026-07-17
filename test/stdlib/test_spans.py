# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from mellea.backends import ModelOption

pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_1_3B
from mellea.core import CBlock
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import ChatContext
from mellea.stdlib.session import MelleaSession, start_session
from test.conftest import hf_skip
from test.predicates import require_gpu

# Module-level markers for all tests using Granite 4.1 3B model
pytestmark = [pytest.mark.huggingface, require_gpu(min_vram_gb=12), pytest.mark.e2e]


# We edit the context type in the async tests below. Don't change the scope here.
@pytest.fixture(scope="function")
def m_session(gh_run):
    with hf_skip():
        m = start_session(
            "hf",
            model_id=IBM_GRANITE_4_1_3B,
            model_options={ModelOption.MAX_NEW_TOKENS: 64},
        )
    yield m

    from test.conftest import cleanup_gpu_backend

    cleanup_gpu_backend(m.backend, "spans")
    del m


@pytest.mark.qualitative
async def test_lazy_spans(m_session) -> None:
    m: MelleaSession = m_session
    backend, ctx = m.backend, m.ctx

    x, _ = await m.backend.generate_from_context(CBlock("What is 1+1?"), ctx=ctx)
    y, _ = await m.backend.generate_from_context(CBlock("What is 2+2?"), ctx=ctx)
    # here, x and y have not necessarily been computed!

    response, _ = await backend.generate_from_context(
        SimpleComponent(instruction="What is x+y?", x=x, y=y), ctx=ctx
    )
    result = await response.avalue()
    assert "6" in result, f"Expected 6 ( 1+1 + 2+2 ) but found {result}"


@pytest.mark.qualitative
async def test_kv(m_session) -> None:
    m: MelleaSession = m_session
    backend = m.backend

    # Use a ChatContext (not the session's default SimpleContext): SimpleContext's
    # view_for_generation() always returns [], so any documents added to it are
    # dropped before generation and never reach the model. The KV-cache path must
    # actually see the context for this test to be meaningful.
    ctx = ChatContext().add(
        SimpleComponent(
            doc1="Nathan Fulton is a scientist at the MIT-IBM Watson AI Lab.",
            doc2="The MIT-IBM Watson AI Lab is located at 314 Main Street, Cambridge, MA.",
        )
    )

    assert isinstance(backend, LocalHFBackend)
    # Ask for the lab's street address directly (a single-hop extraction from doc2)
    # rather than "Nathan's work address", which triggered a model safety refusal
    # (see issue #398).
    response = await backend._generate_from_context_with_kv_cache(
        action=CBlock("What is the street address of the MIT-IBM Watson AI Lab?"),
        ctx=ctx,
        model_options=dict(),
    )
    result = await response.avalue()
    assert "314" in result, f"Expected correct answer (314 main st) but found: {result}"


if __name__ == "__main__":
    pytest.main([__file__])
