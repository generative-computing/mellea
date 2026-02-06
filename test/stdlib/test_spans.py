import pytest

from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_HYBRID_MICRO
from mellea.core import CBlock
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.session import MelleaSession, start_session

# Module-level markers for all tests using Granite 4 hybrid micro (3B model)
pytestmark = [pytest.mark.huggingface, pytest.mark.requires_gpu, pytest.mark.llm]


# We edit the context type in the async tests below. Don't change the scope here.
@pytest.fixture(scope="function")
def m_session(gh_run):
    m = start_session(
        "hf",
        model_id=IBM_GRANITE_4_HYBRID_MICRO,
        model_options={ModelOption.MAX_NEW_TOKENS: 64},
    )
    yield m
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


@pytest.mark.xfail(
    strict=False, reason="Model safety refusal despite context - see issue #398"
)
@pytest.mark.qualitative
async def test_kv(m_session) -> None:
    m: MelleaSession = m_session
    backend, ctx = m.backend, m.ctx  # type: ignore

    ctx = ctx.add(
        SimpleComponent(
            doc1="Nathan Fulton is a scientist at the MIT-IBM Watson AI Lab.",
            doc2="The MIT-IBM Watson AI Lab is located at 314 Main Street, Cambridge, MA.",
        )
    )

    assert isinstance(backend, LocalHFBackend)
    response = await backend._generate_from_context_with_kv_cache(
        action=CBlock("What is Nathan's work address?"), ctx=ctx, model_options=dict()
    )
    result = await response.avalue()
    assert "314" in result, f"Expected correct answer (314 main st) but found: {result}"


if __name__ == "__main__":
    pytest.main([__file__])
