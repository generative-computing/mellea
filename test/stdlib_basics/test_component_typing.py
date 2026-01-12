"""Tests for checking the functionality of typed components, model output thunks and sampling results."""

import pytest
from typing import get_args
from mellea import start_session
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import (
    CBlock,
    ChatContext,
    Component,
    ComponentParseError,
    Context,
    ModelOutputThunk,
    SimpleContext,
)
from mellea.stdlib.chat import Message
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.sampling.base import BaseSamplingStrategy
from mellea.stdlib.session import MelleaSession

import mellea.stdlib.functional as mfuncs


class FloatComp(Component[float]):
    def __init__(self, value: str) -> None:
        self.value = value

    def parts(self) -> list[Component | CBlock]:
        return []

    def format_for_llm(self) -> str:
        return self.value

    def _parse(self, computed: ModelOutputThunk) -> float:
        if computed.value is None:
            return -1
        return float(computed.value)


class IntComp(FloatComp, Component[int]):
    def _parse(self, computed: ModelOutputThunk) -> int:
        if computed.value is None:
            return -1
        try:
            return int(computed.value)
        except:
            return -2


class ExceptionRaisingComp(Component[int]):
    def parts(self) -> list[Component | CBlock]:
        return []

    def format_for_llm(self) -> str:
        return ""

    def _parse(self, computed: ModelOutputThunk) -> int:
        raise ValueError("random error")


@pytest.fixture(scope="module")
def backend(gh_run: int):
    """Shared backend."""
    if gh_run == 1:
        return OllamaModelBackend(
            model_id=IBM_GRANITE_4_MICRO_3B.ollama_name  # type: ignore
        )
    else:
        return OllamaModelBackend(model_id="granite3.3:8b")


@pytest.fixture(scope="module")
def session(backend) -> MelleaSession:
    return MelleaSession(backend=backend, ctx=SimpleContext())


def test_mot_init_typing():
    mot = ModelOutputThunk[float](value="1")
    assert hasattr(mot, "__orig_class__"), (
        f"mots are generics and should have this field"
    )
    assert get_args(mot.__orig_class__)[0] == float, (
        f"expected float, got {get_args(mot.__orig_class__)[0]} as mot type"
    )  # type: ignore

    unknown_mot = ModelOutputThunk(value="2")
    assert not hasattr(unknown_mot, "__orig_class__"), (
        f"unknown mots / mots with no type defined at instantiate don't have this attribute"
    )


def test_simple_component_parsing():
    fc = FloatComp(value="generate a float")
    mot = ModelOutputThunk[float](value="1")
    assert fc.parse(mot) == 1
    assert isinstance(fc.parse(mot), float)


def test_subclassed_component_parsing():
    ic = IntComp("generate an int")
    mot = ModelOutputThunk[float](value="1")
    assert ic.parse(mot) == 1


def test_component_parsing_fails():
    erc = ExceptionRaisingComp()
    mot = ModelOutputThunk[float](value="1")

    with pytest.raises(ComponentParseError):
        _ = erc.parse(mot) == 1


def test_incorrect_type_override():
    with pytest.raises(TypeError):
        instruction = Instruction[int](description="this is an instruction")  # type: ignore


# Marking as qualitative for now since there's so much generation required for this.
@pytest.mark.qualitative
async def test_generating(session):
    m = session
    ic = IntComp("generate an int")

    out, _ = mfuncs.act(ic, context=ChatContext(), backend=m.backend, strategy=None)
    assert isinstance(out.parsed_repr, int)

    # `out` typed as ModelOutputThunk[str]
    out, _ = await m.backend.generate_from_context(
        CBlock("Say Hello!"), ctx=ChatContext()
    )
    await out.avalue()
    assert isinstance(out.parsed_repr, str)

    # `out` typed as ModelOutputThunk[float]
    out, _ = await m.backend.generate_from_context(ic, ctx=ChatContext())
    await out.avalue()
    assert isinstance(out.parsed_repr, int)

    # `out` typed as ModelOutputThunk[float | str]
    out = await m.backend.generate_from_raw([ic, CBlock("")], ctx=ChatContext())
    for result in out:
        await result.avalue()
    assert isinstance(out[0].parsed_repr, int)
    assert isinstance(out[1].parsed_repr, str)

    # `out` typed as ModelOutputThunk[float]
    out = await m.backend.generate_from_raw([ic, ic], ctx=ChatContext())
    for result in out:
        await result.avalue()
        assert isinstance(result.parsed_repr, int)

    # `out` typed as ModelOutputThunk[str]
    out = await m.backend.generate_from_raw([CBlock("")], ctx=ChatContext())
    for result in out:
        await result.avalue()
        assert isinstance(result.parsed_repr, str)


@pytest.mark.qualitative
def test_message_typing(session):
    m = session
    user_message = Message("user", "Hello!")
    response = m.act(user_message)
    assert response.parsed_repr is not None
    assert isinstance(response.parsed_repr, Message)

    second_response = m.act(response.parsed_repr)
    assert second_response.parsed_repr is not None
    assert isinstance(second_response.parsed_repr, Message)


@pytest.mark.qualitative
async def test_generating_with_sampling(session):
    m = session
    m = start_session()

    class CustomSamplingStrat(BaseSamplingStrategy):
        @staticmethod
        def select_from_failure(
            sampled_actions: list[Component],
            sampled_results: list[ModelOutputThunk],
            sampled_val: list[list[tuple[Requirement, ValidationResult]]],
        ) -> int:
            return len(sampled_actions) - 1

        @staticmethod
        def repair(
            old_ctx: Context,
            new_ctx: Context,
            past_actions: list[Component],
            past_results: list[ModelOutputThunk],
            past_val: list[list[tuple[Requirement, ValidationResult]]],
        ) -> tuple[Component, Context]:
            return Instruction("print another number 100 greater"), old_ctx

    css = CustomSamplingStrat(loop_budget=3)
    out = await css.sample(
        action=IntComp("2000"),
        context=ChatContext(),
        backend=m.backend,
        requirements=[
            Requirement(
                None, validation_fn=lambda x: ValidationResult(False), check_only=True
            )
        ],
    )

    # Even though the intermediate actions are Instructions, the parsed_reprs at each stage
    # are ints.
    for result in out.sample_generations:
        assert isinstance(result.parsed_repr, int), (
            "model output thunks should have the correct parsed_repr type"
        )

    for action in out.sample_actions[1:]:
        assert isinstance(action, Instruction), (
            "repair strategy should force repaired actions to be Instructions"
        )


if __name__ == "__main__":
    pytest.main([__file__])
