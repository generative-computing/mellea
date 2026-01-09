import asyncio
from typing import Any, get_args
from mellea import start_session
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import CBlock, ChatContext, Context, ModelOutputThunk, Component
from mellea.stdlib.chat import Message
from mellea.stdlib.genslot import generative
from mellea.stdlib.instruction import Instruction
from mellea.stdlib.requirement import Requirement, ValidationResult,
from mellea import  generative
from mellea.stdlib.sampling import RejectionSamplingStrategy
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.sampling.base import BaseSamplingStrategy

# 1. Works by default with model output thunks.
mot = ModelOutputThunk[float](value="1")
assert hasattr(mot, "__orig_class__"), f"mots are generics and should have this field"
assert get_args(mot.__orig_class__)[0] == float, f"expected float, got {get_args(mot.__orig_class__)[0]} as mot type" # type: ignore

unknown_mot = ModelOutputThunk(value="2")
assert not hasattr(unknown_mot, "__orig_class__"), f"unknown mots / mots with no type defined at instantiate don't have this attribute"

# 2. The output parse type works.
class FloatComp(Component[float]):
    def __init__(self, value: str) -> None:
        self.value = value

    def parts(self) -> list[Component | CBlock]:
        return []

    def format_for_llm(self) -> str:
        return self.value

    # You could technically get rid of the generic typing for Components since components
    #   have to define a parse function. It doesn't save us any effort though and 
    #   complicates the enforcement / typing for ModelOutputThunks that are returned 
    #   through backend.generate.
    def parse(self, computed: ModelOutputThunk) -> float:
        if computed.value is None:
            return -1
        return float(computed.value)

fc = FloatComp(value="generate a float")
assert fc.parse(mot) == 1

# 3. We can subclass them too as long as the types are covariant.
class IntComp(FloatComp, Component[int]):
    def parse(self, computed: ModelOutputThunk) -> int:
        if computed.value is None:
            return -1
        try:
            return int(computed.value)
        except:
            return -2

ic = IntComp("generate an int")
assert ic.parse(mot) == 1

# 4. We can't override the generic type for component subclasses outside of the
#    class definition.
try:
    instruction = Instruction[int](description="this is an instruction") # type: ignore
    assert False, "previous line should have raised a TypeError exception"
except TypeError as e:
    assert "not a generic class" in str(e)
except Exception:
    assert False, "code in the try block should raise a TypeError exception"

# 5. Test in context of generation
m = start_session(ctx=ChatContext().add(CBlock("goodbye")))
out, _ = mfuncs.act(ic, context=ChatContext(), backend=m.backend)
print(out.parsed_repr)

# `out` typed as ModelOutputThunk[str]
out = m.backend.generate_from_context(CBlock(""), ctx=ChatContext())

# `out` typed as ModelOutputThunk[float]
out = m.backend.generate_from_context(ic, ctx=ChatContext())

# `out` typed as ModelOutputThunk[float | str]
out = m.backend.generate_from_raw([ic, CBlock("")], ctx=ChatContext())
# `out` typed as ModelOutputThunk[float]
out = m.backend.generate_from_raw([ic, ic], ctx=ChatContext())
# `out` typed as ModelOutputThunk[str]
out = m.backend.generate_from_raw([CBlock("")], ctx=ChatContext())

# 6. Components that return Components work correctly.
class CompWithComp(Component[Instruction]):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, computed: ModelOutputThunk) -> Instruction:
        return Instruction()

    def format_for_llm(self) -> str:
        return ""

    def parts(self) -> list[Component[str] | CBlock]:
        return []

# typed as ModelOutputThunk[Instruction]
mot_with_comp_type, _ = mfuncs.act(action=CompWithComp(), context=ChatContext(), backend=m.backend)
assert mot_with_comp_type.parsed_repr is not None

# typed as ModelOutputThunk[str]
mot_using_previous_parsed, _ = mfuncs.act(action=mot_with_comp_type.parsed_repr, context=ChatContext(), backend=m.backend)


# 7. Individual backends are get typed correctly.
ob = OllamaModelBackend()
mot = ob.generate_from_context(CBlock(""), ctx=m.ctx)
mot = ob.generate_from_raw([ic, CBlock("")], ctx=m.ctx)

# 8. Example with messages.
user_message = Message("user", "Hello!")
response = m.act(user_message)
assert response.parsed_repr is not None
second_response = m.act(response.parsed_repr)

# 9. Sampling strategies are correctly typed
strat = RejectionSamplingStrategy()

async def sampling_test():
    # typed as SamplingResult[float]
    sampling_result = await strat.sample(ic, context=m.ctx, backend=m.backend, requirements=None)

    # typed as ModelOutputThunk[float]
    sampling_result.result
    # typed as list[ModelOutputThunk[float]]
    sampling_result.sample_generations

    # typed as list[Component[Any]]
    # NOTE: We can't make any guarantees about what action ended up being used to generate a result.
    sampling_result.sample_actions

# 10. Works when returning sampling results from a session / functional level.
# typed as SamplingResult[float]
results = m.act(ic, return_sampling_results=True)
results = mfuncs.act(ic, context=m.ctx, backend=m.backend, return_sampling_results=True)

# typed as SamplingResult[str]
results = m.instruct("hello", return_sampling_results=True)
results = mfuncs.instruct("Hello", context=m.ctx, backend=m.backend, return_sampling_results=True)

# 11. Works with Genslots as well.
@generative
def test(val1: int) -> bool:
    ...

# typed as bool
out = test(m=m, val1=1)


# 12. Test that sampling strategies with repair strats return the correct parsed_repr.
async def sampling_return_type():
    m = start_session()
    class CustomSamplingStrat(BaseSamplingStrategy):
        @staticmethod
        def select_from_failure(sampled_actions: list[Component[Any]], sampled_results: list[ModelOutputThunk[Any]], sampled_val: list[list[tuple[Requirement, ValidationResult]]]) -> int:
            return len(sampled_actions) - 1
        
        @staticmethod
        def repair(old_ctx: Context, new_ctx: Context, past_actions: list[Component[Any]], past_results: list[ModelOutputThunk[Any]], past_val: list[list[tuple[Requirement, ValidationResult]]]) -> tuple[Component[Any], Context]:
            return Instruction("print another number 100 greater"), old_ctx

    css = CustomSamplingStrat(loop_budget=3)
    out = await css.sample(
        action=IntComp("2000"),
        context=ChatContext(),
        backend=m.backend,
        requirements=[
            Requirement(None, validation_fn=lambda x: ValidationResult(False), check_only=True)
        ]
    )

    # Even though the intermediate actions are Instructions, the parsed_reprs at each stage
    # are ints.
    for result in out.sample_generations:
        assert isinstance(result.parsed_repr, int), "model output thunks should have the correct parsed_repr type"

    for action in out.sample_actions[1:]:
        assert isinstance(action, Instruction), "repair strategy should force repaired actions to be Instructions"

asyncio.run(sampling_return_type())

# 13. Random functions that have components / model output thunks in them.
def rand_comp(action: Component):
    ...
rand_comp(ic)

def rand_mot(mot: ModelOutputThunk):
    ...
rand_mot(ModelOutputThunk[int](""))
