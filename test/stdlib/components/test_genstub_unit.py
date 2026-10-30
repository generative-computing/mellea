# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for genstub pure-logic helpers — no backend, no LLM required.

Covers describe_function, get_argument, bind_function_arguments,
create_response_format, GenerativeStub.format_for_llm, and @generative routing.
"""

import inspect
import json
from typing import Any, Literal, get_type_hints
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from mellea import generative
from mellea.core import (
    Backend,
    ModelOutputThunk,
    Requirement,
    TemplateRepresentation,
    ValidationResult,
)
from mellea.core.base import GenerateLog, GenerateType
from mellea.stdlib.components.genstub import (
    ArgPreconditionRequirement,
    Arguments,
    AsyncGenerativeStub,
    Function,
    PreconditionException,
    SyncGenerativeStub,
    _format_type,
    bind_function_arguments,
    create_response_format,
    describe_function,
    get_argument,
)
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements.requirement import reqify
from test.stdlib.components._postponed_annotation_samples import (
    extract_requirements,
    greet,
    price_item,
)

# --- describe_function ---


def test_describe_function_name():
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello {name}"

    result = describe_function(greet)
    assert result["name"] == "greet"


def test_describe_function_signature_includes_params():
    def add(x: int, y: int) -> int:
        return x + y

    result = describe_function(add)
    assert "x" in result["signature"]
    assert "y" in result["signature"]


def test_describe_function_docstring():
    def noop() -> None:
        """Does nothing."""

    result = describe_function(noop)
    assert result["docstring"] == "Does nothing."


def test_describe_function_no_docstring():
    def bare():
        pass

    result = describe_function(bare)
    assert result["docstring"] is None


def test_describe_function_resolves_postponed_annotations():
    # Regression test: `from __future__ import annotations` made
    # `describe_function` render literal annotation strings (e.g.
    # "(product_description: 'str') -> 'list[Requirement]'") instead of the
    # resolved types, corrupting the prompt sent to the model.
    # Guard the precondition: if the fixture module ever drops its
    # `from __future__ import annotations`, this test would otherwise keep
    # passing without exercising postponed annotations at all.
    assert extract_requirements.__annotations__["return"] == "list[Requirement]"

    result = describe_function(extract_requirements)
    assert "'str'" not in result["signature"]
    assert "'list[Requirement]'" not in result["signature"]
    assert "product_description: str" in result["signature"]
    # Types render by name, without the module path (which is noise to the model).
    assert result["signature"].endswith("-> list[Requirement]")
    assert "_postponed_annotation_samples" not in result["signature"]


def test_describe_function_type_checking_only_return_annotation():
    # Regression test: resolving the whole signature with `eval_str=True` also
    # evaluates the return annotation, so a return type imported only under
    # `if TYPE_CHECKING:` raised NameError and no prompt could be built at all.
    # Parameters must still resolve; the unresolvable return stays a postponed
    # string and renders bare, which still names the type for the model.
    # Guard the preconditions: the annotation must still be postponed, and the
    # whole-signature path must still fail, or this stops testing the fallback.
    assert price_item.__annotations__["return"] == "Decimal"
    with pytest.raises(NameError):
        inspect.signature(price_item, eval_str=True)

    signature = describe_function(price_item)["signature"]

    # Parameters resolved: no quoted parameter annotations.
    assert "quantity: int" in signature
    assert "'int'" not in signature
    assert "item: Requirement" in signature
    # Return left postponed, so the model still sees the type name. It renders
    # bare: a quoted `-> 'Decimal'` reads to a small model as "returns a string".
    assert signature.endswith("-> Decimal")


# --- get_argument ---


def test_get_argument_type_checking_only_return_annotation():
    # Regression test: `get_argument` reads only the named parameter's
    # annotation, but resolved the whole signature, so an unresolvable return
    # type raised NameError even though the parameter resolved cleanly.
    # Guard the precondition: same reasoning as the describe_function case.
    with pytest.raises(NameError):
        inspect.signature(price_item, eval_str=True)

    arg = get_argument(price_item, "quantity", 3)
    assert arg._argument_dict["value"] == "3"
    assert "int" in str(arg._argument_dict["annotation"])


def test_get_argument_string_value_quoted():
    def fn(name: str) -> None:
        pass

    arg = get_argument(fn, "name", "Alice")
    assert arg._argument_dict["value"] == '"Alice"'
    assert arg._argument_dict["name"] == "name"


def test_get_argument_int_value_not_quoted():
    def fn(count: int) -> None:
        pass

    arg = get_argument(fn, "count", 42)
    assert arg._argument_dict["value"] == "42"
    assert arg._argument_dict["annotation"] == "int"


def test_get_argument_string_value_quoted_under_postponed_annotations():
    # Regression test: under `from __future__ import annotations`,
    # `param.annotation` was the literal string "str" rather than the `str`
    # type, so the `is str` check failed and string arguments were rendered
    # unquoted in the prompt.
    # Guard the precondition: same reasoning as above.
    assert greet.__annotations__["name"] == "str"

    arg = get_argument(greet, "name", "Alice")
    assert arg._argument_dict["value"] == '"Alice"'


def test_get_argument_no_annotation_falls_back_to_runtime_type():
    # No annotation on kwargs — should fall back to type(val)
    def fn(**kwargs) -> None:
        pass

    arg = get_argument(fn, "x", 3.14)
    assert "float" in str(arg._argument_dict["annotation"])


# --- bind_function_arguments ---


def test_bind_function_arguments_basic():
    def fn(x: int, y: int) -> int:
        return x + y

    result = bind_function_arguments(fn, x=1, y=2)
    assert result == {"x": 1, "y": 2}


def test_bind_function_arguments_with_defaults():
    def fn(x: int, y: int = 10) -> int:
        return x + y

    result = bind_function_arguments(fn, x=5)
    assert result == {"x": 5, "y": 10}


def test_bind_function_arguments_missing_required_raises():
    def fn(x: int, y: int) -> int:
        return x + y

    with pytest.raises(TypeError, match="missing required parameter"):
        bind_function_arguments(fn, x=1)


def test_bind_function_arguments_no_params():
    def fn() -> str:
        return "hi"

    result = bind_function_arguments(fn)
    assert result == {}


# --- create_response_format ---


def test_create_response_format_class_name_derived_from_func():
    def get_sentiment() -> str: ...

    model = create_response_format(get_sentiment)
    assert "GetSentiment" in model.__name__


def test_create_response_format_result_field_accessible():
    def score_text() -> float: ...

    model = create_response_format(score_text)
    instance = model(result=0.9)
    assert instance.result == 0.9


def test_create_response_format_literal_type():
    def classify() -> Literal["pos", "neg"]: ...

    model = create_response_format(classify)
    instance = model(result="pos")
    assert instance.result == "pos"


def test_create_response_format_type_checking_only_return_falls_back_to_any():
    # Regression test: `get_type_hints` resolves every annotation, so a return
    # type imported only under `if TYPE_CHECKING:` raised NameError here. This
    # runs from `GenerativeStub.__init__`, i.e. at decoration time, so the
    # failure made the whole module unimportable rather than degrading anything.
    # Guard the precondition: resolution must still fail, or this tests nothing.
    with pytest.raises(NameError):
        get_type_hints(price_item)

    model = create_response_format(price_item)

    # Degrades to `Any` — the same fallback an unannotated function already
    # gets — rather than failing the decoration.
    assert model.model_fields["result"].annotation is Any
    instance = model(result="anything")
    assert instance.result == "anything"


def test_generative_decoration_survives_type_checking_only_return():
    # The user-facing path: `@generative` calls `create_response_format` and
    # `describe_function` during construction, so both must tolerate a return
    # annotation that cannot be resolved.
    stub = generative(price_item)

    assert stub._response_model.model_fields["result"].annotation is Any
    # Parameters still resolve for the prompt; only the return stays postponed.
    signature = stub._function._function_dict["signature"]
    assert "quantity: int" in signature
    assert signature.endswith("-> Decimal")


# --- GenerativeStub.format_for_llm ---


def test_generative_stub_format_for_llm_returns_template_repr():
    @generative
    def summarise(text: str) -> str:
        """Summarise the given text."""

    result = summarise.format_for_llm()
    assert isinstance(result, TemplateRepresentation)


def test_generative_stub_format_for_llm_includes_function_name():
    @generative
    def my_function(x: int) -> int: ...

    result = my_function.format_for_llm()
    assert result.args["function"]["name"] == "my_function"


def test_generative_stub_format_for_llm_includes_docstring():
    @generative
    def documented() -> str:
        """This is the docstring."""

    result = documented.format_for_llm()
    assert result.args["function"]["docstring"] == "This is the docstring."


def test_generative_stub_format_for_llm_no_args_until_called():
    @generative
    def fn() -> str: ...

    result = fn.format_for_llm()
    assert result.args["arguments"] is None


# --- @generative decorator routing ---


def test_generative_sync_function_returns_sync_stub():
    @generative
    def sync_fn() -> str: ...

    assert isinstance(sync_fn, SyncGenerativeStub)


def test_generative_async_function_returns_async_stub():
    @generative
    async def async_fn() -> str: ...

    assert isinstance(async_fn, AsyncGenerativeStub)


def test_generative_disallowed_param_name_raises():
    with pytest.raises(ValueError, match="disallowed parameter names"):

        @generative
        def fn(backend: str) -> str: ...


# --- Arguments (CBlock subclass rendering bound args) ---


def test_arguments_renders_text():
    def fn(name: str, count: int) -> None:
        pass

    args = [get_argument(fn, "name", "Alice"), get_argument(fn, "count", 3)]
    block = Arguments(args)
    assert "name" in block.value
    assert "count" in block.value


def test_arguments_stores_meta_by_name():
    def fn(x: int) -> None:
        pass

    args = [get_argument(fn, "x", 5)]
    block = Arguments(args)
    assert "x" in block._meta


def test_arguments_empty_list():
    block = Arguments([])
    assert block.value == ""


# --- Function (wraps callable with metadata) ---


def test_function_stores_callable():
    def greet(name: str) -> str:
        """Say hi."""
        return f"hi {name}"

    f = Function(greet)
    assert f._func is greet
    assert f._function_dict["name"] == "greet"
    assert f._function_dict["docstring"] == "Say hi."


# --- ArgPreconditionRequirement (requirement wrapper) ---


def test_arg_precondition_delegates_description():
    req = reqify("must be non-empty")
    wrapper = ArgPreconditionRequirement(req)
    assert wrapper.description == req.description


def test_arg_precondition_copy():
    from copy import copy

    req = reqify("be valid")
    wrapper = ArgPreconditionRequirement(req)
    copied = copy(wrapper)
    assert isinstance(copied, ArgPreconditionRequirement)
    assert copied.req is req


def test_arg_precondition_deepcopy():
    from copy import deepcopy

    req = reqify("be clean")
    wrapper = ArgPreconditionRequirement(req)
    cloned = deepcopy(wrapper)
    assert isinstance(cloned, ArgPreconditionRequirement)
    assert cloned.description == req.description


# --- precondition validation context (PR #1628 review) ---


def test_precondition_validation_renders_no_conversation():
    """The arguments must reach the precondition judge exactly once.

    `ArgPreconditionRequirement.jinja2` already inlines them in its `Arguments:` block, so
    validating over a context that renders history sends them a second time as an
    `assistant` turn (any `ModelOutputThunk` maps to `assistant`). The precondition context
    therefore has to be one that renders nothing.
    """
    seen: list[Any] = []

    def capture_and_fail(ctx):
        seen.append(ctx)
        return ValidationResult(result=False, reason="forced failure")

    @generative
    def classify(text: str) -> str: ...

    with pytest.raises(PreconditionException):
        classify(
            context=SimpleContext(),
            backend=MagicMock(),
            text="hello",
            precondition_requirements=[
                Requirement("forced failure", validation_fn=capture_and_fail)
            ],
        )

    assert seen, "the precondition requirement must actually have been validated"
    assert seen[0].view_for_generation() == [], (
        "the precondition context must render no conversation, or the arguments reach the"
        " judge twice"
    )
    assert "hello" in str(seen[0].last_output()), (
        "the arguments must still be the validation target, so the template inlines them"
    )


# --- PreconditionException ---


def test_precondition_exception_message():
    vr = ValidationResult(result=False, reason="failed check")
    exc = PreconditionException("precondition failed", [vr])
    assert "precondition failed" in str(exc)
    assert exc.validation == [vr]


# --- GenerativeStub._parse ---


def test_genstub_parse_json_to_result():
    import json

    from mellea.core import ModelOutputThunk

    @generative
    def classify(text: str) -> str: ...

    mot = ModelOutputThunk(value=json.dumps({"result": "positive"}))
    parsed = classify._parse(mot)
    assert parsed == "positive"


def test_genstub_parse_int_result():
    import json

    from mellea.core import ModelOutputThunk

    @generative
    def compute(x: int) -> int: ...

    mot = ModelOutputThunk(value=json.dumps({"result": 42}))
    parsed = compute._parse(mot)
    assert parsed == 42


# --- prompt rendering (issue #1476) ---


class _Person(BaseModel):
    name: str
    age: int


class _Team(BaseModel):
    lead: _Person
    members: list[_Person]


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (str, "str"),
        (list[int], "list[int]"),
        (dict[str, list[_Person]], "dict[str, list[_Person]]"),
        (_Person | None, "_Person | None"),
        (Literal["a.b", "c"], "Literal['a.b', 'c']"),
        (tuple[int, ...], "tuple[int, ...]"),
        ("Decimal", "Decimal"),
    ],
)
def test_format_type_renders_bare_names(annotation, expected):
    # `str(type)` renders `<class 'str'>` and `inspect.formatannotation` keeps module
    # paths (`test.stdlib...._Person`); both are noise to the model. A `Literal`'s
    # values must survive untouched even when they contain dots.
    assert _format_type(annotation) == expected


def test_get_argument_annotation_is_bare_type_name():
    def fn(name: str, people: list[_Person]) -> None:
        pass

    assert get_argument(fn, "name", "x")._argument_dict["annotation"] == "str"
    assert (
        get_argument(fn, "people", [])._argument_dict["annotation"] == "list[_Person]"
    )


def test_get_argument_renders_structured_values_as_json():
    # `str()` shows a Pydantic model as `name='Ada' age=36`; JSON matches what the
    # model emits for structured output.
    def fn(team: _Team, scores: dict[str, int]) -> None:
        pass

    team = _Team(lead=_Person(name="Ada", age=36), members=[])
    arg = get_argument(fn, "team", team)
    assert json.loads(arg._argument_dict["value"]) == team.model_dump()
    arg = get_argument(fn, "scores", {"a": 1})
    assert arg._argument_dict["value"] == '{"a": 1}'


def test_describe_function_return_type():
    def fn(x: int) -> list[_Person]: ...

    def bare(x: int): ...

    assert describe_function(fn)["return_type"] == "list[_Person]"
    assert describe_function(bare)["return_type"] == "Any"


@pytest.mark.parametrize(
    "return_type, expected",
    [
        (str, "string"),
        (bool, "boolean"),
        (Literal["yes", "no"], '"yes" | "no"'),
        (str | None, "string | null"),
        (list[str], "[string, ...]"),
        (dict, "object"),
        (
            _Team,
            '{"lead": {"name": string, "age": integer},'
            ' "members": [{"name": string, "age": integer}, ...]}',
        ),
    ],
)
def test_result_shape(return_type, expected):
    def fn() -> None: ...

    fn.__annotations__["return"] = return_type
    assert generative(fn).format_for_llm().args["result_shape"] == expected


def test_result_shape_recursive_model_terminates():
    class Node(BaseModel):
        value: int
        children: list["Node"] = []

    @generative
    def tree() -> Node: ...

    assert tree.format_for_llm().args["result_shape"] == (
        '{"value": integer, "children": [Node, ...]}'
    )


def _render(stub, **kwargs) -> str:
    from copy import deepcopy

    from mellea.formatters import TemplateFormatter

    stub = deepcopy(stub)
    stub._arguments = Arguments(
        [get_argument(stub._function._func, k, v) for k, v in kwargs.items()]
    )
    return TemplateFormatter(model_id="default").print(stub)


def test_generative_stub_prompt_states_task_inputs_and_json_shape():
    @generative
    def extract_people(text: str) -> list[_Person]:
        """Extract every person mentioned in the text."""

    prompt = _render(extract_people, text="Ada, 36, met Alan.")
    assert prompt.startswith("Task: extract_people\n\nExtract every person")
    assert '- text: "Ada, 36, met Alan."  (type: str)' in prompt
    assert '{"result": <value of type list[_Person]>}' in prompt
    assert '{"result": [{"name": string, "age": integer}, ...]}' in prompt
    # The old "imitate the output of the following function" framing is gone.
    assert "imitate" not in prompt and "def extract_people" not in prompt


def test_generative_stub_prompt_without_docstring_has_no_placeholder():
    # The old template's `default("No documentation provided.")` never fired for a
    # `None` docstring and rendered an empty `""""""` block instead.
    @generative
    def is_question(sentence: str) -> bool: ...

    prompt = _render(is_question, sentence="Why?")
    assert prompt.startswith("Task: is_question\n\nInputs:")
    assert '""' not in prompt


def test_generative_stub_prompt_keeps_multiline_docstring_intact():
    @generative
    def identify_fruit(text: str) -> str | None:
        """
        Extract fruit from text, or None if none mentioned.
        Ex: "I ate an apple" -> "apple"
        """

    prompt = _render(identify_fruit, text="x")
    assert (
        'Extract fruit from text, or None if none mentioned.\nEx: "I ate an apple"'
        in prompt
    )


def test_generative_stub_prompt_lists_requirements():
    @generative
    def write_headline(article: str) -> str:
        """Write a headline for the article."""

    stub = write_headline
    from copy import deepcopy

    stub = deepcopy(stub)
    stub.requirements = [reqify("At most 10 words.")]
    stub._arguments = Arguments([get_argument(stub._function._func, "article", "a")])
    from mellea.formatters import TemplateFormatter

    prompt = TemplateFormatter(model_id="default").print(stub)
    assert "Requirements:\n- At most 10 words." in prompt


# --- `None` results (issue #1476) ---


class _FixedJSONBackend(Backend):
    """Test double that answers every generation with a fixed JSON body."""

    _model_id = "test-model"
    _provider = "test"

    def __init__(self, body: str):
        self._body = body

    async def _generate_from_context(
        self, action, ctx, *, format=None, model_options=None, tool_calls=False
    ):
        mot = ModelOutputThunk(value=None)
        mot._call.action = action
        mot._gen.generate_type = GenerateType.ASYNC

        async def _process(m: ModelOutputThunk, chunk) -> None:
            if chunk is not None:
                m._underlying_value = (m._underlying_value or "") + chunk

        async def _post_process(m: ModelOutputThunk) -> None:
            m._generate_log = GenerateLog()

        mot._gen.process = _process
        mot._gen.post_process = _post_process
        mot._gen.chunk_size = 0
        mot._gen.queue.put_nowait(self._body)
        mot._gen.queue.put_nowait(None)
        return mot, ctx.add(action).add(mot)

    async def _generate_from_raw(self, *args, **kwargs):
        raise NotImplementedError


def test_sync_stub_returns_none_result():
    """A stub whose return type admits `None` returns `None` instead of crashing.

    Regression test: `{"result": null}` parsed to `None`, which tripped the
    "non-`None` parsed_repr" invariant in `ModelOutputThunk` (and the stub's own
    assert), so the documented `-> str | None` pattern crashed exactly when the
    correct answer was `None`.
    """

    @generative
    def identify_fruit(text: str) -> str | None:
        """Extract the fruit mentioned in the text, or None if none is mentioned."""

    backend = _FixedJSONBackend('{"result": null}')
    result, _ = identify_fruit(
        context=SimpleContext(), backend=backend, text="The car is parked outside."
    )
    assert result is None

    backend = _FixedJSONBackend('{"result": "banana"}')
    result, _ = identify_fruit(context=SimpleContext(), backend=backend, text="x")
    assert result == "banana"


async def test_async_stub_returns_none_result():
    @generative
    async def identify_fruit(text: str) -> str | None:
        """Extract the fruit mentioned in the text, or None if none is mentioned."""

    backend = _FixedJSONBackend('{"result": null}')
    result, _ = await identify_fruit(
        context=SimpleContext(), backend=backend, text="The car is parked outside."
    )
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
