"""Tests for all remaining hook payload models not covered in test_payloads.py.

Covers:
- SessionPostInitPayload, SessionResetPayload, SessionCleanupPayload
- ComponentPreExecutePayload, ComponentPostSuccessPayload, ComponentPostErrorPayload
- GenerationPostCallPayload, GenerationStreamChunkPayload
- ValidationPreCheckPayload, ValidationPostCheckPayload
- SamplingLoopStartPayload, SamplingIterationPayload,
  SamplingRepairPayload, SamplingLoopEndPayload
- ToolPreInvokePayload, ToolPostInvokePayload
"""

import pytest
from pydantic import ValidationError

pytest.importorskip("cpex.framework")

from mellea.plugins.base import MelleaBasePayload
from mellea.plugins.hooks.component import (
    ComponentPostErrorPayload,
    ComponentPostSuccessPayload,
    ComponentPreExecutePayload,
)
from mellea.plugins.hooks.generation import (
    GenerationPostCallPayload,
    GenerationStreamChunkPayload,
)
from mellea.plugins.hooks.sampling import (
    SamplingIterationPayload,
    SamplingLoopEndPayload,
    SamplingLoopStartPayload,
    SamplingRepairPayload,
)
from mellea.plugins.hooks.session import (
    SessionCleanupPayload,
    SessionPostInitPayload,
    SessionResetPayload,
)
from mellea.plugins.hooks.tool import ToolPostInvokePayload, ToolPreInvokePayload
from mellea.plugins.hooks.validation import (
    ValidationPostCheckPayload,
    ValidationPreCheckPayload,
)

# ---------------------------------------------------------------------------
# Sentinel objects used as stand-ins for live Mellea references (Any fields).
# ---------------------------------------------------------------------------
_SENTINEL_SESSION = object()
_SENTINEL_CONTEXT = object()
_SENTINEL_ACTION = object()
_SENTINEL_COMPONENT = object()
_SENTINEL_RESULT = object()
_SENTINEL_STRATEGY = object()
_SENTINEL_FORMAT = object()
_SENTINEL_GENERATE_LOG = object()
_SENTINEL_ERROR = RuntimeError("boom")
_SENTINEL_CALLABLE = lambda: None  # noqa: E731
_SENTINEL_TOOL_CALL = object()
_SENTINEL_TOOL_MESSAGE = object()
_SENTINEL_REQUIREMENT = object()
_SENTINEL_VALIDATION_RESULT = object()


# ===========================================================================
# Session payloads
# ===========================================================================


class TestSessionPostInitPayload:
    def test_defaults(self):
        payload = SessionPostInitPayload()
        assert payload.session is None
        assert payload.session_id is None
        assert payload.request_id == ""
        assert payload.hook == ""
        assert payload.user_metadata == {}
        assert payload.timestamp is not None

    def test_construction_with_values(self):
        payload = SessionPostInitPayload(
            session=_SENTINEL_SESSION,
            session_id="s-001",
            request_id="r-001",
            hook="session_post_init",
        )
        assert payload.session is _SENTINEL_SESSION
        assert payload.session_id == "s-001"
        assert payload.request_id == "r-001"
        assert payload.hook == "session_post_init"

    def test_frozen(self):
        payload = SessionPostInitPayload(session=_SENTINEL_SESSION)
        with pytest.raises(ValidationError):
            payload.session = object()

    def test_frozen_base_field(self):
        payload = SessionPostInitPayload(request_id="r-001")
        with pytest.raises(ValidationError):
            payload.request_id = "r-002"

    def test_model_copy_creates_modified_copy(self):
        new_session = object()
        payload = SessionPostInitPayload(session=_SENTINEL_SESSION, request_id="r-001")
        modified = payload.model_copy(update={"session": new_session})
        assert modified.session is new_session
        assert payload.session is _SENTINEL_SESSION
        assert modified.request_id == "r-001"

    def test_inherits_base_fields(self):
        assert issubclass(SessionPostInitPayload, MelleaBasePayload)
        payload = SessionPostInitPayload(
            session_id="s-abc",
            request_id="r-abc",
            hook="my_hook",
            user_metadata={"env": "test"},
        )
        assert payload.session_id == "s-abc"
        assert payload.user_metadata == {"env": "test"}


class TestSessionResetPayload:
    def test_defaults(self):
        payload = SessionResetPayload()
        assert payload.previous_context is None
        assert payload.session_id is None
        assert payload.request_id == ""

    def test_construction_with_values(self):
        payload = SessionResetPayload(
            previous_context=_SENTINEL_CONTEXT, session_id="s-002", hook="session_reset"
        )
        assert payload.previous_context is _SENTINEL_CONTEXT
        assert payload.session_id == "s-002"
        assert payload.hook == "session_reset"

    def test_frozen(self):
        payload = SessionResetPayload(previous_context=_SENTINEL_CONTEXT)
        with pytest.raises(ValidationError):
            payload.previous_context = object()

    def test_frozen_base_field(self):
        payload = SessionResetPayload(hook="session_reset")
        with pytest.raises(ValidationError):
            payload.hook = "other_hook"

    def test_model_copy_creates_modified_copy(self):
        new_ctx = object()
        payload = SessionResetPayload(
            previous_context=_SENTINEL_CONTEXT, session_id="s-002"
        )
        modified = payload.model_copy(update={"previous_context": new_ctx})
        assert modified.previous_context is new_ctx
        assert payload.previous_context is _SENTINEL_CONTEXT
        assert modified.session_id == "s-002"

    def test_inherits_base_fields(self):
        assert issubclass(SessionResetPayload, MelleaBasePayload)


class TestSessionCleanupPayload:
    def test_defaults(self):
        payload = SessionCleanupPayload()
        assert payload.context is None
        assert payload.interaction_count == 0
        assert payload.session_id is None

    def test_construction_with_values(self):
        payload = SessionCleanupPayload(
            context=_SENTINEL_CONTEXT,
            interaction_count=42,
            session_id="s-003",
            hook="session_cleanup",
        )
        assert payload.context is _SENTINEL_CONTEXT
        assert payload.interaction_count == 42
        assert payload.session_id == "s-003"

    def test_frozen(self):
        payload = SessionCleanupPayload(interaction_count=5)
        with pytest.raises(ValidationError):
            payload.interaction_count = 6

    def test_frozen_context_field(self):
        payload = SessionCleanupPayload(context=_SENTINEL_CONTEXT)
        with pytest.raises(ValidationError):
            payload.context = object()

    def test_model_copy_creates_modified_copy(self):
        payload = SessionCleanupPayload(context=_SENTINEL_CONTEXT, interaction_count=10)
        modified = payload.model_copy(update={"interaction_count": 99})
        assert modified.interaction_count == 99
        assert payload.interaction_count == 10
        assert modified.context is _SENTINEL_CONTEXT

    def test_inherits_base_fields(self):
        assert issubclass(SessionCleanupPayload, MelleaBasePayload)
        payload = SessionCleanupPayload(user_metadata={"run": "cleanup"})
        assert payload.user_metadata == {"run": "cleanup"}


# ===========================================================================
# Component payloads
# ===========================================================================


class TestComponentPreExecutePayload:
    def test_defaults(self):
        payload = ComponentPreExecutePayload()
        assert payload.component_type == ""
        assert payload.action is None
        assert payload.context is None
        assert payload.context_view is None
        assert payload.requirements == []
        assert payload.model_options == {}
        assert payload.format is None
        assert payload.strategy is None
        assert payload.tool_calls_enabled is False

    def test_construction_with_values(self):
        reqs = [_SENTINEL_REQUIREMENT, _SENTINEL_REQUIREMENT]
        payload = ComponentPreExecutePayload(
            component_type="Extractor",
            action=_SENTINEL_ACTION,
            context=_SENTINEL_CONTEXT,
            context_view=["msg_a", "msg_b"],
            requirements=reqs,
            model_options={"temperature": 0.2},
            format=_SENTINEL_FORMAT,
            strategy=_SENTINEL_STRATEGY,
            tool_calls_enabled=True,
            request_id="r-exec-001",
        )
        assert payload.component_type == "Extractor"
        assert payload.action is _SENTINEL_ACTION
        assert payload.context is _SENTINEL_CONTEXT
        assert payload.context_view == ["msg_a", "msg_b"]
        assert len(payload.requirements) == 2
        assert payload.model_options == {"temperature": 0.2}
        assert payload.format is _SENTINEL_FORMAT
        assert payload.strategy is _SENTINEL_STRATEGY
        assert payload.tool_calls_enabled is True

    def test_frozen(self):
        payload = ComponentPreExecutePayload(tool_calls_enabled=False)
        with pytest.raises(ValidationError):
            payload.tool_calls_enabled = True

    def test_frozen_model_options(self):
        payload = ComponentPreExecutePayload(model_options={"temperature": 0.5})
        with pytest.raises(ValidationError):
            payload.model_options = {}

    def test_model_copy_creates_modified_copy(self):
        payload = ComponentPreExecutePayload(
            component_type="Extractor",
            model_options={"temperature": 0.5},
            tool_calls_enabled=False,
        )
        modified = payload.model_copy(
            update={"model_options": {"temperature": 0.9}, "tool_calls_enabled": True}
        )
        assert modified.model_options == {"temperature": 0.9}
        assert modified.tool_calls_enabled is True
        assert payload.model_options == {"temperature": 0.5}
        assert payload.tool_calls_enabled is False
        assert modified.component_type == "Extractor"

    def test_context_view_none_vs_empty(self):
        payload_none = ComponentPreExecutePayload(context_view=None)
        payload_list = ComponentPreExecutePayload(context_view=[])
        assert payload_none.context_view is None
        assert payload_list.context_view == []

    def test_inherits_base_fields(self):
        assert issubclass(ComponentPreExecutePayload, MelleaBasePayload)


class TestComponentPostSuccessPayload:
    def test_defaults(self):
        payload = ComponentPostSuccessPayload()
        assert payload.component_type == ""
        assert payload.action is None
        assert payload.result is None
        assert payload.context_before is None
        assert payload.context_after is None
        assert payload.generate_log is None
        assert payload.sampling_results is None
        assert payload.latency_ms == 0

    def test_construction_with_values(self):
        sampling = [object(), object()]
        payload = ComponentPostSuccessPayload(
            component_type="Instruction",
            action=_SENTINEL_ACTION,
            result=_SENTINEL_RESULT,
            context_before=_SENTINEL_CONTEXT,
            context_after=_SENTINEL_CONTEXT,
            generate_log=_SENTINEL_GENERATE_LOG,
            sampling_results=sampling,
            latency_ms=250,
            request_id="r-success-001",
        )
        assert payload.component_type == "Instruction"
        assert payload.result is _SENTINEL_RESULT
        assert payload.latency_ms == 250
        assert (
            payload.sampling_results is not None and len(payload.sampling_results) == 2
        )

    def test_frozen(self):
        payload = ComponentPostSuccessPayload(latency_ms=100)
        with pytest.raises(ValidationError):
            payload.latency_ms = 200

    def test_frozen_component_type(self):
        payload = ComponentPostSuccessPayload(component_type="Instruction")
        with pytest.raises(ValidationError):
            payload.component_type = "Extractor"

    def test_model_copy_creates_modified_copy(self):
        payload = ComponentPostSuccessPayload(
            component_type="Instruction", latency_ms=100
        )
        modified = payload.model_copy(update={"latency_ms": 999})
        assert modified.latency_ms == 999
        assert payload.latency_ms == 100
        assert modified.component_type == "Instruction"

    def test_sampling_results_none_vs_list(self):
        payload_none = ComponentPostSuccessPayload(sampling_results=None)
        payload_list = ComponentPostSuccessPayload(sampling_results=[object()])
        assert payload_none.sampling_results is None
        assert (
            payload_list.sampling_results is not None
            and len(payload_list.sampling_results) == 1
        )

    def test_inherits_base_fields(self):
        assert issubclass(ComponentPostSuccessPayload, MelleaBasePayload)


class TestComponentPostErrorPayload:
    def test_defaults(self):
        payload = ComponentPostErrorPayload()
        assert payload.component_type == ""
        assert payload.action is None
        assert payload.error is None
        assert payload.error_type == ""
        assert payload.stack_trace == ""
        assert payload.context is None
        assert payload.model_options == {}

    def test_construction_with_values(self):
        payload = ComponentPostErrorPayload(
            component_type="Extractor",
            action=_SENTINEL_ACTION,
            error=_SENTINEL_ERROR,
            error_type="RuntimeError",
            stack_trace="Traceback ...",
            context=_SENTINEL_CONTEXT,
            model_options={"temperature": 0.0},
            request_id="r-err-001",
        )
        assert payload.component_type == "Extractor"
        assert payload.error is _SENTINEL_ERROR
        assert payload.error_type == "RuntimeError"
        assert payload.stack_trace == "Traceback ..."
        assert payload.model_options == {"temperature": 0.0}

    def test_frozen(self):
        payload = ComponentPostErrorPayload(error_type="ValueError")
        with pytest.raises(ValidationError):
            payload.error_type = "RuntimeError"

    def test_frozen_stack_trace(self):
        payload = ComponentPostErrorPayload(stack_trace="trace line 1")
        with pytest.raises(ValidationError):
            payload.stack_trace = "new trace"

    def test_model_copy_creates_modified_copy(self):
        payload = ComponentPostErrorPayload(
            component_type="Extractor",
            error_type="ValueError",
            stack_trace="original trace",
        )
        modified = payload.model_copy(
            update={"error_type": "RuntimeError", "stack_trace": "new trace"}
        )
        assert modified.error_type == "RuntimeError"
        assert modified.stack_trace == "new trace"
        assert payload.error_type == "ValueError"
        assert payload.stack_trace == "original trace"
        assert modified.component_type == "Extractor"

    def test_inherits_base_fields(self):
        assert issubclass(ComponentPostErrorPayload, MelleaBasePayload)


# ===========================================================================
# Generation payloads
# ===========================================================================


class TestGenerationPostCallPayload:
    def test_defaults(self):
        payload = GenerationPostCallPayload()
        assert payload.prompt == ""
        assert payload.model_output is None
        assert payload.latency_ms == 0

    def test_construction_with_string_prompt(self):
        payload = GenerationPostCallPayload(
            prompt="What is the capital of France?",
            model_output=_SENTINEL_RESULT,
            latency_ms=312,
            request_id="r-gen-001",
        )
        assert payload.prompt == "What is the capital of France?"
        assert payload.latency_ms == 312

    def test_construction_with_list_prompt(self):
        messages = [{"role": "user", "content": "Hello"}]
        payload = GenerationPostCallPayload(prompt=messages)
        assert payload.prompt == messages

    def test_frozen_latency(self):
        payload = GenerationPostCallPayload(latency_ms=100)
        with pytest.raises(ValidationError):
            payload.latency_ms = 200

    def test_model_copy_creates_modified_copy(self):
        payload = GenerationPostCallPayload(prompt="original prompt", latency_ms=100)
        modified = payload.model_copy(update={"latency_ms": 500})
        assert modified.latency_ms == 500
        assert payload.latency_ms == 100
        assert modified.prompt == "original prompt"

    def test_inherits_base_fields(self):
        assert issubclass(GenerationPostCallPayload, MelleaBasePayload)


class TestGenerationStreamChunkPayload:
    def test_defaults(self):
        payload = GenerationStreamChunkPayload()
        assert payload.chunk == ""
        assert payload.accumulated == ""
        assert payload.chunk_index == 0
        assert payload.is_final is False

    def test_construction_with_values(self):
        payload = GenerationStreamChunkPayload(
            chunk="aris",
            accumulated="Paris",
            chunk_index=3,
            is_final=False,
            request_id="r-stream-001",
        )
        assert payload.chunk == "aris"
        assert payload.accumulated == "Paris"
        assert payload.chunk_index == 3
        assert payload.is_final is False

    def test_final_chunk(self):
        payload = GenerationStreamChunkPayload(
            chunk=".", accumulated="Paris.", chunk_index=10, is_final=True
        )
        assert payload.is_final is True

    def test_frozen(self):
        payload = GenerationStreamChunkPayload(chunk="hello")
        with pytest.raises(ValidationError):
            payload.chunk = "world"

    def test_frozen_is_final(self):
        payload = GenerationStreamChunkPayload(is_final=False)
        with pytest.raises(ValidationError):
            payload.is_final = True

    def test_model_copy_creates_modified_copy(self):
        payload = GenerationStreamChunkPayload(
            chunk="hel",  # codespell:ignore
            accumulated="hel",  # codespell:ignore
            chunk_index=0,
        )
        modified = payload.model_copy(
            update={"chunk": "lo", "accumulated": "hello", "chunk_index": 1}
        )
        assert modified.chunk == "lo"
        assert modified.accumulated == "hello"
        assert modified.chunk_index == 1
        assert payload.chunk == "hel"  # codespell:ignore
        assert payload.accumulated == "hel"  # codespell:ignore

    def test_inherits_base_fields(self):
        assert issubclass(GenerationStreamChunkPayload, MelleaBasePayload)


# ===========================================================================
# Validation payloads
# ===========================================================================


class TestValidationPreCheckPayload:
    def test_defaults(self):
        payload = ValidationPreCheckPayload()
        assert payload.requirements == []
        assert payload.target is None
        assert payload.context is None
        assert payload.model_options == {}

    def test_construction_with_values(self):
        reqs = [_SENTINEL_REQUIREMENT, _SENTINEL_REQUIREMENT]
        payload = ValidationPreCheckPayload(
            requirements=reqs,
            target=_SENTINEL_ACTION,
            context=_SENTINEL_CONTEXT,
            model_options={"temperature": 0.0},
            request_id="r-val-001",
        )
        assert len(payload.requirements) == 2
        assert payload.target is _SENTINEL_ACTION
        assert payload.context is _SENTINEL_CONTEXT
        assert payload.model_options == {"temperature": 0.0}

    def test_frozen(self):
        payload = ValidationPreCheckPayload(requirements=[_SENTINEL_REQUIREMENT])
        with pytest.raises(ValidationError):
            payload.requirements = []

    def test_frozen_model_options(self):
        payload = ValidationPreCheckPayload(model_options={"temperature": 0.5})
        with pytest.raises(ValidationError):
            payload.model_options = {}

    def test_model_copy_creates_modified_copy(self):
        reqs = [_SENTINEL_REQUIREMENT]
        payload = ValidationPreCheckPayload(
            requirements=reqs, context=_SENTINEL_CONTEXT
        )
        new_reqs = [_SENTINEL_REQUIREMENT, _SENTINEL_REQUIREMENT]
        modified = payload.model_copy(update={"requirements": new_reqs})
        assert len(modified.requirements) == 2
        assert len(payload.requirements) == 1
        assert modified.context is _SENTINEL_CONTEXT

    def test_inherits_base_fields(self):
        assert issubclass(ValidationPreCheckPayload, MelleaBasePayload)
        payload = ValidationPreCheckPayload(
            session_id="s-val", user_metadata={"phase": "pre"}
        )
        assert payload.session_id == "s-val"
        assert payload.user_metadata == {"phase": "pre"}


class TestValidationPostCheckPayload:
    def test_defaults(self):
        payload = ValidationPostCheckPayload()
        assert payload.requirements == []
        assert payload.results == []
        assert payload.all_validations_passed is False
        assert payload.passed_count == 0
        assert payload.failed_count == 0

    def test_construction_with_values(self):
        reqs = [_SENTINEL_REQUIREMENT, _SENTINEL_REQUIREMENT]
        results = [_SENTINEL_VALIDATION_RESULT, _SENTINEL_VALIDATION_RESULT]
        payload = ValidationPostCheckPayload(
            requirements=reqs,
            results=results,
            all_validations_passed=True,
            passed_count=2,
            failed_count=0,
            request_id="r-val-post-001",
        )
        assert len(payload.requirements) == 2
        assert len(payload.results) == 2
        assert payload.all_validations_passed is True
        assert payload.passed_count == 2
        assert payload.failed_count == 0

    def test_partial_pass(self):
        payload = ValidationPostCheckPayload(
            passed_count=1, failed_count=1, all_validations_passed=False
        )
        assert payload.all_validations_passed is False
        assert payload.passed_count == 1
        assert payload.failed_count == 1

    def test_frozen(self):
        payload = ValidationPostCheckPayload(all_validations_passed=False)
        with pytest.raises(ValidationError):
            payload.all_validations_passed = True

    def test_frozen_counts(self):
        payload = ValidationPostCheckPayload(passed_count=2, failed_count=0)
        with pytest.raises(ValidationError):
            payload.passed_count = 3

    def test_model_copy_creates_modified_copy(self):
        payload = ValidationPostCheckPayload(
            passed_count=1, failed_count=2, all_validations_passed=False
        )
        modified = payload.model_copy(
            update={
                "passed_count": 3,
                "failed_count": 0,
                "all_validations_passed": True,
            }
        )
        assert modified.passed_count == 3
        assert modified.failed_count == 0
        assert modified.all_validations_passed is True
        assert payload.passed_count == 1
        assert payload.failed_count == 2
        assert payload.all_validations_passed is False

    def test_inherits_base_fields(self):
        assert issubclass(ValidationPostCheckPayload, MelleaBasePayload)


# ===========================================================================
# Sampling payloads
# ===========================================================================


class TestSamplingLoopStartPayload:
    def test_defaults(self):
        payload = SamplingLoopStartPayload()
        assert payload.strategy_name == ""
        assert payload.action is None
        assert payload.context is None
        assert payload.requirements == []
        assert payload.loop_budget == 0

    def test_construction_with_values(self):
        reqs = [_SENTINEL_REQUIREMENT]
        payload = SamplingLoopStartPayload(
            strategy_name="BestOfN",
            action=_SENTINEL_ACTION,
            context=_SENTINEL_CONTEXT,
            requirements=reqs,
            loop_budget=5,
            request_id="r-samp-001",
        )
        assert payload.strategy_name == "BestOfN"
        assert payload.action is _SENTINEL_ACTION
        assert payload.context is _SENTINEL_CONTEXT
        assert len(payload.requirements) == 1
        assert payload.loop_budget == 5

    def test_frozen(self):
        payload = SamplingLoopStartPayload(strategy_name="BestOfN", loop_budget=5)
        with pytest.raises(ValidationError):
            payload.loop_budget = 10

    def test_frozen_strategy_name(self):
        payload = SamplingLoopStartPayload(strategy_name="BestOfN")
        with pytest.raises(ValidationError):
            payload.strategy_name = "Retry"

    def test_model_copy_creates_modified_copy(self):
        payload = SamplingLoopStartPayload(
            strategy_name="BestOfN", loop_budget=3, context=_SENTINEL_CONTEXT
        )
        modified = payload.model_copy(update={"loop_budget": 10})
        assert modified.loop_budget == 10
        assert payload.loop_budget == 3
        assert modified.strategy_name == "BestOfN"
        assert modified.context is _SENTINEL_CONTEXT

    def test_inherits_base_fields(self):
        assert issubclass(SamplingLoopStartPayload, MelleaBasePayload)


class TestSamplingIterationPayload:
    def test_defaults(self):
        payload = SamplingIterationPayload()
        assert payload.iteration == 0
        assert payload.action is None
        assert payload.result is None
        assert payload.validation_results == []
        assert payload.all_validations_passed is False
        assert payload.valid_count == 0
        assert payload.total_count == 0

    def test_construction_with_values(self):
        val_results = [
            (_SENTINEL_REQUIREMENT, _SENTINEL_VALIDATION_RESULT),
            (_SENTINEL_REQUIREMENT, _SENTINEL_VALIDATION_RESULT),
        ]
        payload = SamplingIterationPayload(
            iteration=2,
            action=_SENTINEL_ACTION,
            result=_SENTINEL_RESULT,
            validation_results=val_results,
            all_validations_passed=True,
            valid_count=2,
            total_count=2,
            request_id="r-iter-001",
        )
        assert payload.iteration == 2
        assert payload.action is _SENTINEL_ACTION
        assert payload.result is _SENTINEL_RESULT
        assert len(payload.validation_results) == 2
        assert payload.all_validations_passed is True
        assert payload.valid_count == 2
        assert payload.total_count == 2

    def test_partial_validity(self):
        val_results = [(_SENTINEL_REQUIREMENT, _SENTINEL_VALIDATION_RESULT)]
        payload = SamplingIterationPayload(
            iteration=1,
            validation_results=val_results,
            all_validations_passed=False,
            valid_count=0,
            total_count=1,
        )
        assert payload.all_validations_passed is False
        assert payload.valid_count == 0

    def test_frozen(self):
        payload = SamplingIterationPayload(iteration=1)
        with pytest.raises(ValidationError):
            payload.iteration = 2

    def test_frozen_all_validations_passed(self):
        payload = SamplingIterationPayload(all_validations_passed=False)
        with pytest.raises(ValidationError):
            payload.all_validations_passed = True

    def test_model_copy_creates_modified_copy(self):
        payload = SamplingIterationPayload(
            iteration=0, valid_count=0, total_count=2, all_validations_passed=False
        )
        modified = payload.model_copy(
            update={"iteration": 1, "valid_count": 2, "all_validations_passed": True}
        )
        assert modified.iteration == 1
        assert modified.valid_count == 2
        assert modified.all_validations_passed is True
        assert payload.iteration == 0
        assert payload.valid_count == 0
        assert payload.all_validations_passed is False
        assert modified.total_count == 2

    def test_inherits_base_fields(self):
        assert issubclass(SamplingIterationPayload, MelleaBasePayload)


class TestSamplingRepairPayload:
    def test_defaults(self):
        payload = SamplingRepairPayload()
        assert payload.repair_type == ""
        assert payload.failed_action is None
        assert payload.failed_result is None
        assert payload.failed_validations == []
        assert payload.repair_action is None
        assert payload.repair_context is None
        assert payload.repair_iteration == 0

    def test_construction_with_values(self):
        failed_vals = [(_SENTINEL_REQUIREMENT, _SENTINEL_VALIDATION_RESULT)]
        payload = SamplingRepairPayload(
            repair_type="prompt_repair",
            failed_action=_SENTINEL_ACTION,
            failed_result=_SENTINEL_RESULT,
            failed_validations=failed_vals,
            repair_action=_SENTINEL_ACTION,
            repair_context=_SENTINEL_CONTEXT,
            repair_iteration=1,
            request_id="r-repair-001",
        )
        assert payload.repair_type == "prompt_repair"
        assert payload.failed_action is _SENTINEL_ACTION
        assert payload.failed_result is _SENTINEL_RESULT
        assert len(payload.failed_validations) == 1
        assert payload.repair_action is _SENTINEL_ACTION
        assert payload.repair_context is _SENTINEL_CONTEXT
        assert payload.repair_iteration == 1

    def test_frozen(self):
        payload = SamplingRepairPayload(repair_type="prompt_repair")
        with pytest.raises(ValidationError):
            payload.repair_type = "context_repair"

    def test_frozen_repair_iteration(self):
        payload = SamplingRepairPayload(repair_iteration=1)
        with pytest.raises(ValidationError):
            payload.repair_iteration = 2

    def test_model_copy_creates_modified_copy(self):
        failed_vals = [(_SENTINEL_REQUIREMENT, _SENTINEL_VALIDATION_RESULT)]
        payload = SamplingRepairPayload(
            repair_type="prompt_repair",
            repair_iteration=0,
            failed_validations=failed_vals,
        )
        new_ctx = object()
        modified = payload.model_copy(
            update={"repair_iteration": 2, "repair_context": new_ctx}
        )
        assert modified.repair_iteration == 2
        assert modified.repair_context is new_ctx
        assert payload.repair_iteration == 0
        assert payload.repair_context is None
        assert modified.repair_type == "prompt_repair"

    def test_inherits_base_fields(self):
        assert issubclass(SamplingRepairPayload, MelleaBasePayload)
        payload = SamplingRepairPayload(user_metadata={"repair": True})
        assert payload.user_metadata == {"repair": True}


class TestSamplingLoopEndPayload:
    def test_defaults(self):
        payload = SamplingLoopEndPayload()
        assert payload.success is False
        assert payload.iterations_used == 0
        assert payload.final_result is None
        assert payload.final_action is None
        assert payload.final_context is None
        assert payload.failure_reason is None
        assert payload.all_results == []
        assert payload.all_validations == []

    def test_construction_successful(self):
        all_results = [_SENTINEL_RESULT, _SENTINEL_RESULT]
        all_validations = [
            [(_SENTINEL_REQUIREMENT, _SENTINEL_VALIDATION_RESULT)],
            [(_SENTINEL_REQUIREMENT, _SENTINEL_VALIDATION_RESULT)],
        ]
        payload = SamplingLoopEndPayload(
            success=True,
            iterations_used=2,
            final_result=_SENTINEL_RESULT,
            final_action=_SENTINEL_ACTION,
            final_context=_SENTINEL_CONTEXT,
            failure_reason=None,
            all_results=all_results,
            all_validations=all_validations,
            request_id="r-end-001",
        )
        assert payload.success is True
        assert payload.iterations_used == 2
        assert payload.final_result is _SENTINEL_RESULT
        assert payload.failure_reason is None
        assert len(payload.all_results) == 2
        assert len(payload.all_validations) == 2

    def test_construction_failed(self):
        payload = SamplingLoopEndPayload(
            success=False, iterations_used=5, failure_reason="max_iterations_exceeded"
        )
        assert payload.success is False
        assert payload.failure_reason == "max_iterations_exceeded"
        assert payload.final_result is None

    def test_frozen(self):
        payload = SamplingLoopEndPayload(success=False)
        with pytest.raises(ValidationError):
            payload.success = True

    def test_frozen_iterations(self):
        payload = SamplingLoopEndPayload(iterations_used=3)
        with pytest.raises(ValidationError):
            payload.iterations_used = 5

    def test_frozen_failure_reason(self):
        payload = SamplingLoopEndPayload(failure_reason="budget_exhausted")
        with pytest.raises(ValidationError):
            payload.failure_reason = None

    def test_model_copy_creates_modified_copy(self):
        payload = SamplingLoopEndPayload(
            success=False, iterations_used=5, failure_reason="budget_exhausted"
        )
        modified = payload.model_copy(
            update={
                "success": True,
                "iterations_used": 3,
                "failure_reason": None,
                "final_result": _SENTINEL_RESULT,
            }
        )
        assert modified.success is True
        assert modified.iterations_used == 3
        assert modified.failure_reason is None
        assert modified.final_result is _SENTINEL_RESULT
        assert payload.success is False
        assert payload.iterations_used == 5
        assert payload.failure_reason == "budget_exhausted"

    def test_failure_reason_none_vs_string(self):
        payload_none = SamplingLoopEndPayload(failure_reason=None)
        payload_str = SamplingLoopEndPayload(failure_reason="budget_exceeded")
        assert payload_none.failure_reason is None
        assert payload_str.failure_reason == "budget_exceeded"

    def test_inherits_base_fields(self):
        assert issubclass(SamplingLoopEndPayload, MelleaBasePayload)


# ===========================================================================
# Tool payloads
# ===========================================================================


class TestToolPreInvokePayload:
    def test_defaults(self):
        payload = ToolPreInvokePayload()
        assert payload.model_tool_call is None

    def test_construction_with_values(self):
        payload = ToolPreInvokePayload(
            model_tool_call=_SENTINEL_TOOL_CALL,
            request_id="r-tool-001",
            hook="tool_pre_invoke",
        )
        assert payload.model_tool_call is _SENTINEL_TOOL_CALL
        assert payload.hook == "tool_pre_invoke"

    def test_frozen(self):
        payload = ToolPreInvokePayload(model_tool_call=_SENTINEL_TOOL_CALL)
        with pytest.raises(ValidationError):
            payload.model_tool_call = None

    def test_model_copy_creates_modified_copy(self):
        payload = ToolPreInvokePayload(model_tool_call=_SENTINEL_TOOL_CALL)
        replacement = object()
        modified = payload.model_copy(update={"model_tool_call": replacement})
        assert modified.model_tool_call is replacement
        assert payload.model_tool_call is _SENTINEL_TOOL_CALL

    def test_inherits_base_fields(self):
        assert issubclass(ToolPreInvokePayload, MelleaBasePayload)
        payload = ToolPreInvokePayload(
            session_id="s-tool", user_metadata={"source": "unit_test"}
        )
        assert payload.session_id == "s-tool"
        assert payload.user_metadata == {"source": "unit_test"}


class TestToolPostInvokePayload:
    def test_defaults(self):
        payload = ToolPostInvokePayload()
        assert payload.model_tool_call is None
        assert payload.tool_output is None
        assert payload.tool_message is None
        assert payload.execution_time_ms == 0
        assert payload.success is True
        assert payload.error is None

    def test_construction_successful(self):
        payload = ToolPostInvokePayload(
            model_tool_call=_SENTINEL_TOOL_CALL,
            tool_output={"results": ["Paris is the capital of France"]},
            tool_message=_SENTINEL_TOOL_MESSAGE,
            execution_time_ms=87,
            success=True,
            error=None,
            request_id="r-tool-post-001",
        )
        assert payload.model_tool_call is _SENTINEL_TOOL_CALL
        assert payload.tool_output == {"results": ["Paris is the capital of France"]}
        assert payload.tool_message is _SENTINEL_TOOL_MESSAGE
        assert payload.execution_time_ms == 87
        assert payload.success is True
        assert payload.error is None

    def test_construction_failed(self):
        payload = ToolPostInvokePayload(
            model_tool_call=_SENTINEL_TOOL_CALL,
            success=False,
            error=_SENTINEL_ERROR,
            execution_time_ms=12,
        )
        assert payload.success is False
        assert payload.error is _SENTINEL_ERROR
        assert payload.tool_output is None

    def test_frozen(self):
        payload = ToolPostInvokePayload(success=True)
        with pytest.raises(ValidationError):
            payload.success = False

    def test_frozen_model_tool_call(self):
        payload = ToolPostInvokePayload(model_tool_call=_SENTINEL_TOOL_CALL)
        with pytest.raises(ValidationError):
            payload.model_tool_call = None

    def test_frozen_execution_time(self):
        payload = ToolPostInvokePayload(execution_time_ms=50)
        with pytest.raises(ValidationError):
            payload.execution_time_ms = 100

    def test_model_copy_creates_modified_copy(self):
        payload = ToolPostInvokePayload(
            model_tool_call=_SENTINEL_TOOL_CALL,
            success=False,
            error=_SENTINEL_ERROR,
            execution_time_ms=10,
        )
        modified = payload.model_copy(
            update={"success": True, "error": None, "execution_time_ms": 75}
        )
        assert modified.success is True
        assert modified.error is None
        assert modified.execution_time_ms == 75
        assert payload.success is False
        assert payload.error is _SENTINEL_ERROR
        assert payload.execution_time_ms == 10
        assert modified.model_tool_call is _SENTINEL_TOOL_CALL

    def test_tool_output_various_types(self):
        payload_str = ToolPostInvokePayload(tool_output="plain string result")
        payload_dict = ToolPostInvokePayload(tool_output={"key": "value"})
        payload_list = ToolPostInvokePayload(tool_output=[1, 2, 3])
        assert payload_str.tool_output == "plain string result"
        assert payload_dict.tool_output == {"key": "value"}
        assert payload_list.tool_output == [1, 2, 3]

    def test_inherits_base_fields(self):
        assert issubclass(ToolPostInvokePayload, MelleaBasePayload)
        payload = ToolPostInvokePayload(
            session_id="s-tool-post",
            hook="tool_post_invoke",
            user_metadata={"retry": False},
        )
        assert payload.session_id == "s-tool-post"
        assert payload.hook == "tool_post_invoke"
        assert payload.user_metadata == {"retry": False}
