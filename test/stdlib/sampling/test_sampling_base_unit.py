"""Unit tests for sampling/base.py static repair() logic — no backend required."""

import pytest

from mellea.core import (
    ComputedModelOutputThunk,
    ModelOutputThunk,
    Requirement,
    ValidationResult,
)
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling.base import RepairTemplateStrategy

# --- BaseSamplingStrategy.repair ---


def _val(passed: bool, reason: str | None = None) -> ValidationResult:
    return ValidationResult(result=passed, reason=reason)


def test_repair_instruction_builds_repair_string():
    ins = Instruction(description="Write a poem", requirements=["be concise"])
    req = Requirement(description="be concise")
    old_ctx = ChatContext()
    new_ctx = ChatContext()

    action, ctx = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=new_ctx,
        past_actions=[ins],
        past_results=[
            ComputedModelOutputThunk(thunk=ModelOutputThunk(value="long text"))
        ],
        past_val=[[(req, _val(False, reason="Output was too long"))]],
    )
    assert isinstance(action, Instruction)
    assert action._repair_string is not None
    assert "Output was too long" in action._repair_string
    assert ctx is old_ctx


def test_repair_uses_req_description_when_no_reason():
    ins = Instruction(description="task")
    req = Requirement(description="must be brief")
    old_ctx = ChatContext()

    action, _ = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[ins],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[(req, _val(False))]],
    )
    assert "must be brief" in action._repair_string


def test_repair_non_instruction_returns_same_action():
    msg = Message("user", "hello")
    old_ctx = ChatContext()

    action, ctx = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[msg],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[]],
    )
    assert action is msg
    assert ctx is old_ctx


def test_repair_multiple_failures_all_listed():
    ins = Instruction(description="task")
    r1 = Requirement(description="be short")
    r2 = Requirement(description="be polite")
    old_ctx = ChatContext()

    action, _ = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[ins],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[(r1, _val(False, "too long")), (r2, _val(False, "rude tone"))]],
    )
    assert "too long" in action._repair_string
    assert "rude tone" in action._repair_string


def test_repair_passed_requirements_excluded():
    ins = Instruction(description="task")
    r_pass = Requirement(description="format ok")
    r_fail = Requirement(description="content wrong")
    old_ctx = ChatContext()

    action, _ = RepairTemplateStrategy.repair(
        old_ctx=old_ctx,
        new_ctx=ChatContext(),
        past_actions=[ins],
        past_results=[ComputedModelOutputThunk(thunk=ModelOutputThunk(value="x"))],
        past_val=[[(r_pass, _val(True)), (r_fail, _val(False, "incorrect"))]],
    )
    assert "format ok" not in action._repair_string
    assert "incorrect" in action._repair_string


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
