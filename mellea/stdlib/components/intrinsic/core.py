"""Intrinsic functions for core model capabilities."""

from ....backends.adapters import AdapterMixin
from ...components import Message
from ...context import ChatContext
from ._util import call_intrinsic


def check_certainty(context: ChatContext, backend: AdapterMixin) -> float:
    """Estimate the model's certainty about its last response.

    Intrinsic function that evaluates how certain the model is about the
    assistant's response to a user's question. The context should end with
    a user question followed by an assistant answer.

    :param context: Chat context containing user question and assistant answer.
    :param backend: Backend instance that supports LoRA/aLoRA adapters.

    :return: Certainty score as a float (higher = more certain).
    """
    result_json = call_intrinsic("uncertainty", context, backend)
    return result_json["certainty"]


_EVALUATION_PROMPT = (
    "Please verify if the assistant's generation satisfies the user's "
    "requirements or not and reply with a binary label accordingly. "
    'Respond with a json {"score": "yes"} if the constraints are '
    'satisfied or respond with {"score": "no"} if the constraints are not '
    "satisfied."
)


def requirement_check(
    context: ChatContext, backend: AdapterMixin, requirement: str
) -> float:
    """Detect if text adheres to provided requirements.

    Intrinsic function that determines if the text satisfies the given
    requirements. Appends an evaluation prompt to the context following
    the format specified by the Granite Guardian requirement checker model card.

    :param context: Chat context containing user question and assistant answer.
    :param backend: Backend instance that supports LoRA/aLoRA adapters.
    :param requirement: set of requirements to satisfy

    :return: Score as a float between 0.0 and 1.0 (higher = more likely satisfied).
    """
    eval_message = f"<requirements>: {requirement}\n{_EVALUATION_PROMPT}"
    context = context.add(Message("user", eval_message))
    result_json = call_intrinsic("requirement_check", context, backend)
    return result_json["requirement_check"]["score"]
