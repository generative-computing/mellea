"""Intrinsic functions for core model capabilities."""

from ....backends.adapters import AdapterMixin
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
