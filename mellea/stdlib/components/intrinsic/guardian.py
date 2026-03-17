"""Intrinsic functions for the Guardian component."""

from ....backends.adapters import AdapterMixin
from ...components import Message
from ...context import ChatContext
from ._util import call_intrinsic


def factuality_detection(context: ChatContext, backend: AdapterMixin) -> float:
    """Determine is the last response is factually incorrect.

    Intrinsic function that evaluates the factuality of the
    assistant's response to a user's question. The context should end with
    a user question followed by an assistant answer.

    :param context: Chat context containing user question and assistant answer.
    :param backend: Backend instance that supports LoRA/aLoRA adapters.

    :return: Factuality score as a "yes/no" label (yes = factually incorrect).
    """

    detector_message = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'.
"""

    context = context.add(Message("user", detector_message))
    result_json = call_intrinsic("factuality_detection", context, backend)
    return result_json["score"]

def factuality_correction(context: ChatContext, backend: AdapterMixin) -> float:
    """Corrects the last response so that it is factually correct relative
    to the given contextual information.

    Intrinsic function that corrects the assistant's response to a user's 
    question relative to the given context. 

    :param context: Chat context containing user question and assistant answer.
    :param backend: Backend instance that supports LoRA/aLoRA adapters.

    :return: Correct assistant response.
    """

    corrector_message = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return a corrected version of the assistant's message based on the given context; otherwise, return 'none'.
"""

    context = context.add(Message("user", corrector_message))
    result_json = call_intrinsic("factuality_correction", context, backend)
    return result_json["correction"]
