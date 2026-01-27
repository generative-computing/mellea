from collections.abc import Callable
from typing import Any
from mellea.backends.adapters.adapter import AdapterMixin, GraniteCommonAdapter
from mellea.backends.adapters.catalog import AdapterType
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_options import ModelOption
from mellea.core.backend import Backend, BaseModelSubclass
from mellea.core.base import CBlock, Context, ModelOutputThunk
from mellea.core.requirement import (
    Requirement,
    ValidationResult,
    default_output_to_bool,
)
from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.requirement import ALoraRequirement

# Combine generation with citations.

# answer relevance requirement...
# instruction -> response -> ensure response is relevant to instruction
# shove the response into check_context_relevance as a document

# TODO: JAL. Should this be an alora? or some other "LORA" esque requirement instead and rely on built in behavior?
class AnswerRelevanceCheck(Requirement):
    def __init__(
        self,
    ):
        super().__init__(
            None,
            None,
            check_only=True,
        )

    async def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ValidationResult:
        assert isinstance(backend, AdapterMixin), (
            "must use an adapter compatible backend"
        )

        # Context should be: [query, response]
        qna = ctx.last_turn()
        assert qna is not None

        query = qna.model_input
        answer = qna.output

        # Special handling if the query is an instruction.
        if isinstance(query, Instruction):
            query_str = str(query._description)
        else:
            query_str = str(query)

        answer_doc = Document(str(answer), doc_id="1")

        # TODO: JAL. if we can't just make this an alora requirement because of the context mangling, then there's a bit of a weirdness in how our code is structured...
        relevance = rag.check_context_relevance(
            query_str, answer_doc, ChatContext(), backend
        )

        return ValidationResult(
            result=relevance > 0.8, reason="relevance", score=relevance
        )


if __name__ == "__main__":
    query = "What color is the sky?"
    backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

    response = mfuncs.instruct(
        query,
        context=ChatContext(),
        backend=backend,
        requirements=[AnswerRelevanceCheck()],
        return_sampling_results=True,
    )

    print(response)
