from mellea import model_ids, start_backend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import rag

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)
# NOTE: This example can also be run with the OpenAIBackend using a GraniteSwitch model. See docs/examples/granite-switch/.

ctx_with_question = (
    ctx.add(Message("assistant", "Welcome to pet questions!"))
    .add(
        Message(
            "user",
            "I have two pets, a dog named Rex and a cat named Lucy. "
            "Rex spends a lot of time in the backyard and outdoors, "
            "and Lucy is always inside.",
        )
    )
    .add(
        Message(
            "assistant",
            "Sounds good! Rex must love exploring outside, while Lucy "
            "probably enjoys her cozy indoor life.",
        )
    )
    .add(Message("user", "But is he more likely to get fleas because of that?"))
)

print("Original user question: 'But is he more likely to get fleas because of that?'")

result = rag.rewrite_question(None, ctx_with_question, backend)
print(f"Rewritten user question: {result}")
