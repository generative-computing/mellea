# Installing langchain is necessary for this example, but it works for any library
# you may want to use Mellea with.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Messages from a different library.
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there!"),
]

# Some libraries have conversion functions that make it easier to ingest into Mellea.
from langchain_core.messages import convert_to_openai_messages

messages = convert_to_openai_messages(messages=messages)

# Import Mellea.
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.backends import ModelOption
from mellea import start_session

# Mellea uses explicit contexts. Cast the OpenAI formatted messages into
# Mellea messages and add them to the context.
ctx = ChatContext()
for msg in messages:
    ctx = ctx.add(
        # NOTE: If your messages aren't in OpenAI format or have additional data like
        #       documents / images, you need to explicitly grab those fields as well.
        Message(role=msg["role"], content=msg["content"])
    )

# Utilize that new ChatContext to ask the assistant about its past messages.
m = start_session(ctx=ctx)
response = m.chat(
    "What was the last assistant message?",
    model_options={
        ModelOption.SEED: 2
    },  # Utilizing a seed for consistency in the example.
).content

assert "Hi there!" in response
