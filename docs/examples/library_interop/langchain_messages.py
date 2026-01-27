# LangChain is required for this example. Install it with:
#   pip install "mellea[langchain]"
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Messages from LangChain (or any LangChain-compatible library).
lc_messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there! We're going to talk to Mellea!"),
]

# Use Mellea's chat_converters to convert LangChain messages directly.
from mellea.stdlib.components.chat_converters import langchain_messages_to_mellea

mellea_messages = langchain_messages_to_mellea(lc_messages)

# Import Mellea session utilities.
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.context import ChatContext

# Add the converted messages to a ChatContext.
ctx = ChatContext()
for msg in mellea_messages:
    ctx = ctx.add(msg)

# Utilize that new ChatContext to ask the assistant about its past messages.
m = start_session(ctx=ctx)
response = m.chat(
    "What was the last assistant message?",
    model_options={
        ModelOption.SEED: 2
    },  # Utilizing a seed for consistency in the example.
).content

assert "Hi there! We're going to talk to Mellea!" in response
