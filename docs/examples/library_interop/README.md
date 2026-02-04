# Library Interoperability Examples

This directory demonstrates how to integrate Mellea with other LLM libraries and frameworks.

## Files

### langchain_messages.py
Shows how to convert LangChain messages to Mellea format and use them in a session.

**Key Features:**
- Converting LangChain messages to Mellea messages
- Using OpenAI-formatted messages as an intermediate format
- Building Mellea contexts from external message formats
- Maintaining conversation history across libraries

## Concepts Demonstrated

- **Message Conversion**: Translating between different message formats
- **Context Building**: Creating Mellea contexts from external data
- **Library Integration**: Using Mellea alongside other frameworks
- **Format Compatibility**: Working with OpenAI-compatible message formats

## Basic Pattern

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import convert_to_openai_messages
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea import start_session

# 1. Get messages from another library
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there!")
]

# 2. Convert to OpenAI format (if needed)
messages = convert_to_openai_messages(messages=messages)

# 3. Build Mellea context
ctx = ChatContext()
for msg in messages:
    ctx = ctx.add(Message(role=msg["role"], content=msg["content"]))

# 4. Use in Mellea session
m = start_session(ctx=ctx)
response = m.chat("What was the last assistant message?")
```

## Integration Tips

- **OpenAI Format**: Many libraries support OpenAI message format as a common interchange
- **Explicit Contexts**: Mellea uses explicit context objects - build them from external messages
- **Additional Data**: If messages contain images, documents, or other data, extract those fields explicitly
- **Bidirectional**: You can also convert Mellea messages back to other formats

## Supported Libraries

This pattern works with any library that can export to OpenAI-compatible format:
- LangChain
- LlamaIndex
- Haystack
- Semantic Kernel
- And many others

## Related Documentation

- See `mellea/stdlib/components/chat.py` for Message API
- See `mellea/stdlib/context.py` for context management