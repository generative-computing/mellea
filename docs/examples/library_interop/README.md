# Library Interoperability Examples

Examples demonstrating how to bring data from other LLM libraries into Mellea.
For the full LangChain and smolagents integrations, including tool wrapping, see
[LangChain](../../docs/integrations/langchain.md) and
[smolagents](../../docs/integrations/smolagents.md).

## Examples

| File | Description |
| :--- | :--- |
| [`langchain_messages.py`](./langchain_messages.py) | Converting LangChain messages to OpenAI format, building a Mellea `ChatContext` from them, and continuing the conversation in a session. Needs `uv sync --extra tools`. |
