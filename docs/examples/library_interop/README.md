# Library Interoperability Examples

This directory demonstrates how to integrate Mellea with other LLM libraries and
frameworks. For the full LangChain and smolagents integrations, including tool
wrapping, see [LangChain](../../docs/integrations/langchain.md) and
[smolagents](../../docs/integrations/smolagents.md).

## Examples

| File | Description |
| :--- | :--- |
| [`langchain_messages.py`](./langchain_messages.py) | Shows how to convert LangChain messages to Mellea format and use them in a session. Needs `uv sync --extra tools`. |
