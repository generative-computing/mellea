# Library Interoperability Examples

This directory demonstrates how to integrate Mellea with other LLM libraries and
frameworks. For the full LangChain and smolagents integrations, including tool
wrapping, see
[LangChain](../../docs/integrations/langchain.md) and
[smolagents](../../docs/integrations/smolagents.md).

## Examples

| File | Description |
| :--- | :--- |
| [`langchain_messages.py`](./langchain_messages.py) | Shows how to convert LangChain messages to Mellea format and use them in a session. Needs `uv sync --extra tools`. |

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
- See `mellea/stdlib/context/` for context management
