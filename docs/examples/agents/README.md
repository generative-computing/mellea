# Agent Examples

This directory contains examples of implementing agent patterns with Mellea, specifically the ReACT (Reasoning and Acting) pattern.

## Files

### react.py
A complete implementation of the ReACT agent pattern that combines reasoning and action in an iterative loop. The agent:
- Thinks about what to do next
- Selects an appropriate tool/action
- Generates arguments for the tool
- Observes the result
- Determines if the goal is achieved

**Key Features:**
- Custom `ReactTool` and `ReactToolbox` classes for tool management
- Dynamic tool selection using Pydantic schemas
- Iterative thought-action-observation loop
- Example with weather lookup tools

**Usage:**
```python
python docs/examples/agents/react.py
```

### react_instruct.py
An alternative implementation of the ReACT pattern using Mellea's instruct-validate-repair paradigm.

## Concepts Demonstrated

- **Tool Management**: Creating and organizing tools for agent use
- **Dynamic Prompting**: Building system prompts with tool descriptions
- **Chat Context**: Using `ChatContext` for multi-turn conversations
- **Structured Output**: Using Pydantic models for type-safe responses
- **Iterative Reasoning**: Implementing thought-action-observation loops

## Related Documentation

- See `docs/dev/tool_calling.md` for more on tool integration
- See `mellea/stdlib/requirements/tool_reqs.py` for tool requirements