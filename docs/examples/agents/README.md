# Agent Examples

Examples demonstrating agent architectures and the ReACT (Reasoning and Acting) loop in Mellea.

Start with `react_using_mellea.py` for the shortest path; the `react_from_scratch/`
examples build the same loop by hand to show what the framework does for you.

## Examples

| File | Description |
| :--- | :--- |
| [`react/react_using_mellea.py`](./react/react_using_mellea.py) | ReACT agent using Mellea's stdlib `react()` framework with LangChain tool integration. Needs `langchain-community` and `ddgs`. |
| [`react/react_from_scratch/react.py`](./react/react_from_scratch/react.py) | A complete implementation of the ReACT agent pattern that combines reasoning and action in an iterative loop. |
| [`react/react_from_scratch/react_instruct.py`](./react/react_from_scratch/react_instruct.py) | An alternative implementation of the ReACT pattern using Mellea's instruct-validate-repair paradigm. |
