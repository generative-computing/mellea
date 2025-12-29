# Mellea-Native KG-RAG Implementation

This document explains the Mellea-native implementation of KG-RAG and how it showcases Mellea's core patterns.

## Overview

The KG-RAG example now includes **two parallel implementations**:

1. **Traditional** (`run/run_qa.py`) - Direct LLM API calls with manual validation
2. **Mellea-Native** (`run/run_qa_mellea.py`) - Uses Mellea's @generative, Requirements, and Components

Both produce the same results, but the Mellea-native version demonstrates best practices for building robust, composable LLM applications.

## Key Benefits

✅ **Type Safety** - Pydantic models ensure valid outputs  
✅ **Robustness** - Automatic validation and retry logic  
✅ **Composability** - Reusable functions and components  
✅ **Maintainability** - Self-documenting code  
✅ **Testability** - Easy to test individual pieces  

## Quick Start

```bash
# Run Mellea-native implementation
cd docs/examples/kgrag
uv run --with mellea run/run_qa_mellea.py --num-workers 1 --prefix mellea

# Compare with traditional
uv run --with mellea run/run_qa.py --num-workers 1 --prefix traditional
```

## Architecture

See full documentation for detailed architecture comparison and migration guide.
