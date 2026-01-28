# Mellea Examples

This directory contains comprehensive examples demonstrating Mellea's features and capabilities. Examples are organized by topic and complexity level.

## 🚀 Getting Started

**New to Mellea?** Start here:
1. [tutorial/simple_email.py](tutorial/) - Your first Mellea program
2. [instruct_validate_repair/](instruct_validate_repair/) - Core paradigm
3. [generative_slots/](generative_slots/) - Type-safe LLM functions
4. [notebooks/](notebooks/) - Interactive tutorials

## 📚 Example Categories

### Core Concepts

**[instruct_validate_repair/](instruct_validate_repair/)**
Learn Mellea's core instruct-validate-repair paradigm for reliable LLM outputs.
- Basic instruction without requirements
- Adding validation constraints
- Automatic repair on validation failure
- Custom validation functions

**[generative_slots/](generative_slots/)**
Type-safe, composable LLM functions using the `@generative` decorator.
- Sentiment classification
- Text summarization
- Function composition
- Type-constrained outputs

**[context/](context/)**
Understanding and working with Mellea's context system.
- Context inspection
- Sampling with contexts
- Context trees and navigation

**[sessions/](sessions/)**
Creating and customizing Mellea sessions.
- Session configuration
- Custom session types
- Backend selection

### Advanced Features

**[aLora/](aLora/)**
Adaptive Low-Rank Adaptation for fast constraint checking.
- Training custom aLoRA adapters
- Performance optimization
- Constraint validation speedup

**[intrinsics/](intrinsics/)**
Specialized model capabilities through adapters.
- Answer relevance checking
- Hallucination detection
- Citation validation
- Context relevance assessment

**[sofai/](sofai/)**
Two-tier sampling with fast and slow models.
- Cost optimization
- Iterative refinement with fast models
- Escalation to slow models
- Constraint satisfaction problems

### Data & Documents

**[information_extraction/](information_extraction/)**
Extracting structured information from unstructured text.
- Named entity recognition
- Type-safe extraction
- Structured output generation

**[mobject/](mobject/)**
Working with structured data types (tables, documents).
- Table queries and transformations
- Document processing
- Structured data operations

**[mify/](mify/)**
Making custom Python objects work with LLMs.
- Object integration with `@mify`
- Custom string representations
- Template integration
- Tool generation from methods

**[rag/](rag/)**
Retrieval-Augmented Generation pipelines.
- Vector search with FAISS
- Relevance filtering
- Grounded answer generation
- Multi-stage RAG pipelines

### Agents & Tools

**[agents/](agents/)**
Implementing agent patterns (ReACT).
- Reasoning and acting loops
- Tool selection and execution
- Multi-turn agent workflows

**[tools/](tools/)**
Tool calling and code execution.
- Code interpreter integration
- Custom tool definition
- Tool argument validation
- Safe code execution

### Safety & Validation

**[safety/](safety/)**
Content safety with Granite Guardian models.
- Harm detection
- Jailbreak prevention
- Bias checking
- Groundedness validation
- Function call hallucination detection

### Integration & Deployment

**[m_serve/](m_serve/)**
Deploying Mellea programs as REST APIs.
- API service creation
- Production deployment patterns
- Client integration

**[library_interop/](library_interop/)**
Integrating with other LLM libraries.
- LangChain message conversion
- OpenAI format compatibility
- Cross-library workflows

**[mcp/](mcp/)**
Model Context Protocol integration.
- MCP tool creation
- Claude Desktop integration
- Langflow integration

### Multimodal

**[image_text_models/](image_text_models/)**
Working with vision-language models.
- Image understanding
- Multimodal prompting
- Vision model backends

### Complete Applications

**[mini_researcher/](mini_researcher/)**
Full-featured research assistant with RAG and validation.
- Multi-model architecture
- Document retrieval
- Safety checks
- Custom validation pipeline

### Interactive Learning

**[notebooks/](notebooks/)**
Jupyter notebooks for interactive exploration.
- Step-by-step tutorials
- Immediate feedback
- Visualization of results

**[tutorial/](tutorial/)**
Python script versions of tutorials.
- Non-interactive examples
- Easy to run and modify
- Version control friendly

### Experimental

**[melp/](melp/)**
⚠️ Experimental lazy evaluation system.
- Lazy computation
- Thunks and deferred execution
- Advanced control flow

### Utilities

**[helper/](helper/)**
Utility functions used across examples.
- Text formatting helpers
- Common utilities

## 🎯 Examples by Use Case

### Text Generation
- [instruct_validate_repair/](instruct_validate_repair/) - Email generation
- [generative_slots/](generative_slots/) - Summarization
- [tutorial/sentiment_classifier.py](tutorial/) - Classification

### Data Processing
- [information_extraction/](information_extraction/) - Entity extraction
- [mobject/](mobject/) - Table operations
- [rag/](rag/) - Document retrieval

### Agent Systems
- [agents/](agents/) - ReACT agents
- [tools/](tools/) - Tool-using agents
- [mini_researcher/](mini_researcher/) - Research assistant

### Production Deployment
- [m_serve/](m_serve/) - API services
- [safety/](safety/) - Content moderation
- [library_interop/](library_interop/) - Integration

### Performance Optimization
- [aLora/](aLora/) - Fast validation
- [sofai/](sofai/) - Cost optimization
- [intrinsics/](intrinsics/) - Specialized tasks

## 📖 Documentation

- **Main README**: [../../README.md](../../README.md)
- **Agent Guidelines**: [../../AGENTS.md](../../AGENTS.md)
- **Dev Docs**: [../dev/](../dev/)

## 🏃 Running Examples

```bash
# Run any Python example
python docs/examples/tutorial/simple_email.py

# Or with uv
uv run docs/examples/tutorial/simple_email.py

# Run notebooks
jupyter notebook docs/examples/notebooks/

# Run tests
uv run pytest test/
```

## 💡 Tips

- Start with [tutorial/](tutorial/) for basics
- Check [notebooks/](notebooks/) for interactive learning
- See [mini_researcher/](mini_researcher/) for complete application patterns
- Refer to individual README.md files in each directory for details

## 🤝 Contributing

Found a bug or have an improvement? See [../../AGENTS.md](../../AGENTS.md) for contribution guidelines.