# Mellea-Native KG-RAG Implementation

This document explains the Mellea-native implementation of KG-RAG and how it showcases Mellea's core patterns.

## Overview

The KG-RAG example has been **fully migrated to Mellea patterns**. All pipeline components now use Mellea's best practices:

1. **KG Preprocessing Pipeline** (`run/run_kg_preprocess.py`):
   - Statistics tracking with Pydantic models
   - Sequential and concurrent processing modes
   - Enhanced error handling and graceful failure recovery

2. **KG Embedding Pipeline** (`run/run_kg_embed.py`):
   - Mellea session-based embedding generation
   - Supports both API-based and local embeddings
   - Type-safe configuration with Pydantic `EmbeddingConfig`

3. **KG Update Pipeline** (`run/run_kg_update.py`):
   - Uses @generative for extraction, alignment, and merging
   - Component-based architecture with `KGUpdaterComponent`
   - Requirements validation and RejectionSamplingStrategy

4. **QA Pipeline** (`run/run_qa.py`):
   - Uses Mellea's @generative, Requirements, and Components
   - `KGRagComponent` for multi-hop graph reasoning
   - Worker-local session isolation for parallel processing

5. **Evaluation Pipeline** (`run/run_eval.py`):
   - @generative-based LLM-as-judge evaluation
   - Type-safe `EvaluationResult` with Pydantic
   - Async batch processing with progress bars

All implementations follow Mellea best practices for building robust, composable LLM applications.

## Key Benefits

✅ **Type Safety** - Pydantic models ensure valid outputs  
✅ **Robustness** - Automatic validation and retry logic  
✅ **Composability** - Reusable functions and components  
✅ **Maintainability** - Self-documenting code  
✅ **Testability** - Easy to test individual pieces  

## Quick Start

```bash
cd docs/examples/kgrag

# Run preprocessing
uv run --with mellea run/run_kg_preprocess.py --domain movie --verbose

# Run KG embedding
uv run --with mellea run/run_kg_embed.py --batch-size 8192 --verbose

# Run KG update
uv run --with mellea run/run_kg_update.py --num-workers 4 --queue-size 10

# Run QA evaluation
uv run --with mellea run/run_qa.py --num-workers 4 --queue-size 10

# Run evaluation
uv run --with mellea run/run_eval.py --result-path results/_results.json --verbose
```

## Architecture

### 1. KG Preprocessing (run/run_kg_preprocess.py)

The Mellea-native preprocessing implementation showcases:

**Key Features:**
- Statistics tracking with `PreprocessingStats` dataclass
- Sequential and concurrent preprocessing modes
- Detailed summary reporting with per-domain statistics
- Enhanced error handling with graceful failure recovery
- Progress tracking with timestamps and durations
- Dry-run mode for validation before execution

**Example Usage:**
```python
# Create preprocessor
preprocessor = MovieKG_Preprocessor()

# Process with statistics tracking
stats = await preprocess_single_domain(preprocessor, idx=1, total=1)

# Print summary
print_summary([stats])
```

**Benefits:**
- ✅ Detailed statistics for monitoring and debugging
- ✅ Concurrent processing support for multiple domains
- ✅ Comprehensive error reporting per domain
- ✅ Type-safe stats with dataclasses
- ✅ Better observability into preprocessing operations

### 2. KG Embedding (run/run_kg_embed.py)

The embedding implementation showcases:

**Key Features:**
- Uses `utils/utils_mellea.py` for consistent embedding generation
- `MelleaKGEmbedder` class in `kg/kg_embedder.py` for enhanced functionality
- Embedding session testing with `test_embedding_session()`
- Type-safe configuration using Pydantic `EmbeddingConfig`
- Enhanced error handling and retry logic
- Supports both OpenAI-compatible APIs and local SentenceTransformer models

**Example Usage:**
```python
# Create embedding session
emb_session = create_embedding_session(config)

# Test the session
await test_embedding_session(emb_session, config)

# Create Mellea-native embedder
from kg.kg_embedder import MelleaKGEmbedder
embedder = MelleaKGEmbedder(emb_session, config)

# Generate embeddings
embeddings = await embedder.generate_embeddings_mellea(
    texts=entity_descriptions,
    desc="Entity embeddings"
)
```

**Benefits:**
- ✅ Consistent error handling across embedding calls
- ✅ Session validation before processing
- ✅ Better logging and progress tracking
- ✅ Type-safe configuration prevents errors

### 3. KG Update (run/run_kg_update.py)

The Mellea-native KG update implementation demonstrates:

**Key Components:**
- `kg_updater_generative.py` - @generative functions for:
  - `extract_entities_and_relations()` - Entity/relation extraction
  - `align_entity_with_kg()` - Entity alignment
  - `decide_entity_merge()` - Merge decisions
  - `align_relation_with_kg()` - Relation alignment
  - `decide_relation_merge()` - Relation merge decisions

- `kg_updater_component.py` - Component-based architecture:
  - Extends Mellea's `Component` base class
  - Uses `RejectionSamplingStrategy` for robustness
  - Integrates Requirements validation
  - Modular methods for extraction, alignment, and merging

**Example Usage:**
```python
# Create KG updater component
kg_updater = KGUpdaterComponent(
    session=session,
    emb_session=emb_session,
    kg_driver=kg_driver,
    domain="movie",
    config={
        "align_entity": True,
        "merge_entity": True,
        "extraction_loop_budget": 3,
    }
)

# Process document
stats = await kg_updater.update_kg_from_document(
    doc_id=doc_id,
    context=context,
    reference=reference,
    created_at=datetime.now()
)
```

**Benefits:**
- ✅ Automatic validation and retry with RejectionSamplingStrategy
- ✅ Type-safe Pydantic models for all outputs
- ✅ Composable architecture with Component pattern
- ✅ Clear separation of concerns

### 4. QA Pipeline (run/run_qa.py)

The Mellea-native QA implementation showcases:

**Key Components:**
- `kg_generative.py` - @generative functions for:
  - `break_down_question()` - Question decomposition
  - `extract_topic_entities()` - Topic entity extraction
  - `find_relevant_entities()` - Entity relevance scoring
  - `generate_answer()` - Final answer generation

- `kg_rag.py` - Component-based RAG:
  - Extends Mellea's `Component` base class
  - Uses Requirements for output validation
  - Integrates with KG_Driver for graph operations

**Example Usage:**
```python
# Create KG-RAG component
kg_rag = KGRagComponent(
    session=session,
    eval_session=eval_session,
    emb_session=emb_session,
    domain="movie",
    config=model_config,
    logger=qa_logger
)

# Generate answer
q = Query(query=query, query_time=query_time)
prediction = await kg_rag.generate_answer(q)
```

**Benefits:**
- ✅ Self-documenting @generative functions with prompts as docstrings
- ✅ Automatic validation with Requirements
- ✅ Easy to test individual components
- ✅ Composable and reusable

### 5. Evaluation Pipeline (run/run_eval.py)

The Mellea-native evaluation implementation showcases:

**Key Features:**
- Uses `@generative` decorator for LLM-as-judge evaluation
- Type-safe `EvaluationResult` Pydantic model for structured outputs
- `EvaluationStats` dataclass for comprehensive metrics tracking
- `MelleaEvaluator` class for batch evaluation with progress bars
- Requirements validation with `VALID_EVAL_SCORE` requirement
- Async batch processing with error recovery

**Example Usage:**
```python
# Define evaluation function
@generative
async def evaluate_single_prediction(
    query: str,
    ground_truth: str,
    prediction: str
) -> EvaluationResult:
    """Evaluate a single prediction against ground truth.

    You are an expert human evaluator. Judge if the prediction matches
    the ground truth answer following these instructions:
    [Detailed evaluation rubric in docstring...]

    Return: {"score": 0 or 1, "explanation": "..."}
    """
    pass

# Create evaluator
evaluator = MelleaEvaluator(session, batch_size=64)

# Evaluate all predictions
stats, history = await evaluator.evaluate_all(
    queries,
    ground_truths_list,
    predictions
)
```

**Benefits:**
- ✅ Self-documenting evaluation rubric in @generative docstring
- ✅ Type-safe evaluation results with Pydantic
- ✅ Detailed statistics tracking (accuracy, token usage, timing)
- ✅ Async batch processing with progress bars
- ✅ Graceful error handling for failed evaluations
- ✅ Requirements validation ensures valid scores

## Migration Guide

To migrate from traditional to Mellea-native:

1. **Identify LLM calls** - Find direct API calls in your code
2. **Create @generative functions** - Convert prompts to @generative docstrings
3. **Add Pydantic models** - Define structured outputs
4. **Add Requirements** - Specify validation rules
5. **Use Components** - Organize related functionality
6. **Apply sampling strategies** - Add RejectionSamplingStrategy for robustness

See individual Mellea-native files for complete examples.
