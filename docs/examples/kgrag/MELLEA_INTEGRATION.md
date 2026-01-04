# Mellea-Native KG-RAG Implementation

This document explains the Mellea-native implementation of KG-RAG and how it showcases Mellea's core patterns.

## Overview

The KG-RAG example now includes **parallel implementations** for all major pipeline components:

1. **KG Preprocessing Pipeline**:
   - Traditional: `run/run_kg_preprocess.py` - Basic preprocessing with Neo4j
   - Mellea-Native: `run/run_kg_preprocess_mellea.py` - Enhanced with statistics tracking, concurrent processing, and detailed reporting

2. **KG Embedding Pipeline**:
   - Traditional: `run/run_kg_embed.py` - Direct embedding API calls
   - Mellea-Native: `run/run_kg_embed_mellea.py` - Uses kg_utils_mellea with enhanced error handling

3. **KG Update Pipeline**:
   - Traditional: `run/run_kg_update.py` - Manual extraction and validation
   - Mellea-Native: `run/run_kg_update_mellea.py` - Uses @generative for extraction, alignment, and merging

4. **QA Pipeline**:
   - Traditional: `run/run_qa.py` - Direct LLM API calls with manual validation
   - Mellea-Native: `run/run_qa_mellea.py` - Uses Mellea's @generative, Requirements, and Components

5. **Evaluation Pipeline**:
   - Traditional: `run/run_eval.py` - Manual evaluation with direct LLM calls
   - Mellea-Native: `run/run_eval_mellea.py` - Uses @generative for LLM-as-judge evaluation

Both implementations produce the same results, but the Mellea-native versions demonstrate best practices for building robust, composable LLM applications.

## Key Benefits

✅ **Type Safety** - Pydantic models ensure valid outputs  
✅ **Robustness** - Automatic validation and retry logic  
✅ **Composability** - Reusable functions and components  
✅ **Maintainability** - Self-documenting code  
✅ **Testability** - Easy to test individual pieces  

## Quick Start

```bash
cd docs/examples/kgrag

# Run Mellea-native preprocessing
uv run --with mellea run/run_kg_preprocess_mellea.py --domain movie --verbose

# Run Mellea-native KG embedding
uv run --with mellea run/run_kg_embed_mellea.py --batch-size 8192 --verbose

# Run Mellea-native KG update
uv run --with mellea run/run_kg_update_mellea.py --num-workers 64

# Run Mellea-native QA
uv run --with mellea run/run_qa_mellea.py --num-workers 1 --prefix mellea

# Run Mellea-native evaluation
uv run --with mellea run/run_eval_mellea.py --prefix mellea --verbose

# Compare with traditional versions
uv run --with mellea run/run_kg_preprocess.py --domain movie
uv run --with mellea run/run_kg_embed.py
uv run --with mellea run/run_kg_update.py --num-workers 64
uv run --with mellea run/run_qa.py --num-workers 1 --prefix traditional
uv run --with mellea run/run_eval.py --prefix traditional
```

## Architecture

### 1. KG Preprocessing (run_kg_preprocess_mellea.py)

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

### 2. KG Embedding (run_kg_embed_mellea.py)

The Mellea-native embedding implementation showcases:

**Key Features:**
- Uses `kg_utils_mellea.generate_embedding_mellea()` for consistent embedding generation
- Extends `KGEmbedder` with `MelleaKGEmbedder` class for enhanced functionality
- Adds embedding session testing with `test_embedding_session()`
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

### 3. KG Update (run_kg_update_mellea.py)

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

### 4. QA Pipeline (run_qa_mellea.py)

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

### 5. Evaluation Pipeline (run_eval_mellea.py)

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
