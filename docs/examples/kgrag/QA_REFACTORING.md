## KG QA Refactoring Guide

This guide explains how the KG QA evaluation script has been refactored to follow Mellea patterns and modern Python best practices.

## Overview

The refactored QA module transforms the original evaluation script from a procedural structure into a clean, type-safe, and maintainable system with proper configuration management and modern async patterns.

## Key Improvements

### 1. Pydantic Configuration Models

**Before** ([run_qa.py](run/run_qa.py)):
```python
# Scattered configuration
DATASET_PATH = os.getenv("KG_BASE_DIRECTORY", ...)
API_BASE = os.getenv("API_BASE", ...)
MODEL_NAME = os.getenv("MODEL_NAME", "")
# ... 10+ environment variables

# Simple dict for config
config = {
    "num_workers": args.num_workers,
    "queue_size": args.queue_size,
    "split": args.split,
}

# No validation
model_config = dict(args.config) if args.config else None
```

**After** ([kg_qa_models.py](kg/kg_qa_models.py)):
```python
from kg.kg_qa_models import QAConfig, QASessionConfig, QADatasetConfig

# Clean, validated configuration
session_config = QASessionConfig(
    api_base="http://localhost:7878/v1",
    model_name="gpt-4",
    eval_model_name="gpt-4",
    emb_model_name="text-embedding-ada-002"
)

qa_config = QAConfig(
    num_workers=128,  # Validated: must be 1-512
    queue_size=128,   # Validated: must be 1-1024
    eval_batch_size=64
)

dataset_config = QADatasetConfig(
    dataset_path="data/crag_movie_dev.jsonl",
    result_path="results/_results.json",
    progress_path="results/_progress.json",
    keep_progress=False
)
```

**Benefits:**
- Type-safe configuration
- Automatic validation
- Clear separation of concerns
- Self-documenting

### 2. Modern Async Patterns

**Before**:
```python
def always_get_an_event_loop():
    """Helper to get or create event loop."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

if __name__ == "__main__":
    # ... setup
    loop = always_get_an_event_loop()  # Old pattern
    loop.run_until_complete(loader.run())
```

**After**:
```python
async def main() -> int:
    """Main async entry point."""
    try:
        # ... setup and processing
        await loader.run()
        return 0
    except KeyboardInterrupt:
        logger.warning("\n⚠️  QA evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ QA evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))  # Modern pattern
```

**Benefits:**
- Modern Python 3.7+ pattern
- Proper exit codes
- Better error handling
- No manual event loop management

### 3. Configuration Factory Functions

**Before**:
```python
# Direct instantiation with environment variables
session = MelleaSession(backend=OpenAIBackend(
    model_id=MODEL_NAME,
    formatter=TemplateFormatter(model_id=MODEL_NAME),
    base_url=API_BASE,
    api_key=API_KEY,
    timeout=TIME_OUT,
    default_headers={'RITS_API_KEY': RITS_API_KEY}
))

eval_session = MelleaSession(backend=OpenAIBackend(
    model_id=EVAL_MODEL_NAME,
    # ... repeat similar code
))
```

**After**:
```python
def create_mellea_session(session_config: QASessionConfig) -> MelleaSession:
    """Create Mellea session for LLM."""
    logger.info(f"Creating main session with model: {session_config.model_name}")
    # ... clean implementation
    return MelleaSession(...)

def create_eval_session(session_config: QASessionConfig) -> MelleaSession:
    """Create evaluation session with fallback to main config."""
    eval_api_base = session_config.eval_api_base or session_config.api_base
    # ... intelligent defaults
    return MelleaSession(...)

# Usage
session = create_mellea_session(session_config)
eval_session = create_eval_session(session_config)
```

**Benefits:**
- DRY (Don't Repeat Yourself)
- Intelligent fallbacks
- Better logging
- Easy to test

### 4. Improved File Path Handling

**Before**:
```python
# Manual string formatting
progress_path = f"results/{f"_{args.prefix}" if args.prefix else ""}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
result_path = f"results/{f"_{args.prefix}" if args.prefix else ""}_results{f"_{args.postfix}" if args.postfix else ""}.json"
```

**After**:
```python
def create_dataset_config(args: argparse.Namespace) -> QADatasetConfig:
    """Create dataset configuration with clean path handling."""
    prefix_str = f"_{args.prefix}" if args.prefix else ""
    postfix_str = f"_{args.postfix}" if args.postfix else ""

    progress_path = f"results/{prefix_str}_progress{postfix_str}.json"
    result_path = f"results/{prefix_str}_results{postfix_str}.json"

    return QADatasetConfig(
        dataset_path=dataset_path,
        result_path=result_path,
        progress_path=progress_path,
        # ... more fields
    )
```

**Benefits:**
- Cleaner code
- Centralized path logic
- Type-safe paths
- Easier to modify

### 5. Better Token Usage Tracking

**Before**:
```python
# Internal helper functions
def _snapshot_token_usage():
    return deepcopy(token_counter.get_token_usage()) if token_counter else {}

def _compute_token_usage_delta(start_usage):
    if not token_counter:
        return {}
    # ... compute delta
```

**After**:
```python
def snapshot_token_usage() -> Dict[str, int]:
    """Snapshot current token usage.

    Returns:
        Dictionary of token counts
    """
    return deepcopy(token_counter.get_token_usage()) if token_counter else {}

def compute_token_usage_delta(start_usage: Dict[str, int]) -> Dict[str, int]:
    """Compute delta in token usage since snapshot.

    Args:
        start_usage: Starting token usage snapshot

    Returns:
        Dictionary of token usage deltas
    """
    # ... implementation with type hints
```

**Benefits:**
- Public, reusable functions
- Full type hints
- Documentation
- Clear intent

### 6. Improved Progress and Results Handling

**Before**:
```python
# Minimal context
print(len(logger.processed_questions))

# No directory creation
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
```

**After**:
```python
# Better progress tracking
print(f"Processed questions: {len(logger.processed_questions)}")
logger.update_progress({"last_question_total": round(elapsed_time, 2)})

# Ensure directory exists
Path("results").mkdir(exist_ok=True)

# Save with informative logging
with open(dataset_config.result_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
logger.info(f"Results saved to: {dataset_config.result_path}")
```

**Benefits:**
- Clearer progress messages
- Prevents file write errors
- Better user feedback
- Path safety

### 7. Comprehensive CLI Arguments

**Before**:
```python
parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=128)
parser.add_argument("--queue-size", type=int, default=128)
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--prefix", type=str)
parser.add_argument("--postfix", type=str)
parser.add_argument("--keep", action='store_true')
parser.add_argument('--config', nargs='*', type=parse_key_value)
args = parser.parse_args()
```

**After**:
```python
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QA evaluation on knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset data/crag_movie_dev.jsonl
  %(prog)s --num-workers 256 --queue-size 256
  %(prog)s --prefix exp1 --postfix test1
  %(prog)s --config route=5 width=30 depth=3
  %(prog)s --verbose --keep
        """
    )
    # ... comprehensive argument definitions with help text
    return parser.parse_args()
```

**Benefits:**
- Comprehensive help text
- Examples in output
- Better user experience
- Self-documenting

### 8. Better Error Handling and Logging

**Before**:
```python
if __name__ == "__main__":
    # ... processing with no error handling

    logger.info(f"Done inference in {args.dataset} dataset ✅")
```

**After**:
```python
async def main() -> int:
    """Main async entry point."""
    try:
        # Verify dataset exists
        if not Path(dataset_config.dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_config.dataset_path}")
            return 1

        # ... processing

        logger.info("=" * 60)
        logger.info("✅ QA evaluation completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {dataset_config.result_path}")
        logger.info(f"Total questions: {len(results) - 1}")
        logger.info(f"Accuracy: {stats.get('accuracy', 'N/A')}")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  QA evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ QA evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
```

**Benefits:**
- Pre-flight validation
- Comprehensive error handling
- Structured logging
- User-friendly messages

## Usage Comparison

### Before
```bash
# All from environment variables
export API_BASE="..."
export MODEL_NAME="..."
# ... many more exports

python run/run_qa.py \
    --num-workers 128 \
    --prefix exp1 \
    --config route=5 width=30
```

### After
```bash
# Environment variables + comprehensive CLI
python run/run_qa_refactored.py \
    --dataset data/crag_movie_dev.jsonl \
    --num-workers 256 \
    --queue-size 256 \
    --prefix exp1 \
    --postfix test1 \
    --config route=5 width=30 depth=3 \
    --eval-batch-size 64 \
    --verbose \
    --keep
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Configuration** | Scattered env vars | Pydantic models |
| **Validation** | None | Automatic |
| **CLI** | 7 arguments | 14 arguments + help |
| **Error Handling** | Minimal | Comprehensive |
| **Logging** | Basic | Structured |
| **Async Pattern** | Old event loops | Modern asyncio.run() |
| **Type Safety** | Minimal | Full type hints |
| **Exit Codes** | None | Proper codes |
| **Factory Functions** | None | Clean factories |
| **Documentation** | Minimal | Comprehensive |

## Examples

### Basic Usage

```bash
# Use default configuration
python run/run_qa_refactored.py
```

### Custom Dataset

```bash
# Specify dataset
python run/run_qa_refactored.py \
    --dataset /path/to/questions.jsonl \
    --domain finance
```

### High Concurrency

```bash
# More workers for faster processing
python run/run_qa_refactored.py \
    --num-workers 256 \
    --queue-size 256
```

### Experiment Naming

```bash
# Organized experiment outputs
python run/run_qa_refactored.py \
    --prefix exp1 \
    --postfix test1
# Creates: results/_exp1_progress_test1.json
#          results/_exp1_results_test1.json
```

### Model Configuration

```bash
# Override model parameters
python run/run_qa_refactored.py \
    --config route=5 width=30 depth=3
```

### Debug Mode

```bash
# Verbose logging and keep progress file
python run/run_qa_refactored.py \
    --verbose \
    --keep
```

## Programmatic Usage

```python
import asyncio
from kg_model import KGModel
from kg.kg_qa_models import QAConfig, QASessionConfig

async def run_evaluation():
    # Create configurations
    session_config = QASessionConfig(
        api_base="http://localhost:7878/v1",
        model_name="gpt-4",
        eval_model_name="gpt-4-turbo"
    )

    qa_config = QAConfig(
        num_workers=256,
        queue_size=256,
        eval_batch_size=128
    )

    # Create sessions
    session = create_mellea_session(session_config)
    eval_session = create_eval_session(session_config)
    emb_session = create_embedding_session(session_config)

    # Create model
    model = KGModel(
        session=session,
        eval_session=eval_session,
        emb_session=emb_session,
        domain="movie",
        config={"route": 5, "width": 30}
    )

    # Generate answer
    answer = await model.generate_answer(
        query="Who won best actor in 2020?",
        query_time=datetime(2024, 3, 19)
    )

    return answer

asyncio.run(run_evaluation())
```

## Migration Path

### Option 1: Use Refactored Version Directly

```bash
python run/run_qa_refactored.py
```

### Option 2: Keep Using Original

```bash
# Original still works
python run/run_qa.py
```

### Option 3: Gradual Migration

Start with the refactored version for new experiments while keeping the original for reproducibility.

## Next Steps

1. ✅ ~~Refactor run script~~ - **DONE**
2. ✅ ~~Add Pydantic configuration~~ - **DONE**
3. ✅ ~~Improve CLI~~ - **DONE**
4. ⏳ Add progress bars (tqdm integration)
5. ⏳ Add result caching
6. ⏳ Support for multiple evaluation methods
7. ⏳ Parallel evaluation
8. ⏳ Unit tests

## Conclusion

The refactored QA script follows Mellea patterns:
- ✅ Type safety with Pydantic
- ✅ Modern async patterns
- ✅ Comprehensive CLI
- ✅ Factory functions
- ✅ Better error handling
- ✅ Proper logging
- ✅ Configuration validation

Both versions coexist for comparison and gradual migration.
