## KG Updater Refactoring Guide

This guide explains how the KG updater module has been refactored to follow Mellea patterns and modern Python best practices.

## Overview

The refactored updater module transforms the original implementation from a procedural, configuration-scattered structure into a clean, type-safe, and maintainable system.

## Key Improvements

### 1. Pydantic Configuration Models

**Before** ([run_kg_update.py](run/run_kg_update.py)):
```python
# Scattered configuration across multiple environment variables
DATASET_PATH = os.getenv("KG_BASE_DIRECTORY", ...)
API_BASE = os.getenv("API_BASE", "http://localhost:7878/v1")
API_KEY = os.getenv("API_KEY", "dummy")
TIME_OUT = int(os.getenv("TIME_OUT", "1800"))
MODEL_NAME = os.getenv("MODEL_NAME", "")
EVAL_API_BASE = os.getenv("EVAL_API_BASE", "")
EVAL_API_KEY = os.getenv("EVAL_API_KEY", "dummy")
# ... 10+ more environment variables

# Hardcoded configuration in kg_updater.py
ALIGN_ENTITY = True
ALIGN_RELATION = True
MERGE_ENTITY = True
MERGE_RELATION = True
MAX_RETRIES = 3
MAX_GENERATION_TOKENS = 20000
# ... more scattered constants
```

**After** ([kg_updater_models.py](kg/kg_updater_models.py)):
```python
from kg.kg_updater_models import UpdaterConfig, SessionConfig, DatasetConfig

# Clean, validated configuration
session_config = SessionConfig(
    api_base="http://localhost:7878/v1",
    api_key="your-key",
    model_name="gpt-4",
    timeout=1800
)

updater_config = UpdaterConfig(
    num_workers=128,
    queue_size=128,
    align_entity=True,
    max_retries=3
)

dataset_config = DatasetConfig(
    dataset_path="data/corpus.jsonl.bz2",
    domain="movie",
    progress_path="results/progress.json"
)
```

**Benefits:**
- Centralized configuration
- Type validation
- Clear separation of concerns
- Easy to override
- Self-documenting code

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
    # ... setup code
    loop = always_get_an_event_loop()  # Old pattern
    loop.run_until_complete(loader.run())
```

**After**:
```python
async def main() -> int:
    """Main async entry point."""
    try:
        # ... setup code
        await loader.run()
        return 0
    except KeyboardInterrupt:
        logger.warning("\n⚠️  KG update interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ KG update failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))  # Modern pattern
```

**Benefits:**
- Modern Python 3.7+ pattern
- Proper exit codes
- Better error handling
- No manual event loop management

### 3. CLI Arguments

**Before**:
```python
parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=64)
parser.add_argument("--queue-size", type=int, default=64)
parser.add_argument("--progress-path", type=str, default="...")

# Limited options, no help text
args = parser.parse_args()
```

**After**:
```python
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update knowledge graph from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset data/corpus.jsonl.bz2
  %(prog)s --num-workers 128 --queue-size 128
  %(prog)s --domain movie --progress-path results/progress.json
  %(prog)s --verbose
        """
    )

    parser.add_argument("--dataset", type=str, help="Path to dataset file")
    parser.add_argument("--domain", type=str, default="movie")
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()
```

**Benefits:**
- Comprehensive help text
- Examples in --help output
- Verbose mode for debugging
- Domain specification
- Dataset path override

### 4. Better Configuration Factory Functions

**Before**:
```python
# Direct instantiation with raw env vars
session = MelleaSession(backend=
    OpenAIBackend(
        model_id=MODEL_NAME,
        formatter=TemplateFormatter(model_id=MODEL_NAME),
        base_url=API_BASE,
        api_key=API_KEY,
        timeout=TIME_OUT,
        default_headers={'RITS_API_KEY': RITS_API_KEY}
    ))

# Conditional logic scattered
if EMB_API_BASE:
    emb_session = openai.AsyncOpenAI(...)
else:
    emb_session = SentenceTransformer(...)
```

**After**:
```python
def create_mellea_session(session_config: SessionConfig) -> MelleaSession:
    """Create Mellea session for LLM."""
    logger.info(f"Creating session with model: {session_config.model_name}")

    headers = {}
    if session_config.rits_api_key:
        headers['RITS_API_KEY'] = session_config.rits_api_key

    return MelleaSession(
        backend=OpenAIBackend(
            model_id=session_config.model_name,
            formatter=TemplateFormatter(model_id=session_config.model_name),
            base_url=session_config.api_base,
            api_key=session_config.api_key,
            timeout=session_config.timeout,
            default_headers=headers if headers else None
        )
    )

def create_embedding_session(session_config: SessionConfig) -> Any:
    """Create embedding session (OpenAI or local model)."""
    if session_config.emb_api_base:
        logger.info(f"Using OpenAI API at {session_config.emb_api_base}")
        return openai.AsyncOpenAI(...)
    else:
        logger.info("Using local SentenceTransformer")
        return SentenceTransformer(...)

# Usage
session = create_mellea_session(session_config)
emb_session = create_embedding_session(session_config)
```

**Benefits:**
- Clear separation of concerns
- Easy to test
- Better logging
- Reusable functions
- Type-safe

### 5. Configuration Validation

**Before**:
```python
# No validation
config = {
    "num_workers": args.num_workers,
    "queue_size": args.queue_size,
}
```

**After**:
```python
class UpdaterConfig(BaseModel):
    """Configuration for KG updater operations."""
    num_workers: int = Field(default=64, ge=1, le=256)  # Validated range
    queue_size: int = Field(default=64, ge=1, le=512)   # Validated range
    max_retries: int = Field(default=3, ge=1, le=10)
    # ... more validated fields

# Will raise validation error if out of range
config = UpdaterConfig(num_workers=1000)  # Error: must be <= 256
```

**Benefits:**
- Automatic validation
- Clear constraints
- Better error messages
- Self-documenting limits

### 6. Better Error Handling

**Before**:
```python
if __name__ == "__main__":
    # ... setup
    loop = always_get_an_event_loop()
    loop.run_until_complete(loader.run())  # No error handling

    logger.info(f"Done updating KG ✅")
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
        await loader.run()

        logger.info("✅ KG update completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  KG update interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ KG update failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))  # Proper exit codes
```

**Benefits:**
- Pre-flight validation
- Graceful error handling
- Proper exit codes
- User-friendly messages
- Optional verbose mode

### 7. Improved Logging

**Before**:
```python
# Minimal context
print(logger.processed_docs)
logger.info(f"Done updating KG ✅")
logger.info(f"Token usage: {token_counter.get_token_usage()}")
```

**After**:
```python
logger.info("=" * 60)
logger.info("KG Update Configuration:")
logger.info("=" * 60)
logger.info(f"Dataset: {dataset_config.dataset_path}")
logger.info(f"Domain: {dataset_config.domain}")
logger.info(f"Workers: {updater_config.num_workers}")
logger.info(f"Queue size: {updater_config.queue_size}")
logger.info(f"Progress path: {dataset_config.progress_path}")
logger.info("=" * 60)

# ... processing

logger.info("=" * 60)
logger.info("✅ KG update completed successfully!")
logger.info("=" * 60)
logger.info(f"Total processed docs: {kg_logger.processed_docs}")
logger.info(f"Token usage: {token_counter.get_token_usage()}")
```

**Benefits:**
- Clear visual structure
- Comprehensive configuration logging
- Progress visibility
- Better debugging

## Usage Comparison

### Before
```bash
# All configuration from environment variables
export API_BASE="..."
export MODEL_NAME="..."
export EMB_API_BASE="..."
# ... many more exports

python run/run_kg_update.py --num-workers 64 --queue-size 64
```

### After
```bash
# Environment variables + CLI args
python run/run_kg_update_refactored.py \
    --dataset data/corpus.jsonl.bz2 \
    --domain movie \
    --num-workers 128 \
    --queue-size 128 \
    --progress-path results/progress.json \
    --verbose
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Configuration** | 15+ env vars | Pydantic models |
| **Validation** | None | Automatic |
| **CLI** | 3 arguments | 7 arguments + help |
| **Error Handling** | Minimal | Comprehensive |
| **Logging** | Basic | Detailed |
| **Async Pattern** | Old event loops | Modern asyncio.run() |
| **Type Safety** | Minimal | Full type hints |
| **Exit Codes** | None | Proper codes |
| **Code Organization** | 95 lines | 280 lines (but cleaner) |
| **Testing** | Hard to test | Easy to test |

## Examples

### Basic Usage

```bash
# Use default configuration from environment
python run/run_kg_update_refactored.py
```

### Custom Dataset

```bash
# Specify dataset path
python run/run_kg_update_refactored.py \
    --dataset /path/to/corpus.jsonl.bz2 \
    --domain finance
```

### High Concurrency

```bash
# More workers for faster processing
python run/run_kg_update_refactored.py \
    --num-workers 256 \
    --queue-size 256
```

### Debug Mode

```bash
# Verbose logging for debugging
python run/run_kg_update_refactored.py --verbose
```

### Custom Progress Path

```bash
# Different progress log location
python run/run_kg_update_refactored.py \
    --progress-path /tmp/my_progress.json
```

## Programmatic Usage

```python
import asyncio
from kg.kg_updater import KG_Updater
from kg.kg_updater_models import UpdaterConfig, SessionConfig
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend, TemplateFormatter

async def main():
    # Create configurations
    session_config = SessionConfig(
        api_base="http://localhost:7878/v1",
        api_key="your-key",
        model_name="gpt-4",
        emb_model_name="text-embedding-ada-002"
    )

    updater_config = UpdaterConfig(
        num_workers=128,
        queue_size=128
    )

    # Create sessions
    session = MelleaSession(
        backend=OpenAIBackend(
            model_id=session_config.model_name,
            base_url=session_config.api_base,
            api_key=session_config.api_key
        )
    )

    # Create updater
    updater = KG_Updater(
        session=session,
        emb_session=create_embedding_session(session_config),
        config=updater_config.model_dump(),
        logger=my_logger
    )

    # Process documents
    await updater.process_doc(doc, domain="movie")

asyncio.run(main())
```

## Migration Path

### Option 1: Use Refactored Version Directly

```bash
python run/run_kg_update_refactored.py
```

### Option 2: Keep Using Original

```bash
# Original still works
python run/run_kg_update.py
```

### Option 3: Gradual Migration

Update environment variables one at a time while moving to the refactored version.

## Next Steps

1. ✅ ~~Refactor run script~~ - **DONE**
2. ✅ ~~Add Pydantic configuration~~ - **DONE**
3. ✅ ~~Improve CLI~~ - **DONE**
4. ⏳ Refactor kg_updater.py itself (large task)
5. ⏳ Add unit tests
6. ⏳ Add progress bars
7. ⏳ Support for batch processing
8. ⏳ Better error recovery

## Conclusion

The refactored updater script follows Mellea patterns:
- ✅ Type safety with Pydantic
- ✅ Modern async patterns
- ✅ Comprehensive CLI
- ✅ Better error handling
- ✅ Proper logging
- ✅ Configuration validation

Both versions coexist for comparison and gradual migration.
