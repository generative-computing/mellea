## KG Embedding Refactoring Guide

This guide explains how the KG embedding module has been refactored to follow Mellea patterns and modern Python best practices.

## Overview

The refactored embedding module transforms the original implementation from a procedural, hard-coded structure into a clean, configurable, and type-safe system.

## Key Improvements

### 1. Pydantic Configuration Models

**Before** ([run_kg_embed.py](run/run_kg_embed.py)):
```python
# Scattered configuration
API_KEY = os.getenv("API_KEY", "dummy")
EMB_API_BASE = os.getenv("EMB_API_BASE", "")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "")
EMB_TIME_OUT = int(os.getenv("EMB_TIME_OUT", "1800"))
RITS_API_KEY = os.getenv("RITS_API_KEY", "")
SEMAPHORE = asyncio.Semaphore(50)  # Global semaphore

# Hardcoded values
batch_size=8192
concurrent_batches=64
storage_batch_size=50_000
```

**After** ([kg_embed_models.py](kg/kg_embed_models.py)):
```python
class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    api_key: str = Field(default="dummy")
    api_base: Optional[str] = Field(default=None)
    model_name: str = Field(default="")
    timeout: int = Field(default=1800, ge=1, le=3600)
    vector_dimensions: int = Field(default=768, ge=1, le=4096)
    batch_size: int = Field(default=8192, ge=1, le=100000)
    concurrent_batches: int = Field(default=64, ge=1, le=256)
    storage_batch_size: int = Field(default=50000, ge=100, le=100000)

# Usage
config = EmbeddingConfig(batch_size=10000, vector_dimensions=1024)
```

**Benefits:**
- Centralized configuration
- Validation of values (ranges, types)
- Easy to override via CLI or code
- Type-safe with IDE support

### 2. Safer Data Handling

**Before**:
```python
async def store_embedding(self, query, data_str, datas, embeddings_np, batch_size=50_000):
    batch = []
    for data, embedding in tqdm(zip(datas, embeddings_np), desc="Storing embedding"):
        batch.append(eval(data_str))  # ⚠️ UNSAFE: Using eval()!

        if len(batch) == batch_size:
            params = {"data": list(batch)}
            await kg_driver.run_query_async(query, params)
            batch = []
```

**After** ([kg_embedder_refactored.py](kg/kg_embedder_refactored.py)):
```python
async def store_embeddings_batched(
    self,
    query: str,
    embeddings_data: List[Dict[str, Any]],
    desc: str = "Storing embeddings"
) -> None:
    """Store embeddings in Neo4j in batches."""
    batch = []

    for data in tqdm(embeddings_data, desc=desc):
        batch.append(data)  # ✅ SAFE: No eval()!

        if len(batch) >= self.config.storage_batch_size:
            await kg_driver.run_query_async(query, {"data": batch})
            batch = []

# Usage
embeddings_data = [
    {"id": entity.id, "embedding": embedding.tolist()}
    for entity, embedding in zip(entities, embeddings_np)
]
await self.store_embeddings_batched(query, embeddings_data)
```

**Benefits:**
- No use of `eval()` - much safer
- Clear data structures
- Better type safety
- Easier to test

### 3. Better Error Handling

**Before**:
```python
async def get_embedding(self, description_list, batch_size=8192, concurrent_batches=64):
    tasks = []
    embeddings_list = []
    for i in tqdm(range(0, len(description_list), batch_size), desc="Embedding"):
        tasks.append(embed_batch(i))

        if len(tasks) >= concurrent_batches:
            results = await asyncio.gather(*tasks)  # No error handling!
            for r in results:
                embeddings_list.extend(np.array(r))
            tasks = []
```

**After**:
```python
async def generate_embeddings_batched(
    self,
    texts: List[str],
    desc: str = "Embedding"
) -> np.ndarray:
    """Generate embeddings with proper error handling."""
    # ... setup code

    if len(tasks) >= self.config.concurrent_batches:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch embedding failed: {result}")
                self.stats.failed_batches += 1
            else:
                all_embeddings.extend(np.array(result))
                self.stats.total_batches += 1

        tasks = []
```

**Benefits:**
- Errors don't crash the entire pipeline
- Failed batches are tracked
- Better logging
- More robust

### 4. Statistics Tracking

**Before**:
```python
# No tracking of what was embedded or any failures
logger.info("Neo4j KG embedding completed ✅")
```

**After**:
```python
class EmbeddingStats(BaseModel):
    """Statistics about embedding operations."""
    total_entities: int = 0
    total_relations: int = 0
    total_entity_schemas: int = 0
    total_relation_schemas: int = 0
    entities_embedded: int = 0
    relations_embedded: int = 0
    schemas_embedded: int = 0
    total_batches: int = 0
    failed_batches: int = 0

# Usage
stats = await embedder.embed_all()

# Logging
logger.info("Embedding Statistics:")
logger.info(f"  Entities: {stats.entities_embedded}/{stats.total_entities}")
logger.info(f"  Relations: {stats.relations_embedded}/{stats.total_relations}")
logger.info(f"  Failed Batches: {stats.failed_batches}")
```

**Benefits:**
- Visibility into the embedding process
- Track failures
- Debugging support
- Progress monitoring

### 5. Modern Async Patterns

**Before**:
```python
if __name__ == "__main__":
    embedder = KG_Embedder(emb_session)

    async def main():
        await embedder.embed()

    loop = asyncio.new_event_loop()  # Old pattern
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
```

**After**:
```python
async def main() -> int:
    """Main async entry point."""
    try:
        embedder = KGEmbedder(emb_session, config)
        stats = await embedder.embed_all()
        return 0
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Embedding interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))  # Modern pattern
```

**Benefits:**
- Modern Python 3.7+ pattern
- Proper exit codes
- Better error handling
- Cleaner code

### 6. CLI Arguments

**Before**:
```python
# No CLI arguments - everything from environment variables
if __name__ == "__main__":
    embedder = KG_Embedder(emb_session)
    # ...
```

**After**:
```python
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and store KG embeddings"
    )

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--storage-batch-size", type=int, default=None)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()

# Usage
python run_kg_embed_refactored.py --batch-size 10000 --verbose
```

**Benefits:**
- Flexible configuration
- Override env vars easily
- Better for testing
- Standard CLI patterns

### 7. Modular Design

**Before**: Single monolithic class (291 lines)

**After**: Separated concerns
```
kg/
├── kg_embed_models.py           # 90 lines - Pydantic models
├── kg_embedder_refactored.py    # 450 lines - Clean implementation
└── kg_embed.py                  # 291 lines - Original
run/
├── run_kg_embed_refactored.py   # 180 lines - CLI with args
└── run_kg_embed.py              # 291 lines - Original
```

**Benefits:**
- Easier to understand
- Easier to test
- Easier to extend
- Better organization

## Usage Comparison

### Before
```python
# Only environment variables
export API_KEY="..."
export EMB_API_BASE="..."
export EMB_MODEL_NAME="..."

python run/run_kg_embed.py
```

### After
```python
# Environment variables + CLI args
python run/run_kg_embed_refactored.py

# Or with custom configuration
python run/run_kg_embed_refactored.py \
    --batch-size 10000 \
    --storage-batch-size 100000 \
    --dimensions 1024 \
    --verbose

# Programmatic usage
from kg.kg_embedder_refactored import KGEmbedder
from kg.kg_embed_models import EmbeddingConfig

config = EmbeddingConfig(
    batch_size=10000,
    vector_dimensions=1024
)

embedder = KGEmbedder(emb_session, config)
stats = await embedder.embed_all()
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Configuration** | Scattered env vars | Pydantic config |
| **Safety** | Uses eval() | Safe data structures |
| **Error Handling** | Basic | Comprehensive |
| **Statistics** | None | Full tracking |
| **CLI** | No arguments | Full argparse |
| **Async Pattern** | Old event loops | Modern asyncio.run() |
| **Type Safety** | Minimal | Full type hints |
| **Documentation** | Minimal | Comprehensive |
| **Testing** | Hard to test | Easy to test |
| **Code Organization** | Monolithic | Modular |

## Migration Path

### Option 1: Use Refactored Version Directly

```python
python run/run_kg_embed_refactored.py
```

### Option 2: Gradual Migration

```python
# Import refactored components
from kg.kg_embedder_refactored import KGEmbedder
from kg.kg_embed_models import EmbeddingConfig

# Use with custom config
config = EmbeddingConfig(batch_size=10000)
embedder = KGEmbedder(emb_session, config)
```

### Option 3: Keep Using Original

```python
# Original still works
python run/run_kg_embed.py
```

## Examples

### Basic Usage

```bash
# Use default configuration
python run/run_kg_embed_refactored.py
```

### Custom Batch Sizes

```bash
# Larger batches for better performance
python run/run_kg_embed_refactored.py \
    --batch-size 16384 \
    --storage-batch-size 100000
```

### Different Vector Dimensions

```bash
# For models with different dimensions
python run/run_kg_embed_refactored.py --dimensions 1024
```

### Verbose Logging

```bash
# Debug mode
python run/run_kg_embed_refactored.py --verbose
```

### Programmatic Usage

```python
import asyncio
from kg.kg_embedder_refactored import KGEmbedder
from kg.kg_embed_models import EmbeddingConfig
import openai

async def main():
    # Custom configuration
    config = EmbeddingConfig(
        api_key="your-api-key",
        api_base="https://api.openai.com/v1",
        batch_size=10000,
        vector_dimensions=1536,  # OpenAI ada-002
    )

    # Create session
    session = openai.AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.api_base
    )

    # Create embedder
    embedder = KGEmbedder(session, config)

    # Run embedding
    stats = await embedder.embed_all()

    print(f"Embedded {stats.entities_embedded} entities")
    print(f"Failed batches: {stats.failed_batches}")

asyncio.run(main())
```

## Testing

```python
# Test with small batch sizes
python run/run_kg_embed_refactored.py \
    --batch-size 100 \
    --storage-batch-size 1000 \
    --verbose
```

## Next Steps

1. ✅ ~~Refactor embedding generation~~ - **DONE**
2. ✅ ~~Add Pydantic configuration~~ - **DONE**
3. ✅ ~~Improve error handling~~ - **DONE**
4. ⏳ Add unit tests
5. ⏳ Add retry logic for failed batches
6. ⏳ Support for multiple embedding models
7. ⏳ Incremental embedding updates
8. ⏳ Embedding quality validation

## Conclusion

The refactored embedding module follows Mellea patterns:
- ✅ Type safety with Pydantic
- ✅ Better error handling
- ✅ Modern async patterns
- ✅ Modular, testable code
- ✅ Comprehensive documentation
- ✅ No unsafe operations (eval)

Both versions coexist for comparison and gradual migration.
