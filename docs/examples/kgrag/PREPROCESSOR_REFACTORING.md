# KG Preprocessor Refactoring Guide

This guide explains how the KG preprocessor has been refactored to follow Mellea's design patterns and modern Python best practices.

## Overview

The refactored preprocessor transforms the original implementation from a monolithic, hard-to-maintain structure into a clean, composable, and type-safe system.

## Key Improvements

### 1. Pydantic Models for Type Safety

**Before** ([kg_preprocessor.py](kg/kg_preprocessor.py)):
```python
# Raw dictionaries with no validation
movie = {
    "title": "The Matrix",
    "budget": 63000000,
    "cast": [{"name": "Keanu Reeves", "character": "Neo"}]
}
```

**After** ([kg_entity_models.py](kg/kg_entity_models.py)):
```python
from kg.kg_entity_models import Movie, MovieCastMember

# Type-safe, validated models
cast = MovieCastMember(name="Keanu Reeves", character="Neo")
movie = Movie(title="The Matrix", budget=63000000, cast=[cast])
```

**Benefits:**
- Automatic validation of data
- Type hints for IDE support
- Clear data contracts
- Catch errors early

### 2. Better Connection Management

**Before**:
```python
def __init__(self):
    self.driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

async def run_query_async(self, query, parameters=None, retries=5, delay=0.5):
    for attempt in range(retries):
        try:
            async with SEMAPHORE:
                async with self.driver.session() as session:
                    result = await session.run(query, parameters)
                    return [record async for record in result]
        except TransientError as e:
            # ... manual retry logic
```

**After** ([kg_preprocessor_refactored.py](kg/kg_preprocessor_refactored.py)):
```python
class Neo4jConnection:
    """Dedicated connection management with context managers."""

    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    @asynccontextmanager
    async def session(self):
        """Proper resource management."""
        async with self._semaphore:
            async with self.driver.session() as session:
                yield session

    async def execute_query(self, query, parameters=None, retries=None, delay=None):
        """Clean retry logic with exponential backoff."""
        # Configurable retries from Neo4jConfig
```

**Benefits:**
- Proper resource management with context managers
- Configuration-driven behavior
- Testable connection logic
- Separation of concerns

### 3. Configuration with Pydantic

**Before**:
```python
# Hardcoded or scattered config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
SEMAPHORE = asyncio.Semaphore(50)  # Global semaphore
```

**After**:
```python
from kg.kg_entity_models import Neo4jConfig, PreprocessorConfig

config = PreprocessorConfig(
    neo4j=Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="secret",
        max_concurrency=50,
        max_retries=5,
        retry_delay=0.5
    ),
    batch_size=10000,
    sample_fractions={"Movie": 0.6, "Person": 0.6}
)

preprocessor = MovieKGPreprocessor(config)
```

**Benefits:**
- Centralized configuration
- Validation of config values
- Easy to test with different configs
- Type-safe configuration

### 4. Batch Operations

**Before**:
```python
async def insert_all_movies_async(self, movie_db):
    tasks = [self.insert_entity_movie_async(movie) for movie in movie_db.values()]

    for task in tqdm(tasks, total=len(tasks), desc="Movies Entities Inserted"):
        await task
```

**After**:
```python
async def insert_all_movies(self) -> None:
    """Insert all movie entities in batches."""
    query = """
    UNWIND $batch AS movie
    MERGE (m:Movie {name: movie.name})
    SET m.release_date = movie.release_date, ...
    """

    data = [self._prepare_movie_data(movie) for movie in self.movie_db.values()]
    await self.batch_insert(query, data, desc="Movies")
```

**Benefits:**
- Much faster (batched queries)
- Less network overhead
- Cleaner code
- Reusable batch_insert method

### 5. Better Error Handling

**Before**:
```python
for attempt in range(retries):
    try:
        # ... execute query
    except TransientError as e:
        if "DeadlockDetected" in str(e):
            logger.warning(f"Deadlock detected, retrying {attempt + 1}/{retries}")
            await asyncio.sleep(delay * (2 ** attempt))
        else:
            raise e
raise RuntimeError("Max retries reached.")
```

**After**:
```python
async def execute_query(self, query, parameters=None, retries=None, delay=None):
    """Execute query with proper error handling."""
    retries = retries or self.config.max_retries
    delay = delay or self.config.retry_delay

    for attempt in range(retries):
        try:
            async with self.session() as session:
                result = await session.run(query, parameters)
                return [dict(record) async for record in result]
        except TransientError as e:
            if "DeadlockDetected" in str(e):
                if attempt < retries - 1:
                    wait_time = delay * (2 ** attempt)
                    logger.warning(
                        f"Deadlock detected, retrying {attempt + 1}/{retries} "
                        f"(waiting {wait_time:.2f}s)"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for query: {query[:100]}...")
                    raise RuntimeError(f"Max retries exceeded: {e}")
            else:
                raise

    raise RuntimeError("Max retries reached")
```

**Benefits:**
- Detailed error messages
- Configurable retry behavior
- Better logging
- Clear error states

### 6. Separation of Concerns

**Before**: Single file with 1100+ lines mixing:
- Connection management
- Data loading
- Entity insertion
- Relation insertion
- Multiple domain preprocessors

**After**: Organized into focused modules:
```
kg/
├── kg_entity_models.py          # Pydantic models (200 lines)
├── kg_preprocessor_refactored.py # Base classes (600 lines)
├── kg_preprocessor.py            # Original (for comparison)
└── kg_rep.py                     # Utilities
```

**Benefits:**
- Easier to understand
- Easier to test
- Easier to extend
- Better code organization

### 7. Modern Python Patterns

**Before**:
```python
# Manual event loop management
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(main())

# No type hints
def insert_entity_movie_async(self, movie):
    # ...

# No docstrings
async def insert_all_movies_async(self, movie_db):
    # ...
```

**After**:
```python
# Modern async
async def main() -> int:
    """Main entry point with return code."""
    # ...

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

# Full type hints
async def insert_all_movies(self) -> None:
    """Insert all movie entities in batches.

    Uses the UNWIND pattern for efficient batch insertion.
    """
    # ...

# Context managers
@asynccontextmanager
async def session(self):
    """Get a Neo4j session with proper resource management."""
    # ...
```

**Benefits:**
- Modern Python 3.7+ patterns
- Better IDE support
- Self-documenting code
- Proper resource management

## Migration Path

### Option 1: Use Refactored Version Directly

```python
from kg.kg_preprocessor_refactored import MovieKGPreprocessor

preprocessor = MovieKGPreprocessor()
await preprocessor.connect()
await preprocessor.preprocess()
await preprocessor.close()
```

### Option 2: Gradual Migration

The refactored version maintains backward compatibility through aliases:

```python
# This still works
from kg.kg_preprocessor_refactored import KG_Preprocessor, MovieKG_Preprocessor

# Same interface as before
preprocessor = MovieKG_Preprocessor()
```

### Option 3: Custom Configuration

```python
from kg.kg_entity_models import Neo4jConfig, PreprocessorConfig
from kg.kg_preprocessor_refactored import MovieKGPreprocessor

# Custom configuration
config = PreprocessorConfig(
    neo4j=Neo4jConfig(
        uri="bolt://my-neo4j:7687",
        user="admin",
        password="secret",
        max_concurrency=100  # Higher concurrency
    ),
    batch_size=50000,  # Larger batches
    sample_fractions={"Movie": 0.8, "Person": 0.8}  # Keep more data
)

preprocessor = MovieKGPreprocessor(config)
```

## File Comparison

### Before
```
docs/examples/kgrag/kg/
└── kg_preprocessor.py           # 1124 lines
    ├── KG_Preprocessor           # Base class
    ├── MovieKG_Preprocessor      # 350+ lines
    ├── SoccerKG_Preprocessor     # 200+ lines
    ├── NBAKG_Preprocessor        # 300+ lines
    ├── MusicKG_Preprocessor      # 230+ lines
    └── ...                       # Other preprocessors
```

### After
```
docs/examples/kgrag/kg/
├── kg_entity_models.py          # 250 lines - Pydantic models
├── kg_preprocessor_refactored.py # 680 lines - Clean implementation
└── kg_preprocessor.py            # 1124 lines - Original (for reference)
```

## Usage Examples

### Basic Usage (Same as Before)

```python
from kg.kg_preprocessor_refactored import MovieKGPreprocessor

async def main():
    preprocessor = MovieKGPreprocessor()
    await preprocessor.connect()

    try:
        await preprocessor.preprocess()
        print("✅ Preprocessing complete!")
    finally:
        await preprocessor.close()

asyncio.run(main())
```

### With Custom Configuration

```python
from kg.kg_entity_models import PreprocessorConfig
from kg.kg_preprocessor_refactored import MovieKGPreprocessor

# Load config from environment or customize
config = PreprocessorConfig.from_env()
config.neo4j.max_concurrency = 100

preprocessor = MovieKGPreprocessor(config)
await preprocessor.connect()
await preprocessor.preprocess()
await preprocessor.close()
```

### With run_kg_preprocess.py

The refactored `run_kg_preprocess.py` now supports:

```bash
# Process single domain
python run/run_kg_preprocess.py --domain movie

# Process multiple domains
python run/run_kg_preprocess.py --domain movie soccer

# Process all domains
python run/run_kg_preprocess.py --domain all

# Dry run to preview
python run/run_kg_preprocess.py --domain all --dry-run

# Verbose logging
python run/run_kg_preprocess.py --domain movie --verbose
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Type Safety** | Raw dicts | Pydantic models |
| **Configuration** | Scattered env vars | Centralized config |
| **Connection Management** | Manual | Context managers |
| **Batch Operations** | One-by-one | Batch insert |
| **Error Handling** | Basic | Detailed with retries |
| **Code Organization** | Monolithic | Modular |
| **Testing** | Hard to test | Easy to test |
| **Documentation** | Minimal | Comprehensive |
| **Performance** | Slower (individual inserts) | Faster (batched) |
| **Maintainability** | Difficult | Easy |

## Next Steps

1. **Migrate other preprocessors** - Apply the same patterns to Soccer, NBA, Music, etc.
2. **Add unit tests** - Easier to test with the refactored structure
3. **Performance benchmarks** - Measure improvements from batch operations
4. **Add more Pydantic validation** - Stricter data validation rules
5. **Integration with Mellea Components** - Use as part of larger Mellea pipelines

## Extending the Refactored Preprocessor

### Adding a New Domain

```python
from kg.kg_preprocessor_refactored import KGPreprocessorBase
from kg.kg_entity_models import PreprocessorConfig

class MusicKGPreprocessor(KGPreprocessorBase):
    """Music domain preprocessor."""

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        super().__init__(config)
        # Load music data
        self.artist_db = self.load_json_file("dataset/music/artists.json")
        self.song_db = self.load_json_file("dataset/music/songs.json")

    async def create_indices(self) -> None:
        """Create music-specific indices."""
        await self.create_index_if_not_exists("Artist", "name")
        await self.create_index_if_not_exists("Song", "name")

    async def preprocess(self) -> None:
        """Run music preprocessing pipeline."""
        await self.create_indices()
        await self.insert_artists()
        await self.insert_songs()
        # ... more steps
```

### Custom Validation

```python
from pydantic import validator

class CustomMovie(Movie):
    """Movie with additional validation."""

    @validator("budget")
    def budget_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("Budget cannot be negative")
        return v

    @validator("release_date")
    def valid_release_date(cls, v):
        if v:
            # Ensure date format is valid
            datetime.strptime(v, "%Y-%m-%d")
        return v
```

## Conclusion

The refactored preprocessor follows Mellea's philosophy of:
- **Type safety** with Pydantic models
- **Composability** with modular design
- **Maintainability** with clean separation of concerns
- **Modern Python** with async/await and type hints

Both versions coexist in the repository for comparison and gradual migration.
