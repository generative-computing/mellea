# Creating a Demo Dataset

The full CRAG movie database is quite large (225MB+), which can make demos slow. This guide shows you how to create a smaller, focused demo dataset.

## Quick Start

### Option 1: Recent Oscar Winners (Recommended for Demos)

This creates a small dataset focused on recent Oscar winners and nominees from 2020-2024:

```bash
cd /home/yzhu/mellea/docs/examples/kgrag

# Create demo dataset with ~100 recent movies
uv run python create_demo_dataset.py \
    --year-start 2020 \
    --year-end 2024 \
    --max-movies 100 \
    --topics "oscar,academy award" \
    --include-related
```

### Option 2: Animated Films

Great for testing with animated movie questions:

```bash
uv run python create_demo_dataset.py \
    --year-start 2015 \
    --year-end 2024 \
    --max-movies 80 \
    --topics "animated,pixar,disney,dreamworks" \
    --include-related
```

### Option 3: Marvel Cinematic Universe

For superhero movie questions:

```bash
uv run python create_demo_dataset.py \
    --year-start 2018 \
    --year-end 2024 \
    --max-movies 50 \
    --topics "marvel,avengers,mcu" \
    --include-related
```

### Option 4: Minimal Dataset

Smallest possible dataset for quick testing:

```bash
uv run python create_demo_dataset.py \
    --year-start 2023 \
    --year-end 2024 \
    --max-movies 30 \
    --include-related
```

## Using the Demo Dataset

After creating the demo dataset, you have two options:

### Option A: Switch Directories (Recommended)

```bash
# Backup the original full database
mv dataset/movie dataset/movie_full

# Use the demo database
mv dataset/movie_demo dataset/movie

# To restore the full database later:
# mv dataset/movie dataset/movie_demo
# mv dataset/movie_full dataset/movie
```

### Option B: Update Environment Variable

Update your `.env` file to point to the demo directory:

```bash
# In .env file, change:
KG_BASE_DIRECTORY=/path/to/mellea/docs/examples/kgrag/dataset

# To use demo, either:
# 1. Keep KG_BASE_DIRECTORY the same and rename directories (Option A above)
# 2. Or change to:
# KG_BASE_DIRECTORY=/path/to/mellea/docs/examples/kgrag/dataset_demo
```

## Processing the Demo Dataset

Once you've switched to the demo dataset, run the preprocessing steps:

```bash
# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export KG_BASE_DIRECTORY="$(pwd)/dataset"

# Step 1: Preprocess (much faster with demo data!)
uv run --with mellea run/run_kg_preprocess.py

# Step 2: Generate embeddings
uv run --with mellea run/run_kg_embed.py

# Step 3: Update knowledge graph
uv run --with mellea run/run_kg_update.py --num-workers 4 --queue-size 10
```

## Size Comparison

| Dataset Type | Movies | People | Years | Total Size | Processing Time* |
|-------------|--------|--------|-------|------------|-----------------|
| **Full Database** | ~50,000 | ~100,000 | 100+ | 225 MB | 4-6 hours |
| **Recent Oscar (100)** | ~100 | ~500 | 5 | ~5 MB | 15-20 minutes |
| **Animated (80)** | ~80 | ~400 | 10 | ~4 MB | 10-15 minutes |
| **Marvel (50)** | ~50 | ~300 | 7 | ~3 MB | 8-12 minutes |
| **Minimal (30)** | ~30 | ~200 | 2 | ~2 MB | 5-8 minutes |

*Approximate processing time on 16GB RAM, 8 CPU cores

## Command Line Options

```
usage: create_demo_dataset.py [-h] [--year-start YEAR_START] [--year-end YEAR_END]
                              [--max-movies MAX_MOVIES] [--topics TOPICS]
                              [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                              [--include-related]

Create a smaller demo dataset from the full CRAG movie database

optional arguments:
  -h, --help            show this help message and exit
  --year-start YEAR_START
                        Start year for movies (default: 2020)
  --year-end YEAR_END   End year for movies (default: 2024)
  --max-movies MAX_MOVIES
                        Maximum number of movies to include (default: 100)
  --topics TOPICS       Comma-separated topics to filter by (e.g., 'oscar,marvel,animated')
  --input-dir INPUT_DIR
                        Input directory with full database (default: dataset/movie)
  --output-dir OUTPUT_DIR
                        Output directory for demo database (default: dataset/movie_demo)
  --include-related     Include all people and years related to selected movies
```

## Tips for Demo Datasets

### 1. Use `--include-related`

This flag ensures that all people (actors, directors) and years referenced in your selected movies are included. Without it, you might get incomplete graph traversals.

### 2. Choose Coherent Topics

Pick topics that have overlapping entities:
- ✅ Good: `"oscar,academy award,golden globe"` (award ceremonies overlap)
- ✅ Good: `"marvel,superhero,comic"` (related themes)
- ❌ Less ideal: `"horror,romantic comedy"` (very different, fewer connections)

### 3. Adjust for Your Use Case

- **For demos**: 30-50 movies is usually enough
- **For testing**: 100-200 movies provides good coverage
- **For benchmarking**: Keep the full dataset

### 4. Test with Sample Questions

After creating your demo dataset, test it with questions you know should work:

```bash
# Create a test questions file
cat > dataset/demo_questions.jsonl << 'EOF'
{"query": "Who won best actor Oscar in 2023?", "query_time": "2024-01-01T00:00:00"}
{"query": "Which animated film won best animated feature in 2024?", "query_time": "2024-03-15T00:00:00"}
EOF

# Run QA on demo questions
uv run --with mellea run/run_qa.py \
    --dataset dataset/demo_questions.jsonl \
    --num-workers 2
```

## Reverting to Full Database

To switch back to the full database:

```bash
# If you used Option A (directory swap):
mv dataset/movie dataset/movie_demo
mv dataset/movie_full dataset/movie

# If you used Option B (environment variable):
# Just update .env back to the original KG_BASE_DIRECTORY

# Then re-run preprocessing if needed
```

## Troubleshooting

### "No matching movies found"

Your topic filters might be too restrictive. Try:
- Broadening the year range
- Using more general topics
- Increasing `--max-movies`
- Removing the `--topics` filter entirely

### "Missing person/year references"

Make sure to use `--include-related` flag to include all referenced entities.

### "Demo dataset too large"

Reduce the size by:
- Narrowing the year range (e.g., 2023-2024 only)
- Lowering `--max-movies`
- Using more specific topics

### "Demo dataset too small"

Increase the size by:
- Expanding the year range
- Increasing `--max-movies`
- Using broader topics or removing topic filter

## Example Workflows

### Creating Multiple Demo Sets

You can create multiple demo sets for different purposes:

```bash
# Oscar-focused demo
uv run python create_demo_dataset.py \
    --topics "oscar" --max-movies 100 \
    --output-dir dataset/movie_oscar --include-related

# Marvel demo
uv run python create_demo_dataset.py \
    --topics "marvel" --max-movies 50 \
    --output-dir dataset/movie_marvel --include-related

# Recent releases
uv run python create_demo_dataset.py \
    --year-start 2023 --max-movies 30 \
    --output-dir dataset/movie_recent --include-related
```

Then switch between them as needed:

```bash
# Use Oscar demo
ln -sf movie_oscar dataset/movie

# Use Marvel demo
ln -sf movie_marvel dataset/movie
```

## Performance Impact

Processing time scales roughly linearly with dataset size:

| Movies | Preprocessing | Embedding | Update | Total |
|--------|--------------|-----------|--------|-------|
| 30 | 2 min | 1 min | 2 min | ~5 min |
| 100 | 5 min | 3 min | 7 min | ~15 min |
| 500 | 20 min | 12 min | 30 min | ~1 hour |
| 5,000 | 3 hours | 2 hours | 5 hours | ~10 hours |

Choose your dataset size based on your time constraints and testing needs.
