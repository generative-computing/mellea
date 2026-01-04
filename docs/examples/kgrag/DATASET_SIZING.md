# Dataset Sizing Guide

This document explains the different dataset sizes available for the KG-RAG example and when to use each.

## Available Dataset Sizes

### 1. Full Dataset (565 documents)
**File**: `dataset/crag_movie_dev.jsonl.bz2`
**Size**: 565 documents

**Use when:**
- Running final experiments
- Benchmarking performance
- Generating publication results
- Testing at scale

**Processing time:** 1-3 hours depending on hardware and LLM speed

### 2. Demo Dataset (100 movies → 20 movies)
**Created by**: `create_demo_dataset.py`
**Default**: 100 movies (configurable with `--max-movies`)
**Recommended**: 20 movies for faster testing

**Use when:**
- Demonstrating the system to others
- Initial development and debugging
- Testing new features
- Creating reproducible examples

**Processing time:** 10-30 minutes

### 3. Tiny Dataset (10 documents)
**Created by**: `create_tiny_dataset.py`
**File**: `dataset/crag_movie_tiny.jsonl.bz2`
**Size**: 10 documents (configurable with `--num-docs`)

**Use when:**
- Rapid iteration during development
- Testing bug fixes
- CI/CD pipelines
- Quick smoke tests

**Processing time:** 1-3 minutes

### 4. Micro Dataset (5 documents)
**Created by**: `create_tiny_dataset.py --num-docs 5`
**File**: Custom location
**Size**: 5 documents

**Use when:**
- Ultra-fast testing
- Debugging specific issues
- Testing error handling
- Validating API connections

**Processing time:** 30 seconds - 1 minute

## Creating Datasets

### Create Tiny Dataset (Recommended for Development)
```bash
# 10 documents (default)
python create_tiny_dataset.py --num-docs 10

# 20 documents (slightly larger)
python create_tiny_dataset.py --num-docs 20

# 5 documents (micro testing)
python create_tiny_dataset.py --num-docs 5
```

### Create Demo Dataset with Fewer Movies
```bash
# 20 movies (recommended for quick demos)
python create_demo_dataset.py --year-start 2022 --year-end 2024 --max-movies 20

# 50 movies (medium-sized demo)
python create_demo_dataset.py --year-start 2020 --year-end 2024 --max-movies 50

# 10 movies (minimal KG)
python create_demo_dataset.py --year-start 2023 --year-end 2024 --max-movies 10
```

## Running with Different Datasets

### Using run.sh (Automatic)
The `run.sh` script automatically uses smaller datasets if available:
- Creates 20-movie demo database
- Creates 10-document tiny dataset
- Uses tiny dataset for KG update

```bash
./run.sh
```

### Manual Control

**Preprocessing (KG structure)**:
```bash
# Uses movies in dataset/movie_demo/
python run/run_kg_preprocess.py --domain movie
```

**KG Update (document processing)**:
```bash
# Tiny dataset (10 docs)
python run/run_kg_update.py --dataset dataset/crag_movie_tiny.jsonl.bz2 --num-workers 1

# Full dataset (565 docs)
python run/run_kg_update.py --num-workers 64
```

**QA**:
```bash
# Small batch for testing
python run/run_qa.py --num-workers 1 --queue-size 1

# Full parallel processing
python run/run_qa.py --num-workers 64 --queue-size 64
```

## Dataset Size Comparison

| Dataset | Movies | Documents | KG Nodes | KG Edges | Processing Time |
|---------|--------|-----------|----------|----------|-----------------|
| Full | N/A | 565 | ~10,000+ | ~50,000+ | 1-3 hours |
| Demo (100) | 100 | Variable | ~2,000 | ~10,000 | 10-30 min |
| Demo (20) | 20 | Variable | ~400 | ~2,000 | 5-10 min |
| Tiny | N/A | 10 | ~100 | ~500 | 1-3 min |
| Micro | N/A | 5 | ~50 | ~250 | 30-60 sec |

*Note: Actual numbers depend on data content and complexity*

## Recommended Workflow

### 1. Initial Development
```bash
# Create micro dataset for ultra-fast testing
python create_tiny_dataset.py --num-docs 5

# Test with micro dataset
python run/run_kg_update.py --dataset dataset/crag_movie_tiny.jsonl.bz2 --num-workers 1
```

### 2. Feature Testing
```bash
# Create tiny dataset
python create_tiny_dataset.py --num-docs 10

# Run full pipeline quickly
./run.sh
```

### 3. Integration Testing
```bash
# Create small demo dataset
python create_demo_dataset.py --max-movies 20

# Run with tiny documents
./run.sh
```

### 4. Final Validation
```bash
# Use full dataset
python run/run_kg_update.py --num-workers 64

# Or modify run.sh to skip tiny dataset creation
```

## Environment Variables

You can control dataset locations via environment variables:

```bash
# Custom dataset directory
export KG_BASE_DIRECTORY="$(pwd)/dataset_custom"

# Use in scripts
python run/run_kg_preprocess.py
```

## Tips for Fast Iteration

1. **Always start with tiny datasets** (5-10 documents) during development
2. **Use Mellea-native versions** for better error messages and debugging
3. **Monitor token usage** to estimate costs before scaling up
4. **Cache KG preprocessing results** - only rerun when schema changes
5. **Use `--verbose` flag** to see detailed progress

## Troubleshooting

**Q: KG update is still slow with tiny dataset**
- Check `--num-workers` (use 1 for debugging)
- Verify LLM API latency
- Check if using local vs remote LLM

**Q: Demo dataset creation fails**
- Ensure `dataset/movie/` contains full database files
- Check disk space
- Verify JSON file integrity

**Q: Tiny dataset doesn't have enough variety**
- Increase `--num-docs` to 20 or 30
- Or manually select specific documents from the full dataset

**Q: Results differ between dataset sizes**
- This is expected - smaller datasets have less context
- For reproducible results, use the same dataset size
- Document dataset size in experiment notes
