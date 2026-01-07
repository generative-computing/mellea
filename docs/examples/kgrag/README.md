# KGRag: Knowledge Graph-Enhanced RAG with Mellea

This example demonstrates a Knowledge Graph-enhanced Retrieval-Augmented Generation (KG-RAG) system built with the Mellea framework, adapted from the [Bidirection](https://github.com/junhongmit/Bidirection) project for temporal reasoning over movie domain knowledge.

## 🎉 What's New - Fully Refactored!

This codebase has been **completely refactored** to follow Mellea's design patterns and modern Python best practices:

- ✅ **Type-Safe Configuration**: Pydantic models with automatic validation
- ✅ **Modern Async Patterns**: Python 3.7+ `asyncio.run()` instead of manual event loops
- ✅ **Factory Functions**: Clean session creation with intelligent defaults
- ✅ **Comprehensive CLI**: Rich argparse with examples and help text
- ✅ **Better Error Handling**: Proper exit codes (0=success, 1=error, 130=interrupt)
- ✅ **Robust Code**: Graceful handling of edge cases and missing data
- ✅ **Clean File Structure**: Removed `_refactored` suffixes, single source of truth
- ✅ **Full Documentation**: Detailed refactoring guides for each component

**Documentation:**
- [MELLEA_INTEGRATION.md](MELLEA_INTEGRATION.md) - Mellea patterns showcase with code examples for all pipeline components
- [DEVELOPMENT_SUMMARY.md](DEVELOPMENT_SUMMARY.md) - Complete development history, bug fixes, and migration details
- [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) - Comprehensive refactoring patterns and best practices

## Overview

KGRag combines the power of Knowledge Graphs with Large Language Models to answer complex questions that require multi-hop reasoning over structured knowledge. The system uses a Neo4j graph database to store and query entities, relationships, and temporal information, enabling more accurate and explainable answers compared to traditional RAG approaches.

### What Problem Does It Solve?

Traditional LLMs and RAG systems struggle with:
- **Multi-hop reasoning**: Questions requiring multiple inference steps
- **Temporal reasoning**: Questions involving time-sensitive information
- **Structured relationships**: Understanding complex entity relationships
- **Knowledge provenance**: Providing explainable reasoning paths

KGRag addresses these challenges by:
1. **Knowledge Graph Construction**: Building a structured graph from unstructured documents
2. **Bidirectional Search**: Traversing relationships in both forward and backward directions
3. **Temporal-Aware Reasoning**: Incorporating query time and temporal constraints
4. **Multi-Route Exploration**: Breaking down complex questions into multiple solving routes

## Architecture

The system consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           KGRagComponent (kg/kg_rag.py)                     │
│  • Question breakdown into solving routes                   │
│  • Topic entity extraction                                  │
│  • Entity alignment with KG                                 │
│  • Multi-hop graph traversal                                │
│  • Answer synthesis and validation                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Neo4j Knowledge Graph                          │
│  • Entities (Movies, Awards, Persons, etc.)                 │
│  • Relations (WON, NOMINATED_FOR, PRODUCED, etc.)           │
│  • Properties (temporal info, descriptions)                 │
│  • Vector embeddings for similarity search                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Answer + Reasoning Path                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**Core Modules:**
- **[kg/kg_rag.py](kg/kg_rag.py)**: KGRagComponent implementing the reasoning pipeline following Mellea patterns
- **[kg/kg_driver.py](kg/kg_driver.py)**: Neo4j database driver for graph operations
- **[kg/kg_preprocessor.py](kg/kg_preprocessor.py)**: Entity and relation extraction from structured databases
- **[kg/kg_embedder.py](kg/kg_embedder.py)**: Embedding generation with batch processing
- **[kg/kg_updater_component.py](kg/kg_updater_component.py)**: Incremental graph updates with document processing

**Configuration Models (Pydantic):**
- **[kg/kg_entity_models.py](kg/kg_entity_models.py)**: Type-safe entity models (Movie, Person, Award, etc.)
- **[kg/kg_embed_models.py](kg/kg_embed_models.py)**: Embedding configuration and validation
- **[kg/kg_updater_models.py](kg/kg_updater_models.py)**: Updater configuration models
- **[kg/kg_qa_models.py](kg/kg_qa_models.py)**: QA configuration models

**Run Scripts:**
- **[run/run_kg_preprocess.py](run/run_kg_preprocess.py)**: Preprocessing with modern async patterns
- **[run/run_kg_embed.py](run/run_kg_embed.py)**: Embedding generation script
- **[run/run_kg_update.py](run/run_kg_update.py)**: Graph update with comprehensive CLI
- **[run/run_qa.py](run/run_qa.py)**: QA evaluation with factory functions and proper exit codes

**Utilities:**
- **[dataset/movie_dataset.py](dataset/movie_dataset.py)**: Movie domain dataset loader
- **[demo/demo.py](demo/demo.py)**: Complete demo showing KGRag usage

**Data Preparation Scripts:**
- **[run/create_demo_dataset.py](run/create_demo_dataset.py)**: Create smaller demo KG database
- **[run/create_tiny_dataset.py](run/create_tiny_dataset.py)**: Create tiny document dataset for testing
- **[run/create_truncated_dataset.py](run/create_truncated_dataset.py)**: Truncate documents for faster processing

## Prerequisites

### System Requirements

- Python 3.9+
- Neo4j 5.x or later
- 8GB+ RAM (16GB+ recommended)
- GPU recommended for faster embedding generation

### Required Software

1. **Neo4j Database**
   ```bash
   # Install Neo4j Desktop or use Docker
   docker run \
       --name neo4j \
       -p7474:7474 -p7687:7687 \
       -e NEO4J_AUTH=neo4j/your_password \
       -e NEO4J_PLUGINS='["apoc"]' \
       neo4j:latest
   ```

2. **Python Dependencies**
   ```bash
   # Install Mellea and dependencies
   uv sync --all-extras --all-groups

   # Or install specific dependencies
   pip install neo4j python-dotenv beautifulsoup4 trafilatura
   pip install sentence-transformers  # For local embeddings
   ```

### Neo4j Configuration

After starting Neo4j, you need to create vector indices:

```cypher
// Create vector index for entity embeddings
CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
FOR (n:Entity)
ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 512,
  `vector.similarity_function`: 'cosine'
}};

// Create index for entity names (for fuzzy search)
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);
```

## Setup

### 1. Environment Configuration

Create a `.env` file in the `kgrag` directory based on the .env_template

### 2. Dataset Preparation

This example uses the **CRAG (Comprehensive RAG) Benchmark** for evaluation. The knowledge graph is built from movie domain data including structured databases and question-answer pairs.

#### Download CRAG Benchmark and Mock API

```bash
# Navigate to the kgrag directory
cd docs/examples/kgrag

# Clone the CRAG Benchmark repository
# Note: You may need to install Git LFS to properly download all datasets
git lfs install
git clone https://github.com/facebookresearch/CRAG.git

# Copy the mock_api folder to the dataset directory
# The mock_api contains the knowledge graph databases (movie_db.json, person_db.json, year_db.json)
# These files are essential for building the knowledge graph
cp -r CRAG/mock_api/movie dataset/movie

# Download the CRAG movie dataset (questions and answers)
cd dataset
# The dataset file should be named crag_movie_dev.jsonl or crag_movie_dev.jsonl.bz2
# If compressed, extract it:
bunzip2 crag_movie_dev.jsonl.bz2  # if .bz2 format
```

#### Dataset Structure

After setup, your dataset directory should contain:

```
dataset/
├── crag_movie_dev.jsonl          # Questions and answers
└── movie/                         # Mock API databases
    ├── movie_db.json             # Movie entity database
    ├── person_db.json            # Person entity database
    └── year_db.json              # Year/temporal database
```

**JSONL Dataset Format**: Each line in `crag_movie_dev.jsonl` contains:
- `domain`: "movie"
- `query`: The question to answer
- `query_time`: Timestamp of the query
- `search_results`: List of web pages with content
- `answer`: Ground truth answer
- `interaction_id`: Unique identifier

**Mock API Format**: The `*_db.json` files contain structured knowledge graph data:
- `movie_db.json`: Movie entities with properties (title, release date, cast, awards, etc.)
- `person_db.json`: Person entities (actors, directors, producers, etc.)
- `year_db.json`: Temporal information and year-specific events

#### Creating a Demo Dataset (Optional but Recommended)

The full database is quite large (225MB+). For faster demos and testing, create a smaller focused dataset:

```bash
# Create a demo dataset with ~100 recent movies (2020-2024)
cd docs/examples/kgrag
uv run python run/create_demo_dataset.py \
    --year-start 2020 \
    --year-end 2024 \
    --max-movies 100 \
    --topics "oscar,academy award" \
    --include-related

# Switch to the demo dataset
mv dataset/movie dataset/movie_full
mv dataset/movie_demo dataset/movie
```

**Benefits of using a demo dataset:**
- ⚡ **10-20x faster processing** (15-20 minutes vs 4-6 hours)
- 💾 **95% smaller** (~5MB vs 225MB)
- 🎯 **Focused testing** with coherent topic clusters
- 🚀 **Quick iteration** for development and demos

#### Document Truncation for Faster Processing

For even faster KG updates during development, truncate long documents to reduce processing time:

```bash
# Truncate documents to 50k characters (88.9% size reduction)
python3 run/create_truncated_dataset.py \
  --input dataset/crag_movie_tiny.jsonl.bz2 \
  --output dataset/crag_movie_tiny_truncated.jsonl.bz2 \
  --max-chars 50000
```

**Benefits:**
- ⚡ **80-90% faster processing** - Less text to extract entities from
- 💰 **Lower API costs** - Fewer tokens sent to LLM
- 🎯 **Smart truncation** - Ends at sentence boundaries, preserves context
- 📦 **Automatic usage** - `run.sh` uses truncated dataset if available

**Recommended settings:**
| Dataset | max-chars | Use Case |
|---------|-----------|----------|
| Tiny (10 docs) | 30k-50k | Quick testing, debugging |
| Dev (565 docs) | 50k-100k | Development, experimentation |
| Full dataset | 100k-200k | Production (or no truncation) |

### 3. Knowledge Graph Construction

Build the knowledge graph from the dataset:

```bash
# Set up environment
cd docs/examples/kgrag
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export KG_BASE_DIRECTORY="$(pwd)/dataset"

# Step 1: Preprocess documents and extract entities/relations
uv run --with mellea run/run_kg_preprocess.py

# Step 2: Generate embeddings for entities
uv run --with mellea run/run_kg_embed.py

# Step 3: Update the knowledge graph with extracted information
uv run --with mellea run/run_kg_update.py --num-workers 4 --queue-size 10
```

**Note**: The preprocessing and graph construction can take several hours depending on dataset size and hardware.

## Usage

### Running Question Answering

After building the knowledge graph, run QA inference:

```bash
# Run with default settings
uv run --with mellea run/run_qa.py --num-workers 4 --queue-size 10

# Run with custom configuration
uv run --with mellea run/run_qa.py \
    --num-workers 8 \
    --queue-size 16 \
    --config route=3 width=20 depth=2 \
    --prefix my_experiment \
    --postfix v1 \
    --verbose

# Run with specific dataset
uv run --with mellea run/run_qa.py \
    --dataset dataset/custom_questions.jsonl \
    --domain movie \
    --eval-batch-size 64 \
    --eval-method llama
```

**Parameters:**
- `--dataset`: Path to dataset file (default: uses KG_BASE_DIRECTORY)
- `--domain`: Knowledge domain (default: movie)
- `--num-workers`: Number of parallel workers for inference (default: 128)
- `--queue-size`: Size of the data loading queue (default: 128)
- `--split`: Dataset split index (default: 0)
- `--config`: Override model configuration (e.g., `route=5 width=30 depth=3`)
  - `route`: Number of solving routes to explore (default: 5)
  - `width`: Maximum number of relations to consider at each step (default: 30)
  - `depth`: Maximum graph traversal depth (default: 3)
- `--prefix`: Prefix for output file names
- `--postfix`: Postfix for output file names
- `--keep`: Keep progress file after completion
- `--eval-batch-size`: Batch size for evaluation (default: 64)
- `--eval-method`: Evaluation method (default: llama)
- `--verbose` or `-v`: Enable verbose logging

### Using the Convenience Script

```bash
# Edit run.sh to uncomment the desired step
bash run.sh
```

### Interactive Demo (Optional)

For a quick demonstration of the KGRag pipeline with example queries:

```bash
# Run the interactive demo
uv run --with mellea python demo/demo.py
```

**Note**: The demo is a standalone demonstration tool separate from the main QA evaluation pipeline. It's useful for:
- Understanding how KGRag works with example queries
- Testing the system with custom questions interactively
- Debugging and exploring the reasoning process

For production use and benchmark evaluation, use `run/run_qa.py` instead.

## How It Works

The KGRag system follows a multi-step reasoning pipeline:

### 1. Question Breakdown
The system breaks down complex questions into multiple solving routes:

```
Question: "Which animated film won the best animated feature Oscar in 2024?"

Routes:
1. ["Identify 2024 Oscars best animated feature award", "Find the winner"]
2. ["List 2024 Oscar nominees", "Filter animated features", "Identify winner"]
3. ["Search for 2024 Oscar results", "Extract best animated feature winner"]
```

### 2. Topic Entity Extraction
Extract relevant entities from the question considering entity types:

```
Extracted: ["2024 Oscars best animated feature award"]
Entity Type: Award
```

### 3. Entity Alignment
Align extracted entities with knowledge graph entities using:
- Fuzzy string matching for exact name matches
- Vector similarity search for semantic matching

### 4. Multi-Hop Graph Traversal
For each aligned entity, traverse the graph to find relevant information:

```
Depth 0: Start entity → (Award: 2024 OSCARS BEST ANIMATED FEATURE)
Depth 1: Find relations → [WON, NOMINATED_FOR]
Depth 2: Follow WON relation → (Movie: THE BOY AND THE HERON)
```

At each depth:
- **Relation Pruning**: Select relevant relation types using LLM
- **Triplet Pruning**: Score individual relation instances
- **Relevance Scoring**: Rank entities and relations by relevance

### 5. Answer Synthesis
Synthesize the final answer using:
- Retrieved entities and relations
- Multi-route validation for consensus
- Temporal alignment verification

## Output Format

Results are saved to `results/*_results.json`:

```json
[
  {
    "accuracy": 0.85,
    "inf_prompt_tokens": 125000,
    "inf_completion_tokens": 15000,
    "eval_prompt_tokens": 50000,
    "eval_completion_tokens": 5000
  },
  {
    "id": 0,
    "query": "Which animated film won the best animated feature Oscar in 2024?",
    "query_time": "03/19/2024, 23:49:30 PT",
    "ans": "The Boy and the Heron",
    "prediction": "The Boy and the Heron",
    "processing_time": 12.34,
    "token_usage": {
      "prompt_tokens": 2500,
      "completion_tokens": 150
    },
    "score": 1.0,
    "explanation": "The prediction correctly identifies the winner..."
  }
]
```

### LLM-Powered Functions
The system extensively uses LLM calls for:
- Question decomposition
- Entity extraction and alignment
- Relation pruning and scoring
- Answer synthesis and validation

## Performance Optimization

### Parallel Processing
The system supports parallel processing with configurable workers for efficient batch processing. See `run/run_qa.py` for the production implementation.

### Caching
- Neo4j vector indices for fast similarity search
- Schema caching for reduced database queries
- Entity/relation caching during traversal

### Resource Management
- Configure `--num-worker` and `--queue-size` based on available resources
- Use local embedding models to reduce API costs
- Adjust `route`, `width`, and `depth` for speed/accuracy tradeoffs

## Troubleshooting

### Common Issues

**Neo4j Connection Error**
```
neo4j.exceptions.ServiceUnavailable: Unable to connect to localhost:7687
```
- Ensure Neo4j is running: `docker ps` or check Neo4j Desktop
- Verify NEO4J_PASSWORD in `.env` matches your database password
- Check firewall settings allow port 7687

**Out of Memory**
- Reduce `--num-worker` and `--queue-size`
- Reduce `width` parameter in config
- Use a machine with more RAM or enable swap

**Slow Inference**
- Use GPU for embedding generation
- Increase `--num-worker` for parallel processing
- Use local models instead of API calls
- Reduce `route` and `depth` parameters

**Empty Knowledge Graph**
- Verify dataset path in environment: `echo $KG_BASE_DIRECTORY`
- Check Neo4j for entities: `MATCH (n) RETURN count(n)`
- Re-run preprocessing: `run/run_kg_preprocess.py`

**Import Errors**
```
ModuleNotFoundError: No module named 'kg'
```
- Ensure PYTHONPATH is set: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
- Run from the `kgrag` directory

## Limitations

- **Domain-Specific**: Currently optimized for movie domain; requires prompt adaptation for other domains
- **Cold Start**: Requires pre-built knowledge graph or documents to update the knowledge graph; cannot answer questions about entities not in the graph
- **Computational Cost**: Multi-hop graph traversal and multiple LLM calls can be expensive
- **English-Only**: Prompts and evaluation are in English


