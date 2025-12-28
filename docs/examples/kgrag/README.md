# KGRag: Knowledge Graph-Enhanced RAG with Mellea

This example demonstrates a Knowledge Graph-enhanced Retrieval-Augmented Generation (KG-RAG) system built with the Mellea framework, adapted from the [Bidirection](https://github.com/junhongmit/Bidirection) project for temporal reasoning over movie domain knowledge.

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
│              KGModel (kg_model.py)                          │
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

- **[kg_model.py](kg_model.py)**: Core KGModel class implementing the reasoning pipeline
- **[kg/kg_driver.py](kg/kg_driver.py)**: Neo4j database driver for graph operations
- **[kg/kg_preprocessor.py](kg/kg_preprocessor.py)**: Entity and relation extraction from documents
- **[kg/kg_updater.py](kg/kg_updater.py)**: Incremental graph updates with new information
- **[dataset/movie_dataset.py](dataset/movie_dataset.py)**: Movie domain dataset loader
- **[eval.py](eval.py)**: LLM-as-a-judge evaluation framework

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
uv run --with mellea run/run_kg_update.py --num-worker 4 --queue-size 10
```

**Note**: The preprocessing and graph construction can take several hours depending on dataset size and hardware.

## Usage

### Running Question Answering

After building the knowledge graph, run QA inference:

```bash
# Run with default settings
uv run --with mellea run/run_qa.py --num-worker 4 --queue-size 10

# Run with custom configuration
uv run --with mellea run/run_qa.py \
    --num-worker 8 \
    --queue-size 16 \
    --config route=3 width=20 depth=2 \
    --prefix my_experiment \
    --postfix v1
```

**Parameters:**
- `--num-worker`: Number of parallel workers for inference
- `--queue-size`: Size of the data loading queue
- `--config`: Override model configuration (e.g., `route=5 width=30 depth=3`)
  - `route`: Number of solving routes to explore (default: 5)
  - `width`: Maximum number of relations to consider at each step (default: 30)
  - `depth`: Maximum graph traversal depth (default: 3)
- `--prefix`: Prefix for output file names
- `--postfix`: Postfix for output file names
- `--keep`: Keep progress file after completion

### Using the Convenience Script

```bash
# Edit run.sh to uncomment the desired step
bash run.sh
```

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

## Example Queries

```python
from kg_model import KGModel
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from datetime import datetime

# Initialize sessions
session = MelleaSession(backend=OpenAIBackend(
    model_id="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
))

# Create KG model
kg_model = KGModel(
    session=session,
    eval_session=session,
    emb_session=embedding_model,
    domain="movie",
    config={"route": 5, "width": 30, "depth": 3}
)

# Generate answer
answer = await kg_model.generate_answer(
    query="Who won the best actor Oscar in 2020?",
    query_time=datetime(2024, 3, 19, 23, 49, 30)
)

print(answer)
# Output: "Joaquin Phoenix won the Best Actor Oscar in 2020 for his role in Joker."
```

## Integration with Mellea

This example demonstrates several Mellea framework patterns:

### Backend Configuration
```python
from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

session = MelleaSession(backend=OpenAIBackend(
    model_id=MODEL_NAME,
    base_url=API_BASE,
    api_key=API_KEY,
    timeout=TIME_OUT
))
```

### LLM-Powered Functions
The system extensively uses LLM calls for:
- Question decomposition
- Entity extraction and alignment
- Relation pruning and scoring
- Answer synthesis and validation

### Retry Logic
All LLM calls use retry decorators for robustness:
```python
@llm_retry(max_retries=MAX_RETRIES, default_output=[])
async def extract_entity(self, query: Query) -> List[str]:
    # LLM call with automatic retries
    pass
```

## Evaluation

The system includes an LLM-as-a-judge evaluation framework:

```python
from eval import evaluate_predictions

stats, history = evaluate_predictions(
    queries=queries,
    ground_truths_list=ground_truths,
    predictions=predictions,
    model_type='llama',
    batch_size=64
)

print(f"Accuracy: {stats['accuracy']}")
print(f"Average Score: {stats['average_score']}")
```

## Performance Optimization

### Parallel Processing
The system supports parallel processing at multiple levels:

```python
# Parallel route exploration
tasks = [explore_one_route(route) for route in queries[:3]]
results = await asyncio.gather(*tasks)

# Parallel relation search
tasks = [self.relation_search_prune(route, entity)
         for entity in entities]
results = await asyncio.gather(*tasks)
```

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
- **Cold Start**: Requires pre-built knowledge graph; cannot answer questions about entities not in the graph
- **Computational Cost**: Multi-hop graph traversal and multiple LLM calls can be expensive
- **English-Only**: Prompts and evaluation are in English

## Future Enhancements

See [TODO.md](TODO.md) for planned improvements:
- Dockerization for easier deployment
- Support for additional domains (finance, scientific literature, etc.)
- Integration with more Mellea patterns (Requirements, Sampling Strategies)
- Improved prompt templates using Jinja2
- Web UI for interactive querying
- Benchmark comparisons with vanilla RAG

## Citation

If you use this example in your research, please cite:

```bibtex
@software{mellea_kgrag_2024,
  title={KGRag: Knowledge Graph-Enhanced RAG with Mellea},
  author={Mellea Team},
  year={2024},
  url={https://github.com/IBM/mellea}
}
```

This implementation is adapted from the Bidirection project for temporal knowledge graph reasoning.

### CRAG Benchmark

This example uses the CRAG (Comprehensive RAG Benchmark) dataset:

```bibtex
@article{yang2024crag,
  title={CRAG -- Comprehensive RAG Benchmark},
  author={Yang, Xiao and Yue, Kai and Zhang, Haotian and Fan, Zhiyuan and Xu, Wenhao and others},
  journal={arXiv preprint arXiv:2406.04744},
  year={2024},
  url={https://github.com/facebookresearch/CRAG}
}
```

For more information about the CRAG benchmark, visit: https://github.com/facebookresearch/CRAG

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This example is provided under the same license as the Mellea framework.

## Support

For questions and issues:
- Open an issue on GitHub
- Check the Mellea documentation
- Join the Mellea community discussions
