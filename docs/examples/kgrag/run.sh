#!/bin/bash

# Exit on error
set -e

# Change to the script's directory to ensure correct module paths
cd "$(dirname "$0")"

# Set PYTHONPATH to include the current directory so Python can find kg, utils, etc.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set KG_BASE_DIRECTORY to the dataset directory relative to current location
export KG_BASE_DIRECTORY="$(pwd)/dataset"

# Disable OpenTelemetry if OTEL collector is not available
# This prevents "connection refused" errors to port 3000
export OTEL_SDK_DISABLED=true

echo "=================================================="
echo "KGRAG Pipeline Execution"
echo "=================================================="

# Step 1: Empty the Neo4j database if it exists
echo ""
echo "Step 1: Cleaning Neo4j database..."
if command -v cypher-shell &> /dev/null; then
    # Load Neo4j credentials from environment
    NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"
    if [ -n "$NEO4J_PASSWORD" ]; then
        echo "Clearing all nodes and relationships from Neo4j..."
        cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) DETACH DELETE n" || echo "Warning: Failed to clear database (it may already be empty)"
    else
        echo "Warning: NEO4J_PASSWORD not set, skipping database cleanup"
    fi
else
    echo "Warning: cypher-shell not found, skipping database cleanup"
    echo "You can manually clear the database with: MATCH (n) DETACH DELETE n"
fi

# Step 2: Create the demo datasets
echo ""
echo "Step 2: Creating demo datasets..."
# Create a smaller KG database (20 movies instead of 100)
if [ -f "run/create_demo_dataset.py" ]; then
    echo "Creating small movie database (20 movies)..."
    uv run --with mellea run/create_demo_dataset.py --year-start 2022 --year-end 2024 --max-movies 20
else
    echo "Warning: run/create_demo_dataset.py not found, skipping demo dataset creation"
fi

# Create a tiny document dataset (10 documents instead of 565)
if [ -f "run/create_tiny_dataset.py" ]; then
    echo "Creating tiny document dataset (10 documents)..."
    uv run --with mellea run/create_tiny_dataset.py --num-docs 10

    # Optionally truncate documents to 50k chars for faster processing
    if [ -f "dataset/crag_movie_tiny.jsonl.bz2" ]; then
        echo "Truncating documents to 50k chars for faster processing..."
        python3 run/create_truncated_dataset.py --input dataset/crag_movie_tiny.jsonl.bz2 --output dataset/crag_movie_tiny_truncated.jsonl.bz2 --max-chars 50000
    fi
else
    echo "Warning: run/create_tiny_dataset.py not found, will use full dataset"
fi

# Step 3: Run preprocessing
echo ""
echo "Step 3: Running KG preprocessing..."
uv run --with mellea run/run_kg_preprocess.py

# Step 4: Run KG embedding
echo ""
echo "Step 4: Running KG embedding..."
uv run --with mellea run/run_kg_embed.py

# Step 5: Run KG update (using truncated tiny dataset if available)
echo ""
echo "Step 5: Running KG update..."
TRUNCATED_DATASET="dataset/crag_movie_tiny_truncated.jsonl.bz2"
TINY_DATASET="dataset/crag_movie_tiny.jsonl.bz2"

if [ -f "$TRUNCATED_DATASET" ]; then
    echo "Using truncated tiny dataset: $TRUNCATED_DATASET"
    uv run --with mellea run/run_kg_update.py --dataset "$TRUNCATED_DATASET" --num-workers 1 --queue-size 1
elif [ -f "$TINY_DATASET" ]; then
    echo "Using tiny dataset: $TINY_DATASET"
    uv run --with mellea run/run_kg_update.py --dataset "$TINY_DATASET" --num-workers 1 --queue-size 1
else
    echo "Tiny dataset not found, using default dataset"
    uv run --with mellea run/run_kg_update.py --num-workers 64 --queue-size 64
fi

# Step 6: Run QA
echo ""
echo "Step 6: Running QA..."
uv run --with mellea run/run_qa.py --num-workers 1 --queue-size 1

# Step 7: Run eval if QA did not already call it
echo ""
echo "Step 7: Checking if evaluation is needed..."
# Check if the results file exists and contains evaluation scores
RESULTS_FILE="results/_results.json"
if [ -f "$RESULTS_FILE" ]; then
    # Check if the results file contains a "score" field (indicating eval was already run)
    if grep -q '"score"' "$RESULTS_FILE"; then
        echo "Evaluation already completed in QA step, skipping separate eval"
    else
        echo "Running separate evaluation..."
        uv run --with mellea run/run_eval.py --result-path "$RESULTS_FILE"
    fi
else
    echo "Warning: Results file not found at $RESULTS_FILE"
    echo "Running evaluation anyway..."
    uv run --with mellea run/run_eval.py --result-path "$RESULTS_FILE"
fi

echo ""
echo "=================================================="
echo "Pipeline execution completed successfully!"
echo "=================================================="