#!/bin/bash

# Exit on error
set -e

# Change to the script's directory to ensure correct module paths
cd "$(dirname "$0")"

# Set PYTHONPATH to include the current directory so Python can find kg, utils, etc.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set KG_BASE_DIRECTORY to the dataset directory relative to current location
export KG_BASE_DIRECTORY="$(pwd)/dataset"

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

# Step 2: Create the demo dataset
echo ""
echo "Step 2: Creating demo dataset..."
if [ -f "create_demo_dataset.py" ]; then
    uv run --with mellea create_demo_dataset.py --year-start 2020 --year-end 2024 --max-movies 100
else
    echo "Warning: create_demo_dataset.py not found, skipping demo dataset creation"
fi

# Step 3: Run preprocessing
echo ""
echo "Step 3: Running KG preprocessing..."
uv run --with mellea run/run_kg_preprocess.py

# Step 4: Run KG embedding
echo ""
echo "Step 4: Running KG embedding..."
uv run --with mellea run/run_kg_embed.py

# Step 5: Run KG update
echo ""
echo "Step 5: Running KG update..."
uv run --with mellea run/run_kg_update.py --num-workers 1 --queue-size 1

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