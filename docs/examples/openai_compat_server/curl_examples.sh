#!/bin/bash
# Example curl commands for the OpenAI-compatible server

# Base URL for the server
BASE_URL="http://localhost:8000"

echo "=== OpenAI-Compatible Server Examples ==="
echo ""

# 1. Health check
echo "1. Health Check:"
curl -s "${BASE_URL}/health" | jq .
echo ""

# 2. List models
echo "2. List Models:"
curl -s "${BASE_URL}/v1/models" | jq .
echo ""

# 3. Basic chat completion
echo "3. Basic Chat Completion:"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite4:micro",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
  }' | jq .
echo ""

# 4. Chat completion with system message
echo "4. Chat Completion with System Message:"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite4:micro",
    "messages": [
      {"role": "system", "content": "You are a helpful math tutor."},
      {"role": "user", "content": "Explain what a prime number is."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }' | jq .
echo ""

# 5. Streaming chat completion
echo "5. Streaming Chat Completion:"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite4:micro",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "stream": true,
    "max_tokens": 50
  }'
echo ""

# 6. Chat completion with seed for reproducibility
echo "6. Chat Completion with Seed:"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite4:micro",
    "messages": [
      {"role": "user", "content": "Pick a random number"}
    ],
    "seed": 42,
    "max_tokens": 20
  }' | jq .
echo ""

# 7. Chat completion with custom parameters
echo "7. Chat Completion with Custom Parameters:"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite4:micro",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding"}
    ],
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 100,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.5
  }' | jq .
echo ""

echo "=== All examples completed ==="
