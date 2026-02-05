# Using Mellea with Bedrock

Mellea can be used with Bedrock models via Mellea's LiteLLM or OpenAI backends.

## Pre-requisites

To get started you will need a to set the `AWS_BEARER_TOKEN_BEDROCK` environment variable and install the optional `aws` dependencies:

```python
export AWS_BEARER_TOKEN_BEDROCK=<your API key goes here>
uv pip install mellea[litellm]
```

## Running the example

You can then run the examples using:

```python
uv run bedrock_litellm_example.py
```

or

```python
uv run bedrock_openai_example.py
```