# Using Mellea with Bedrock

This directory demonstrates how to use Mellea with Bedrock, AWS's model inferencing platform.

## Pre-requisites

To get started you will need a to set the `AWS_BEARER_TOKEN_BEDROCK` environment variable and install the optional `aws` dependencies:

```python
export AWS_BEARER_TOKEN_BEDROCK=<your API key goes here>
uv pip install mellea[aws,litellm]
```

## Running the example

You can then run the example using:

```python
python bedrock_session_example.py
```