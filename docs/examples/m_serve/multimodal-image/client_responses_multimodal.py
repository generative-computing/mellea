# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example: Using multimodal inputs with the Responses API endpoint.

This example demonstrates how to send images along with text prompts
to the /v1/responses endpoint, which supports multimodal content blocks.

Prerequisites:
    - Server running with a multimodal-capable model (e.g., LLaVA)
    - OpenAI Python client installed

Usage:
    uv run python client_responses_multimodal.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.responses.create(
    model="llava",
    input=[
        {  # type: ignore[list-item, misc]
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What's in this image?"},
                {
                    "type": "input_image",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            ],
        }
    ],
)  # type: ignore[call-overload]

print(response.output_text)

response = client.responses.create(
    model="llava",
    instructions=[
        {  # type: ignore[list-item, misc]
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "You are a helpful assistant that analyzes images in detail.",
                }
            ],
        }
    ],
    input=[
        {  # type: ignore[list-item, misc]
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this scene."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                },
            ],
        }
    ],
)  # type: ignore[call-overload]

print(response.output_text)
