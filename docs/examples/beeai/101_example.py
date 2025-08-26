#!/usr/bin/env python3
"""
BeeAI Backend Example

This example demonstrates how to use the BeeAI backend with Mellea.
You'll need to have the beeai-framework installed and configured.
"""

import os

from mellea.backends.beeai import BeeAIBackend
from mellea.stdlib.base import CBlock
from mellea.stdlib.session import MelleaSession


def main():
    """Demonstrate basic BeeAI backend usage."""

    # Initialize the BeeAI backend
    # For local testing with Ollama, no API key is needed
    base_url = os.getenv("BEEAI_BASE_URL", "http://localhost:11434")
    model_id = os.getenv("BEEAI_MODEL_ID", "granite3.3:2b")

    # Create the backend
    from mellea.backends.formatter import TemplateFormatter

    formatter = TemplateFormatter(model_id=model_id)
    backend = BeeAIBackend(model_id=model_id, formatter=formatter, base_url=base_url)

    # Create a session with the backend
    session = MelleaSession(backend=backend)

    # Simple text generation
    print("🤖 Generating text with BeeAI...")
    result = session.backend.generate_from_context(
        action=CBlock("Write a short poem about AI"), ctx=session.ctx
    )

    print(f"📝 Generated text:\n{result.value}\n")

    # Generate with model options
    print("🎛️  Generating with temperature control...")
    result_with_options = session.backend.generate_from_context(
        action=CBlock("Write a creative story about a robot"),
        ctx=session.ctx,
        model_options={"temperature": 0.8, "max_tokens": 200},
    )

    print(f"📖 Creative story:\n{result_with_options.value}\n")

    # Generate with structured output
    print("🔧 Generating structured output...")
    from pydantic import BaseModel

    class Story(BaseModel):
        title: str
        characters: list[str]
        plot: str

    structured_result = session.backend.generate_from_context(
        action=CBlock("Create a story outline"), ctx=session.ctx, format=Story
    )

    print(f"📋 Structured story:\n{structured_result.parsed_repr}\n")

    print("✅ BeeAI backend example completed successfully!")
    print("\n📝 Note: This example demonstrates the backend structure.")
    print(
        "   For production use, ensure proper API configuration and model availability."
    )


if __name__ == "__main__":
    main()
