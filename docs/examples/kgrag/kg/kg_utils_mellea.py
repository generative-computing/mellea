"""Simplified utils using Mellea's built-in features."""
import asyncio
from typing import Any, List
from mellea import MelleaSession
from utils.logger import logger


async def generate_embedding_mellea(
    session: Any,
    texts: List[str],
    **kwargs
) -> List:
    """Generate embeddings using Mellea session or local model.

    Args:
        session: Either an OpenAI-compatible client or SentenceTransformer model
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    if len(texts) == 0:
        return []

    try:
        # Try OpenAI-compatible API
        if hasattr(session, "embeddings"):
            responses = await session.embeddings.create(
                input=texts,
                **kwargs
            )
            return [data.embedding for data in responses.data]
        # Try SentenceTransformer model
        elif hasattr(session, "encode"):
            embeddings = session.encode(
                sentences=texts,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        else:
            logger.error("Embedding session does not support embeddings or encode method")
            return []
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return []


async def chat_with_mellea(
    session: MelleaSession,
    messages: List[dict],
    max_tokens: int = 8192,
    temperature: float = 0.1,
    **kwargs
) -> str:
    """Chat with LLM using Mellea session.

    Args:
        session: MelleaSession instance
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Response text
    """
    # Reset session context
    session.reset()

    # Add messages to context
    for message in messages:
        if message["role"] == "system":
            session.ctx.add_system_message(message["content"])
        elif message["role"] == "user":
            session.ctx.add_user_message(message["content"])
        elif message["role"] == "assistant":
            session.ctx.add_assistant_message(message["content"])

    # Query with last message
    last_message = messages[-1]
    response = await session.achat(
        content=last_message["content"],
        role=last_message["role"],
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

    return response.content


def get_session_token_usage(session: MelleaSession) -> dict:
    """Get token usage from Mellea session backend.

    Args:
        session: MelleaSession instance

    Returns:
        Dict with token usage statistics
    """
    backend = session.backend
    if hasattr(backend, "get_token_usage"):
        return backend.get_token_usage()
    else:
        logger.warning("Backend does not support token usage tracking")
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
