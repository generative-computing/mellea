"""Simplified utils using Mellea's built-in features."""
from typing import Any, List, Optional
from mellea import MelleaSession
from utils.logger import logger


def create_embedding_session(
    api_base: Optional[str] = None,
    api_key: str = "dummy",
    model_name: Optional[str] = None,
    timeout: int = 1800,
    rits_api_key: Optional[str] = None
) -> Any:
    """Create embedding session (OpenAI API or local model).

    This function creates an embedding session that can be either:
    - An OpenAI-compatible API client (if api_base is provided)
    - A local SentenceTransformer model (if api_base is None)

    Args:
        api_base: API base URL for OpenAI-compatible embedding service (None for local)
        api_key: API key for authentication (default: "dummy")
        model_name: Model name/path for embeddings
        timeout: Request timeout in seconds (default: 1800)
        rits_api_key: Optional RITS API key for custom headers

    Returns:
        Embedding session object (openai.AsyncOpenAI or SentenceTransformer)
    """
    if api_base:
        logger.info("Using OpenAI-compatible embedding API")
        logger.info(f"  API base: {api_base}")
        logger.info(f"  Model: {model_name}")

        import openai

        headers = {}
        if rits_api_key:
            headers['RITS_API_KEY'] = rits_api_key

        return openai.AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            default_headers=headers if headers else None
        )
    else:
        logger.info("Using local SentenceTransformer model")
        logger.info(f"  Model: {model_name}")

        import torch
        from sentence_transformers import SentenceTransformer

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        logger.info(f"  Device: {device}")

        return SentenceTransformer(
            model_name,
            device=device
        )


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
