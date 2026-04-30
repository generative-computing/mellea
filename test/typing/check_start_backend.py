"""Mypy overload-resolution checks for start_backend."""

from typing import assert_type

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.backends.watsonx import WatsonxAIBackend
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.start_backend import start_backend


# ---------------------------------------------------------------------------
# ollama (default backend_name)
# ---------------------------------------------------------------------------
def check_ollama_chat() -> None:
    ctx, backend = start_backend("ollama", context_type="chat")
    assert_type(ctx, ChatContext)
    assert_type(backend, OllamaModelBackend)


def check_ollama_simple() -> None:
    ctx, backend = start_backend("ollama", context_type="simple")
    assert_type(ctx, SimpleContext)
    assert_type(backend, OllamaModelBackend)


def check_ollama_default() -> None:
    ctx, backend = start_backend("ollama")
    assert_type(ctx, SimpleContext)
    assert_type(backend, OllamaModelBackend)


def check_ollama_explicit_ctx() -> None:
    my_ctx = ChatContext()
    ctx, backend = start_backend("ollama", ctx=my_ctx)
    assert_type(ctx, ChatContext)
    assert_type(backend, OllamaModelBackend)


def check_default_backend_name() -> None:
    ctx, backend = start_backend()
    assert_type(ctx, SimpleContext)
    assert_type(backend, OllamaModelBackend)


def check_default_with_chat() -> None:
    ctx, backend = start_backend(context_type="chat")
    assert_type(ctx, ChatContext)
    assert_type(backend, OllamaModelBackend)


# ---------------------------------------------------------------------------
# hf
# ---------------------------------------------------------------------------
def check_hf_chat() -> None:
    ctx, backend = start_backend("hf", context_type="chat")
    assert_type(ctx, ChatContext)
    assert_type(backend, LocalHFBackend)


def check_hf_simple() -> None:
    ctx, backend = start_backend("hf", context_type="simple")
    assert_type(ctx, SimpleContext)
    assert_type(backend, LocalHFBackend)


def check_hf_default() -> None:
    ctx, backend = start_backend("hf")
    assert_type(ctx, SimpleContext)
    assert_type(backend, LocalHFBackend)


def check_hf_explicit_ctx() -> None:
    my_ctx = ChatContext()
    ctx, backend = start_backend("hf", ctx=my_ctx)
    assert_type(ctx, ChatContext)
    assert_type(backend, LocalHFBackend)


# ---------------------------------------------------------------------------
# openai
# ---------------------------------------------------------------------------
def check_openai_chat() -> None:
    ctx, backend = start_backend("openai", context_type="chat")
    assert_type(ctx, ChatContext)
    assert_type(backend, OpenAIBackend)


def check_openai_default() -> None:
    ctx, backend = start_backend("openai")
    assert_type(ctx, SimpleContext)
    assert_type(backend, OpenAIBackend)


# ---------------------------------------------------------------------------
# watsonx
# ---------------------------------------------------------------------------
def check_watsonx_chat() -> None:
    ctx, backend = start_backend("watsonx", context_type="chat")
    assert_type(ctx, ChatContext)
    assert_type(backend, WatsonxAIBackend)


def check_watsonx_default() -> None:
    ctx, backend = start_backend("watsonx")
    assert_type(ctx, SimpleContext)
    assert_type(backend, WatsonxAIBackend)


# ---------------------------------------------------------------------------
# litellm
# ---------------------------------------------------------------------------
def check_litellm_chat() -> None:
    ctx, backend = start_backend("litellm", context_type="chat")
    assert_type(ctx, ChatContext)
    assert_type(backend, LiteLLMBackend)


def check_litellm_default() -> None:
    ctx, backend = start_backend("litellm")
    assert_type(ctx, SimpleContext)
    assert_type(backend, LiteLLMBackend)
