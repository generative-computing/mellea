import asyncio
from dateutil import parser as dateparser
import functools
import html
import json
import openai
import os
import pytz
import re
from transformers import AutoTokenizer, GPT2TokenizerFast, LlamaTokenizerFast
from typing import Any, List

from mellea.stdlib.base import ChatContext
from mellea.stdlib.chat import Message

from constants import (
    MODEL_NAME, CONTEXT_LENGTH, 
    EVAL_MODEL_NAME, 
    EMB_API_BASE, EMB_MODEL_NAME, EMB_CONTEXT_LENGTH,
    MAX_RETRIES
    )
from utils.logger import logger, DefaultProgressLogger, BaseProgressLogger
from mellea import MelleaSession


def get_tokenizer(model_name: str):
    if "qwen" in model_name.lower():
        return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B")  # or Qwen2.5 if hosted
    elif "llama" in model_name.lower():
        # Need to require access
        # return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer")
        return LlamaTokenizerFast.from_pretrained(tokenizer_path)
    elif "roberta" in model_name.lower() or "watbert" in model_name.lower() or "slate" in model_name.lower():
        return AutoTokenizer.from_pretrained("roberta-base")
    elif "gpt" in model_name.lower():
        return GPT2TokenizerFast.from_pretrained('Xenova/gpt-4o')
    elif "deepseek" in model_name.lower():
        return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    elif "granite" in model_name.lower():
        return AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-h-tiny")
    else:
        return AutoTokenizer.from_pretrained(model_name)
_tokenizer = get_tokenizer(MODEL_NAME)
_emb_tokenizer = get_tokenizer(EMB_MODEL_NAME)

def llm_retry(max_retries=10, default_output=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            self = args[0] if args else None
            logger = getattr(self, 'logger', getattr(kwargs, 'logger', DefaultProgressLogger()))
            expected_time = getattr(kwargs, 'expected_time', None)
            for attempt in range(max_retries):
                if expected_time and time.time() >= expected_time:
                    break
                try:
                    return await func(*args, **kwargs)
                except openai.APIConnectionError as e:
                    logger.error(f"[Retry {attempt+1}/{max_retries}] API connection failed", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))  # Exponential backoff (2s, 4s, 8s, etc.)
                except json.decoder.JSONDecodeError:
                    logger.error(f"[Retry {attempt+1}/{max_retries}] JSON Decode error", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))
                except TypeError:
                    logger.error(f"[Retry {attempt+1}/{max_retries}] JSON format error", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))
                except Exception:
                    logger.error(f"[Retry {attempt+1}/{max_retries}] Unexpected error", exc_info=True)
                    await asyncio.sleep(min(2 ** attempt, 30))
            return default_output
        return wrapper
    return decorator

class Token_Counter:
    _instance = None

    # Maintain a singleton driver across files
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, database=None):
        if not hasattr(self, "_initialized"):
            self._initialized = True

            self.counter = {}

    def get_token_usage(self):
        return self.counter
    
    def update_token_usage(self, key, token):
        self.counter[key] = self.counter.get(key, 0) + token

    def reset_token_usage(self):
       self.counter = {}

token_counter = Token_Counter()

@llm_retry(max_retries=MAX_RETRIES, default_output=[])
async def generate_embedding(
    session: Any,
    texts: List[str], 
    timeout=3600,
    logger: BaseProgressLogger = DefaultProgressLogger(),
    **kwargs
    ) -> List:
    texts = [truncate_to_tokens(text, EMB_CONTEXT_LENGTH, tokenizer=_emb_tokenizer) for text in texts]
    if len(texts) == 0:
        return []
    
    if EMB_API_BASE:
        responses = await session.embeddings.create(
            input=texts,
            model=EMB_MODEL_NAME,
            timeout=timeout, 
            **kwargs
        )
        return [data.embedding for data in responses.data]
    else:
        embeddings = session.encode(
            sentences=texts,
            normalize_embeddings=True
        )
        return embeddings.tolist()

@llm_retry(max_retries=MAX_RETRIES, default_output="")
async def generate_response(
    session: MelleaSession,
    prompt, 
    max_tokens=8192, 
    temperature=0.1, 
    top_p=0.9, 
    logger: BaseProgressLogger = DefaultProgressLogger(),
    return_raw: bool = False,
    custom_client = None,
    custom_model = None,
    timeout=3600,
    expected_time = None,
    **kwargs) -> str:
    
    model = custom_model if custom_model else MODEL_NAME
    session.reset()

    max_context_length = CONTEXT_LENGTH - max_tokens - 1024
    for message in prompt[:-1]:
        if message["role"] == "system":
            tokens = _tokenizer.encode(message["content"], truncation=True, max_length=max_context_length)
            max_context_length -= len(tokens)
        if message["role"] == "user":
            message["content"] = truncate_to_tokens(message["content"], max_context_length, tokenizer=_tokenizer)
        
        session.ctx.add(Message(role=message["role"], content=message["content"]))

    # Send a new message
    last_message = prompt[-1]
    response = await session.achat(content=last_message["content"], role=last_message["role"])

    if return_raw:
        return reponse
    return response.content  # Extract response text
    
async def generate_eval_response(**kwargs):
    return await generate_response(
        custom_client=_eval_client, 
        custom_model=EVAL_MODEL_NAME,
        **kwargs
    )

# tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer")
# tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)

# def trim_predictions_to_max_token_length(prediction):
#     """Trims prediction output to 75 tokens"""
#     max_token_length = 75
#     tokenized_prediction = tokenizer.encode(prediction)
#     trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
#     trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
#     return trimmed_prediction

def truncate_to_tokens(text: str, max_tokens: int = EMB_CONTEXT_LENGTH, tokenizer=_tokenizer) -> str:
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens - 1)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
#     """Split a string by multiple markers"""
#     if not markers:
#         return [content]
#     results = re.split("|".join(re.escape(marker) for marker in markers), content)
#     return [r.strip() for r in results if r.strip()]

# def is_float_regex(value):
#     return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def extract_json_objects(text, decoder=json.JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data
    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results

def maybe_load_json(text: str, force_load = True, default_output=None) -> object:
    try:
        res = json.loads(text)
    except Exception as e:
        # logger.error(f"JSON parsing error: {text}", exc_info=True)
        if force_load:
            res = extract_json_objects(text)
            res = res[0] if len(res) else res
        else:
            return default_output
    return res

# def maybe_load_jsons(texts: List[str], **kwargs) -> List[object]:
#     return [maybe_load_json(text, **kwargs) for text in texts]

# # Refer the utils functions of the official GraphRAG implementation:
# # https://github.com/microsoft/graphrag
# def clean_str(input: Any) -> str:
#     """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
#     # If we get non-string input, just give it back
#     if not isinstance(input, str):
#         return input

#     result = html.unescape(input.strip())
#     # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
#     return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

# Explicit mapping for common US time zone abbreviations
TZINFOS = {
    "PT": pytz.timezone("America/Los_Angeles"),  # Pacific Time
    "PST": pytz.timezone("America/Los_Angeles"),
    "PDT": pytz.timezone("America/Los_Angeles"),
    "ET": pytz.timezone("America/New_York"),     # Eastern Time
    "EST": pytz.timezone("America/New_York"),
    "EDT": pytz.timezone("America/New_York"),
    "CT": pytz.timezone("America/Chicago"),
    "CST": pytz.timezone("America/Chicago"),
    "CDT": pytz.timezone("America/Chicago"),
    "MT": pytz.timezone("America/Denver"),
    "MST": pytz.timezone("America/Denver"),
    "MDT": pytz.timezone("America/Denver"),
}
def parse_timestamp(timestamp: str, verbose: bool = False):
    try:
        timestamp_dt = dateparser.parse(timestamp, fuzzy=True, tzinfos=TZINFOS)
        timestamp_dt = timestamp_dt.astimezone(pytz.UTC)
        timestamp_iso = timestamp_dt.isoformat()
    except Exception as e:
        timestamp_iso = None
        if verbose:
            print(f"[Warning] Failed to parse query_time: {timestamp} -> {e}")

    return timestamp_iso

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop