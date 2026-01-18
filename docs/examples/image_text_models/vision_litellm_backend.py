"""Examples of using vision models with LiteLLM backend."""

import os

import litellm
from PIL import Image

from mellea import MelleaSession, start_session
from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.openai import OpenAIBackend
from mellea.core import ImageBlock
import pathlib

# use LiteLLM to talk to Ollama or anthropic or.....
m = MelleaSession(LiteLLMBackend("ollama/granite3.2-vision"))
# m = MelleaSession(LiteLLMBackend("ollama/llava"))
# m = MelleaSession(LiteLLMBackend("anthropic/claude-3-haiku-20240307"))

image_path = pathlib.Path(__file__).parent.joinpath("pointing_up.jpg")
test_pil = Image.open(image_path)

# check if model is able to do text chat
ch = m.chat("What's 1+1?")
print(str(ch.content))

# test with PIL image
res_instruct = m.instruct(
    "Is there a person on the image? Is the subject in the image smiling?",
    images=[test_pil],
)
print(f"Test with PIL and instruct: \n{str(res_instruct)}\n-----")
# print(m.last_prompt())

# with PIL image and using m.chat
res_chat = m.chat(
    "How many eyes can you identify in the image? Explain.", images=[test_pil]
)
print(f"Test with PIL and chat: \n{str(res_chat.content)}\n-----")

# and now without images again...
res_empty = m.instruct("How many eyes can you identify in the image?", images=[])
print(f"Test without image: \n{str(res_empty)}\n-----")
