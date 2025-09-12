"""Examples using vision models with OpenAI backend."""

import os

from PIL import Image

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.base import ImageBlock

# # using anthropic AI model ...
# anth_key = os.environ.get("ANTHROPIC_API_KEY")
# m = MelleaSession(OpenAIBackend(model_id="claude-3-haiku-20240307",
#                                 api_key=anth_key,  # Your Anthropic API key
#                                 base_url="https://api.anthropic.com/v1/"  # Anthropic's API endpoint
#                                 ))

# using LM Studio model locally
m = MelleaSession(
    OpenAIBackend(model_id="qwen/qwen2.5-vl-7b", base_url="http://127.0.0.1:1234/v1")
)

# load PIL image and convert to mellea ImageBlock
test_pil = Image.open("pointing_up.jpg")
test_img = ImageBlock.from_pil_image(test_pil)

# check if model is able to do text chat
ch = m.chat("What's 1+1?")
print(str(ch.content))

# now test with MELLEA image
res = m.instruct(
    "Is there a person on the image? Is the subject in the image smiling?",
    images=[test_img],
)
print(str(res))
# print(m.last_prompt())

# and now with PIL image and using m.chat
res = m.chat("How many eyes can you identify in the image? Explain.", images=[test_pil])
print(str(res.content))

# and now without images again...
res = m.instruct("How many eyes can you identify in the image?", images=[])
print(str(res))
