"""Example of using Ollama with vision models with linear context."""

import pathlib
from PIL import Image

from mellea import start_session
from mellea.stdlib.base import ChatContext, ImageBlock

m = start_session(model_id="granite3.2-vision", ctx=ChatContext())
# m = start_session(model_id="llava", ctx=ChatContext())

# load image
image_path = pathlib.Path(__file__).parent.joinpath("pointing_up.jpg")
test_pil = Image.open(image_path)

# ask a question about the image
res = m.instruct("Is the subject in the image smiling?", images=[test_pil])
print(f"Result:{res!s}")

# This instruction should refer to the first image.
res2 = m.instruct("How many eyes can you identify in the image? Explain.")
print(f"Result:{res2!s}")
