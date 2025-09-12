"""Example of using Ollama with vision models with linear context."""

from PIL import Image

from mellea import LinearContext, start_session
from mellea.stdlib.base import ImageBlock

m = start_session(model_id="granite3.2-vision", ctx=LinearContext())
# m = start_session(model_id="llava", ctx=LinearContext())

# load image
test_img = Image.open("pointing_up.jpg")

# ask a question about the image
res = m.instruct("Is the subject in the image smiling?", images=[test_img])
print(f"Result:{res!s}")

# This instruction should refer to the first image.
res2 = m.instruct("How many eyes can you identify in the image? Explain.")
print(f"Result:{res2!s}")
