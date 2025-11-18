import pathlib

from PIL import Image as PILImage

from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from mellea.stdlib.chat import Message


def generate_with_size_optimization(
    s: MelleaSession, images: list[PILImage.Image], query: str
) -> Message:
    modified_images = []
    for image in images:
        # do something
        modified_images.append(image)

    return s.chat(query, images=modified_images)


if __name__ == "__main__":
    m = MelleaSession(LiteLLMBackend("ollama/granite3.2-vision"))

    image_path = pathlib.Path(__file__).parent.joinpath("pointing_up.jpg")
    test_pil = PILImage.open(image_path)
    response = generate_with_size_optimization(
        m, [test_pil], "How many eyes can you identify in the image? Explain."
    )
    print(str(response.content))
