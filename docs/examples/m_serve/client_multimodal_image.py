# pytest: skip_always
import base64
from io import BytesIO

import openai
from PIL import Image

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

# Create a simple 100x100 red square image (vision models need reasonable-sized images)
img = Image.new("RGB", (100, 100), color="cyan")
img_io = BytesIO()
img.save(img_io, "PNG")
test_image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

query = "What color is this image?"

response = client.chat.completions.create(
    model="ignored",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{test_image_base64}"},
                },
            ],
        }
    ],
)

print("Query: ", query)
print("Response: ", response.choices[0].message.content)
