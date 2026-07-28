# pytest: skip_always
import base64

import openai
import requests

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

# "Roses are red, violets are blue." — the same sample used in mellea's audio tests.
_AUDIO_URL = "https://ai.google.dev/gemma/docs/audio/roses-are.wav"


def _download_audio() -> bytes:
    """Download the roses-are-red speech sample from Google's Gemma docs."""
    response = requests.get(_AUDIO_URL, timeout=30)
    response.raise_for_status()
    return response.content


wav_bytes = _download_audio()
audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

query = "What colors and flowers are mentioned in the audio?"

response = client.chat.completions.create(
    model="ignored",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_base64, "format": "wav"},
                },
            ],
        }
    ],
)

print("Query: ", query)
print("Response: ", response.choices[0].message.content)
