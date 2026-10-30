# Audio-Text-to-Text Examples

Examples for audio-text-to-text workflows using Mellea with an
OpenAI-compatible backend. For background and supported backends, see
[Use Speech and Audio Input](../../docs/how-to/use-speech-and-audio.md).

## Examples

| File | Description |
| :--- | :--- |
| [`audio_examples.py`](./audio_examples.py) | Downloads sample audio from OpenAI CDN, demonstrates both `session.instruct()` and `session.chat()`. Configurable via `LLAMA_SERVER_URL`, `LLAMA_SERVER_API_KEY`, and `LLAMA_SERVER_MODEL` environment variables. |
