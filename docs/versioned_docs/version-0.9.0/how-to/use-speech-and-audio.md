---
title: "Use Speech and Audio Input"
description: "Pass audio to instruct() and chat() calls, and check which backends can send it."
sidebar_label: "Use Speech and Audio"
# diataxis: how-to
---

Mellea can send audio alongside your text prompt: pass it to any `instruct()` or `chat()`
call using the `audio` parameter.

**Prerequisites:** an audio-capable model reachable through an OpenAI-compatible endpoint,
and an audio file.

> **Backend note:** Only the OpenAI-compatible backends can send audio. `OllamaModelBackend`,
> `WatsonxAIBackend`, and `LocalHFBackend` raise a `ValueError` rather than silently dropping
> the clip — see [Backend support](#backend-support) below.

---

## Basic usage

`audio` takes a list of audio blocks. Build one with the constructor that matches your
source — `from_file` for a path on disk:

```python
# Requires: mellea
# Returns: str
from mellea import start_session
from mellea.core import AudioBlock

m = start_session("openai", model_id="gpt-audio-1.5")

result = m.instruct(
    "Transcribe the speech in this clip.",
    audio=[AudioBlock.from_file("speech.wav")],
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

`audio` deliberately does **not** accept bare paths or URLs. Converting them is an
explicit step, so the type you pass says exactly what will be sent and any read or
download failure surfaces where you wrote it rather than mid-request.

---

## Choosing a constructor

| Source | Use |
| ------ | --- |
| A file on disk | `AudioBlock.from_file(path)` |
| Bytes in memory | `AudioBlock.from_bytes(data)` |
| A remote URL, fetched now | `AudioBlock.from_url(url)` |
| A remote URL, fetched at send time and cached | `AudioUrlBlock(url, format=...)` |
| Base64 you already have | `AudioBlock(value, format=...)` |

`from_file`, `from_bytes`, and `from_url` all detect the format from the data's magic
bytes, so a mislabelled file is reported accurately — a WAV named `.mp3` yields
`format == "wav"`:

```python
# Requires: mellea
# Returns: str
from mellea.core import AudioBlock

# `speech.mp3` here is actually a WAV file that was given the wrong extension.
clip = AudioBlock.from_file("speech.mp3")
print(clip.format)  # "wav" — read from the file's contents, not its name
```

```python
# Requires: mellea, requests
# Returns: AudioBlock
import requests
from mellea.core import AudioBlock

wav = requests.get("https://cdn.openai.com/API/docs/audio/alloy.wav").content
clip = AudioBlock.from_bytes(wav)
```

Pass `format=` explicitly to skip detection when you already know it, or when the payload
is not one mellea recognises:

```python
# Requires: mellea
# Returns: AudioBlock
from mellea.core import AudioBlock

clip = AudioBlock.from_file("recording.opus", format="opus")
```

You can also construct a block from base64 directly. With a data URI the format is read from
the MIME type; with raw base64 you must supply `format`:

```python
# Requires: mellea
# Returns: None
import base64
from mellea.core import AudioBlock

with open("speech.wav", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

from_data_uri = AudioBlock(f"data:audio/wav;base64,{b64}")  # format inferred
from_raw = AudioBlock(b64, format="wav")  # format required
```

### Remote audio

OpenAI Chat Completions has no audio-by-URL content part, so Mellea downloads the clip and
inlines it for you. There are two ways, differing only in *when* the fetch happens.

`AudioBlock.from_url()` downloads immediately, so a bad URL fails at the call site:

```python
# Requires: mellea
# Returns: AudioBlock
from mellea.core import AudioBlock

clip = AudioBlock.from_url("https://example.com/speech.wav")
print(clip.format)  # detected from the downloaded bytes
```

`AudioUrlBlock` defers the download to send time and memoizes it per URL, so a clip reused
across several turns is fetched once:

```python
# Requires: mellea
# Returns: str
from mellea import start_session
from mellea.core import AudioUrlBlock

m = start_session("openai", model_id="gpt-audio-1.5")

clip = AudioUrlBlock("https://example.com/speech.wav", format="wav")
result = m.instruct("Transcribe this clip.", audio=[clip])
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

Prefer `AudioUrlBlock` when the same URL is used repeatedly; prefer `from_url` when you
want the failure surfaced eagerly. Downloads are capped at 50 MB with a 30-second
timeout, and `AudioUrlBlock` requires an explicit `format` because nothing has been
fetched yet at construction time.

> **Note:** some servers do accept audio by URL through non-standard extensions to the
> OpenAI schema — [vLLM's `audio_url`](https://docs.vllm.ai/en/v0.6.2/getting_started/examples/openai_audio_api_client.html)
> is one. Mellea always downloads and inlines instead, which works everywhere. Passing a
> URL straight through to a server that supports it would be a future addition, gated on
> detecting such a server; it would let the download and cache be skipped.

---

## Supported formats

OpenAI Chat Completions accepts only `wav` and `mp3` for audio input. Other
OpenAI-compatible servers may accept more, so mellea does not restrict the format it sends —
`flac` and `ogg` are detected and passed through, and an explicit `format=` is always
honoured. If a server rejects a format, that surfaces as a server-side error.

**Mellea does not transcode audio.** Convert the file yourself (for example with `ffmpeg`),
or transcribe it to text and pass the transcript as part of your prompt.

Format detection covers `wav`, `mp3`, `flac`, and `ogg`. When the bytes match none of these
and you did not pass `format=`, construction raises:

```python
# Requires: mellea
# Returns: None
from mellea.core import AudioBlock

try:
    AudioBlock.from_file("notes.txt")
except ValueError as e:
    print(e)
    # Could not identify the audio format of 'notes.txt'. Pass format explicitly ...
```

---

## Multi-turn audio with ChatContext

Audio passed to `instruct()` or `chat()` is stored in the
[`ChatContext`](../reference/glossary.md) turn history, so later calls in the same session
can refer back to the clip without passing it again:

```python
# Requires: mellea
# Returns: None
from mellea import start_session
from mellea.core import AudioBlock
from mellea.stdlib.context import ChatContext

m = start_session("openai", model_id="gpt-audio-1.5", ctx=ChatContext())

# First turn — attach the clip
r1 = m.instruct("Transcribe this clip.", audio=[AudioBlock.from_file("meeting.wav")])
print(str(r1))

# Second turn — the clip is still in context
r2 = m.instruct("Summarise the main point in one sentence.")
print(str(r2))
```

> **Cost warning:** the clip is re-sent on the wire on *every* subsequent turn, not just the
> first. Audio is far larger than text — a few minutes of WAV is megabytes of base64 and
> thousands of audio tokens — so a long conversation over one clip gets expensive quickly.
> For extended multi-turn work over the same audio, consider transcribing once and
> continuing over the transcript.

---

## Backend support

| Backend | Audio support | Notes |
| ------- | ------------- | ----- |
| `OpenAIBackend` | ✓ | Requires an audio-capable model |
| `LiteLLMBackend` | ✓ | Depends on the underlying provider and model |
| `OllamaModelBackend` | ✗ | Ollama's chat API has no audio input |
| `WatsonxAIBackend` | ✗ | The chat path carries no audio |
| `LocalHFBackend` | ✗ | Would require a processor-based audio model |

The ✗ backends raise a `ValueError` when handed audio. This is deliberate: silently dropping
the clip would send a text-only prompt and produce a confident answer about audio the model
never received.

> **Full example:** [`docs/examples/audio_text_models/audio_examples.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/audio_text_models/audio_examples.py)
> **Serving audio via `m serve`:** [`docs/examples/m_serve/multimodal-audio/`](https://github.com/generative-computing/mellea/tree/main/docs/examples/m_serve/multimodal-audio)

---

**See also:** [Use Images and Vision Models](../how-to/use-images-and-vision.md) |
[Working with Data](../how-to/working-with-data.md)
