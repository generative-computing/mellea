# pytest: ollama, llm

from mellea import generative


@generative
def summarize_meeting(transcript: str) -> str:
    """Summarize the meeting transcript into a concise paragraph of main points."""


@generative
def summarize_contract(contract_text: str) -> str:
    """Produce a natural language summary of contract obligations and risks."""


@generative
def summarize_short_story(story: str) -> str:
    """Summarize a short story, with one paragraph on plot and one paragraph on broad themes."""
