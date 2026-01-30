# pytest: ollama, llm

from mellea import generative


@generative
def propose_business_decision(summary: str) -> str:
    """Given a structured summary with clear recommendations, propose a business decision."""


@generative
def generate_risk_mitigation(summary: str) -> str:
    """If the summary contains risk elements, propose mitigation strategies."""


@generative
def generate_novel_recommendations(summary: str) -> str:
    """Provide a list of novel recommendations that are similar in plot or theme to the short story summary."""
