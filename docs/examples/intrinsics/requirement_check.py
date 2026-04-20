# pytest: huggingface, e2e

"""Example usage of the requirement check intrinsic.

Intrinsic function that determines if the text satisfies the given requirements.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/requirement_check.py
```
"""

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext

user_text = "Invite for an IBM office party."
response_text = """
Dear Team,

To celebrate our recent successes and take a well-deserved moment to recharge,
you are cordially invited to a team social. Please join us for an evening of
live music, appetizers, and drinks as we recognize our collective wins.

Event Details
* **Date:** Saturday, April 25, 2026
* **Time:** 6:00 PM
* **Location:** Ryan’s Bar, Chelsea, NY
* **Highlights:** Live entertainment and refreshments

RSVP
To ensure we have an accurate headcount for catering, please confirm your
attendance by **Friday, April 10, 2026**.

We look forward to seeing everyone there and celebrating our hard work together.

**Best regards,**
[Your Name/Management Team]
"""
requirement = "Use a professional tone."

backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
# --- Alternative: OpenAI backend with Granite Switch (requires vLLM server) ---
# Requires the adapter for this intrinsic to be embedded in the Granite Switch
# model. See docs/examples/granite-switch/ for a full runnable example.
# from mellea.backends.openai import OpenAIBackend
# from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_8B
# from mellea.formatters import TemplateFormatter
#
# backend = OpenAIBackend(
#     model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name,
#     formatter=TemplateFormatter(model_id=IBM_GRANITE_SWITCH_4_1_8B.hf_model_name),
#     base_url="http://localhost:8000/v1",  # vLLM server URL
#     api_key="EMPTY",
#     load_embedded_adapters=True,
# )
# --- End alternative ---
context = (
    ChatContext()
    .add(Message("user", user_text))
    .add(Message("assistant", response_text))
)

result = core.requirement_check(context, backend, requirement)
print(f"Requirements Satisfied: {result}")  # float between 0.0 and 1.0
