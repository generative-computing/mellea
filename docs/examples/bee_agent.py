import os
import asyncio
from typing import Annotated

from mellea.stdlib.chat import Message
from a2a.utils.message import get_message_text
from beeai_sdk.server import Server
from beeai_sdk.a2a.types import AgentMessage
from beeai_sdk.a2a.extensions import (
    LLMServiceExtensionServer, LLMServiceExtensionSpec,
    TrajectoryExtensionServer, TrajectoryExtensionSpec, 
    AgentDetail
)
from beeai_sdk.a2a.extensions.ui.form import (
    FormExtensionServer, FormExtensionSpec, FormRender, TextField
)
from mellea import MelleaSession, start_session
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.requirement import req, Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.base import ChatContext
from beeai_framework.backend.chat import ChatModel
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformMemoryManager, BeeAIPlatformServer
from bee_platform import bee_app

server = Server()

from typing import Callable, Tuple


@bee_app
def mellea_func(m: MelleaSession, name: str, age: int) -> str:
    """
    Example email writing module that starts a Mellea session and adds any calls to an internal ChatContext
    Inputs:
        name: str
        age: str
    """
    response = m.chat(f"Write an email for {name}, {name} is {age}")
    return response.content





