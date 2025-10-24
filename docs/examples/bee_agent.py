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
from mellea.bee_platform import bee_app

server = Server()

from typing import Callable, Tuple




@bee_app
def new_func(name: str, age: int) -> Tuple[ str, Callable ]:
    # Takes in name and age, and outputs both prompt and callable (m.chat)
    response = m.chat(f"Write a birthday card for {name}, {name} is {age}")
    return response.value



def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
