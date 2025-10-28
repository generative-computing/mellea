import os
import asyncio
from typing import Annotated, Callable

from a2a.types import Message
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
from new_types import RangeType


def bee_app(func: Callable) -> Callable:
     # Mellea function wrapper as BeeAI Agent
     server = Server()
     @server.agent(name="mellea_agent", detail=AgentDetail(interaction_mode="single-turn"))

     async def wrapper(input: Message,
                     llm: Annotated[LLMServiceExtensionServer, LLMServiceExtensionSpec.single_demand()],
                     trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
                     form: Annotated[FormExtensionServer, 
                                     FormExtensionSpec(params=FormRender(id = "email_form", title="Email Info Form", columns=2, fields=[TextField(id="name", label="First Name", col_span=1),
]))]):
        form_data = form.parse_form_response(message=input)
        name = form_data.values['name'].value if 'name' in form_data.values else get_message_text(input)
        age = RangeType(0, 5)
        print(age)
        for i in range(2):
            yield trajectory.trajectory_metadata(title=f"Attempt {i + 1}/2", content=f"Generating invitation for {name}...")
            llm_config = llm.data.llm_fulfillments.get("default")
            m = MelleaSession(OpenAIBackend(
                model_id=llm_config.api_model,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base
            ))
            
        result = await asyncio.to_thread(func, m, name, age)
        yield AgentMessage(text=m.ctx.last_output().value)

     server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))
 
     return wrapper

#if __name__ == "__main__":
#     server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

