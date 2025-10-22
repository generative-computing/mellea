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


server = Server()

from typing import Callable, Tuple




def generate_response(m: MelleaSession, prompt: str, requirements: list[Requirement]):
    # Logs user messages to the context and passes prompt through m object
    user_msg = Message(role="user", content=prompt)
    m.ctx.add(user_msg) #Adds the user message to the context (Message objects are not JSON serializable)
    return m.instruct(prompt, requirements=requirements)

def bee_app(func: Callable, **inputs) -> Callable:
     @server.agent(name="mellea_agent", detail=AgentDetail(interaction_mode="multi-turn"))

     async def wrapper(llm: Annotated[LLMServiceExtensionServer, LLMServiceExtensionSpec.single_demand()], 
                     trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
		     form: Annotated[FormExtensionServer, FormExtensionSpec(params=None)]):
        
        name = "angela"
        age = "28"
        llm_config = llm.data.llm_fulfillments.get("default")
        model = ChatModel.from_name("ollama:llama3.1")
       	m = MelleaSession(OpenAIBackend(
            model_id=llm_config.api_model,
            api_key=llm_config.api_key,
            base_url=llm_config.api_base
        ))
        requirements = [req("Be formal."), req("Be funny."), Requirement("Use less than 100 words.", validation_fn=simple_validate(lambda o: len(o.split()) < 100))]
        prompt, callable = func(name, age) 
        result = await asyncio.to_thread(callable, m, prompt, requirements)
        yield AgentMessage(text=m.ctx.last_output().value)
     return wrapper

@bee_app
def new_func(name: str, age: int) -> Tuple[ str, Callable ]:
    # Takes in name and age, and outputs both prompt and callable (m.chat)
    prompt = f"Write a birthday card for {name}. Include their age: {age}"
    return (prompt, generate_response)



def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
