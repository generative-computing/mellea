import os
import asyncio
from typing import Annotated

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

server = Server()

from typing import Callable, Tuple

#form_render = FormRender(
#    id="party_invite_form",
#    title="Party Invitation",
#    description="Enter the person's name to generate an invitation",
#    columns=1,
#    submit_label="Generate",
#    fields=[
#        TextField(
#            type="text",
#            id="name",
#            label="Name",
#            placeholder="Enter the person's name",
#            required=True,
#            col_span=1
#        )
#    ]
#)

def generate_response(m: MelleaSession, prompt: str):
    return m.chat(prompt)

def bee_app(func: Callable, **inputs):
    inputs = inputs.get("inputs", {})
    @server.agent(name="mellea_agent", detail=AgentDetail(interaction_mode="single-turn"))
    async def wrapper(llm: Annotated[LLMServiceExtensionServer, LLMServiceExtensionSpec.single_demand()], 
                     trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
		     form: Annotated[FormExtensionServer, FormExtensionSpec(params=None)]):
      	
        name = "Alison"
        age = "28" 

        llm_config = llm.data.llm_fulfillments.get("default")
    
        m = MelleaSession(OpenAIBackend(
            model_id=llm_config.api_model,
            api_key=llm_config.api_key,
            base_url=llm_config.api_base
        ))

	#requirements = [req("Be formal."), req("Be funny."), Requirement("Use less than 100 words.", validation_fn=simple_validate(lambda o: len(o.split()) < 100))]
 
        prompt, callable = func(name, age) 
        result = await asyncio.to_thread(
            callable,
            m,
            prompt,
        )
        yield AgentMessage(text=m.ctx.last_output().value)
    return wrapper

@bee_app
def new_func(name: str, age: int) -> Tuple[ str, Callable[ [str], str] ]:
    prompt = f"Write a birthday card for {name}. Include their age: {age}"
    return (prompt, generate_response)



def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
