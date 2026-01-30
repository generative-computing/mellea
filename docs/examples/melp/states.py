# pytest: ollama, llm

import asyncio

from mellea.core import Context, CBlock, Backend
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.components import SimpleComponent


async def main(backend: Backend, ctx: Context):
    a_states = "Alaska,Arizona,Arkansas".split(",")
    m_states = "Missouri", "Minnesota", "Montana", "Massachusetts"

    a_state_pops = dict()
    for state in a_states:
        a_state_pops[state], _ = await backend.generate_from_context(
            CBlock(f"What is the population of {state}? Respond with an integer only."),
            SimpleContext(),
        )
    a_total_pop = SimpleComponent(
        instruction=CBlock(
            "What is the total population of these states? Respond with an integer only."
        ),
        **a_state_pops,
    )
    a_state_total, _ = await backend.generate_from_context(a_total_pop, SimpleContext())

    m_state_pops = dict()
    for state in m_states:
        m_state_pops[state], _ = await backend.generate_from_context(
            CBlock(f"What is the population of {state}? Respond with an integer only."),
            SimpleContext(),
        )
    m_total_pop = SimpleComponent(
        instruction=CBlock(
            "What is the total population of these states? Respond with an integer only."
        ),
        **m_state_pops,
    )
    m_state_total, _ = await backend.generate_from_context(m_total_pop, SimpleContext())

    print(await a_state_total.avalue())
    print(await m_state_total.avalue())


backend = OllamaModelBackend(model_id="granite4:latest")
asyncio.run(main(backend, SimpleContext()))
