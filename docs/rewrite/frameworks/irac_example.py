# import mellea.stdlib.frameworks.irac as irac
import asyncio
import mellea.stdlib.frameworks.irac as irac


async def main():
    from mellea.backends.ollama import OllamaModelBackend
    from mellea.stdlib.context import SimpleContext

    backend = OllamaModelBackend(model_id="granite4:latest")
    ctx = SimpleContext()

    scenario = "Suzanne is renting an apartment in Sprinfield, Massachusetts. Her lease stipulates that she will be charged charged a non-refundable fee for the installation of a new lock and key upon signing the lease. Is this charge permissible?"

    i, r, a, c, s = await irac.irac(ctx, backend, scenario)
    print(s)
    print([r.format_for_llm() for r in r])


if __name__ == "__main__":
    asyncio.run(main())
