import mellea
from mellea import MelleaSession, generative
from mellea.backends import ModelOption
from mellea.backends.litellm import LiteLLMBackend
from mellea.stdlib.chat import Message
from mellea.stdlib.sampling import RejectionSamplingStrategy


class TestLitellmOllama:
    m = MelleaSession(LiteLLMBackend())

    def test_litellm_ollama_chat(self):
        res = self.m.chat("hello world")
        assert res is not None
        assert isinstance(res, Message)

    def test_litellm_ollama_instruct(self):
        res = self.m.instruct(
            "Write an email to the interns.",
            requirements=["be funny"],
            strategy=RejectionSamplingStrategy(loop_budget=3),
        )
        assert res is not None
        assert isinstance(res.value, str)

    def test_litellm_ollama_instruct_options(self):
        res = self.m.instruct(
            "Write an email to the interns.",
            requirements=["be funny"],
            model_options={
                ModelOption.SEED: 123,
                ModelOption.TEMPERATURE: 0.5,
                ModelOption.THINKING: True,
                ModelOption.MAX_NEW_TOKENS: 100,
                "reasoning_effort":True,
                "stream": False,
                "homer_simpson": "option should be kicked out",
            },
        )
        assert res is not None
        assert isinstance(res.value, str)
        assert "homer_simpson" not in self.m.ctx.last_output_and_logs()[1].model_options

    def test_gen_slot(self):
        @generative
        def is_happy(text: str) -> bool:
            """Determine if text is of happy mood."""

        h = is_happy(self.m, text="I'm enjoying life.")

        assert isinstance(h, bool)
        assert h is True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
