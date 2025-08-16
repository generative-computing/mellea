import mellea
from mellea import MelleaSession
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
            strategy=RejectionSamplingStrategy(loop_budget=3)
        )
        assert res is not None
        assert isinstance(res.value, str)

    def test_litellm_ollama_instruct_options(self):
        res = self.m.instruct(
            "Write an email to the interns.",
            requirements=["be funny"],
            model_options={
                ModelOption.SEED: 123,
                ModelOption.TEMPERATURE: .5,
                ModelOption.THINKING:True,
                ModelOption.MAX_NEW_TOKENS:100,
                "stream":False,
                "homer_simpson":"option should be kicked out"
            }
        )
        assert res is not None
        assert isinstance(res.value, str)


