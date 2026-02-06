# test/rits_backend_tests/test_openai_integration.py
import os
from typing import Annotated

import pydantic
import pytest

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.adapters import GraniteCommonAdapter
from mellea.backends.openai import OpenAIBackend
from mellea.core import CBlock, Context, ModelOutputThunk
from mellea.formatters import TemplateFormatter
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ALoraRequirement, LLMaJRequirement

# The vllm tests are disabled by default, because we need a test environment with the vLLM server running.
# We use an env var VLLM_TESTS_ENABLED to enable these tests.
# to run the tests, do this: VLLM_TESTS_ENABLED="1' pytest test_openai_vllm.py
_vllm_tests_enabled = (
    os.environ.get("VLLM_TESTS_ENABLED") is not None
    and str(os.environ.get("VLLM_TESTS_ENABLED")) == "1"
)
if not _vllm_tests_enabled:
    pytest.skip(
        reason="OpenAI vLLM tests should only be run with install scripts.",
        allow_module_level=True,
    )


class TestOpenAIBackend:
    backend = OpenAIBackend(
        model_id="ibm-granite/granite-3.3-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-3.3-8b-instruct"),
        base_url="http://0.0.0.0:8000/v1",
        api_key="EMPTY",
    )
    m = MelleaSession(backend, ctx=ChatContext())

    def test_instruct(self) -> None:
        self.m.reset()
        result = self.m.instruct("Compute 1+1.")
        assert isinstance(result, ModelOutputThunk)
        assert "2" in result.value  # type: ignore
        self.m.reset()

    def test_multiturn(self) -> None:
        self.m.instruct("What is the capital of France?")
        answer = self.m.instruct("Tell me the answer to the previous question.")
        assert "Paris" in answer.value  # type: ignore
        self.m.reset()

    # def test_api_timeout_error(self):
    #     self.m.reset()
    #     # Mocking the client to raise timeout error is needed for full coverage
    #     # This test assumes the exception is properly propagated
    #     with self.assertRaises(Exception) as context:
    #         self.m.instruct("This should trigger a timeout.")
    #     assert "APITimeoutError" in str(context.exception)
    #     self.m.reset()

    # def test_model_id_usage(self):
    #     self.m.reset()
    #     result = self.m.instruct("What model are you using?")
    #     assert "granite3.3:8b" in result.value
    #     self.m.reset()

    def test_format(self) -> None:
        class Person(pydantic.BaseModel):
            name: str
            # it does not support regex patterns in json schema
            email_address: str
            # email_address: Annotated[
            #     str,
            #     pydantic.StringConstraints(pattern=r"[a-zA-Z]{5,10}@example\.com"),
            # ]

        class Email(pydantic.BaseModel):
            to: Person
            subject: str
            body: str

        output = self.m.instruct(
            "Write a short email to Olivia, thanking her for organizing a sailing activity. Her email server is example.com. No more than two sentences. ",
            format=Email,
            model_options={ModelOption.MAX_NEW_TOKENS: 2**8},
        )
        print("Formatted output:")
        email = Email.model_validate_json(
            output.value  # type: ignore
        )  # this should succeed because the output should be JSON because we passed in a format= argument...
        print(email)

        print("address:", email.to.email_address)
        # this is not guaranteed, due to the lack of regexp pattern
        # assert "@" in email.to.email_address
        # assert email.to.email_address.endswith("example.com")

    async def test_generate_from_raw(self) -> None:
        prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

        results = await self.m.backend.generate_from_raw(
            actions=[CBlock(value=prompt) for prompt in prompts], ctx=self.m.ctx
        )

        assert len(results) == len(prompts)
        assert results[0].value is not None

    async def test_generate_from_raw_with_format(self) -> None:
        prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

        class Answer(pydantic.BaseModel):
            name: str
            value: int

        results = await self.m.backend.generate_from_raw(
            actions=[CBlock(value=prompt) for prompt in prompts],
            format=Answer,
            ctx=self.m.ctx,
        )

        assert len(results) == len(prompts)

        random_result = results[0]
        try:
            Answer.model_validate_json(random_result.value)  # type: ignore
        except pydantic.ValidationError as e:
            assert False, (
                f"formatting directive failed for {random_result.value}: {e.json()}"
            )


class TestOpenAIALoraStuff:
    backend = OpenAIBackend(
        model_id="ibm-granite/granite-3.3-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-3.3-8b-instruct"),
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    backend.add_adapter(
        GraniteCommonAdapter(
            "requirement_check", base_model_name=backend.base_model_name
        )
    )

    m = MelleaSession(backend, ctx=ChatContext())

    def test_adapters(self) -> None:
        assert len(self.backend._added_adapters.items()) > 0

        adapter = self.backend._added_adapters["requirement_check_alora"]
        self.backend.load_adapter(adapter.qualified_name)
        assert adapter.qualified_name in self.backend._loaded_adapters

        # Ensure you can load the same adapter twice.
        self.backend.load_adapter(adapter.qualified_name)

        # Ensure you can unload an adapter.
        self.backend.unload_adapter(adapter.qualified_name)
        self.backend.unload_adapter(adapter.qualified_name)
        assert adapter.qualified_name not in self.backend._loaded_adapters

    def test_system_prompt(self) -> None:
        self.m.reset()
        result = self.m.chat(
            "Where are we going?",
            model_options={ModelOption.SYSTEM_PROMPT: "Talk like a pirate."},
        )
        print(result)

    def test_constraint_lora_with_requirement(self) -> None:
        self.m.reset()
        self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        validation_outputs = self.m.validate(
            ALoraRequirement(
                "The answer should mention that there is a b in the middle of one of the strings but not the other."
            )
        )
        assert len(validation_outputs) == 1
        val_result = validation_outputs[0]
        assert "requirement_likelihood" in str(val_result.reason)
        self.m.reset()

    def test_constraint_lora_override(self) -> None:
        self.m.reset()
        self.backend.default_to_constraint_checking_alora = False  # type: ignore
        self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        validation_outputs = self.m.validate(
            LLMaJRequirement(
                "The answer should mention that there is a b in the middle of one of the strings but not the other."
            )
        )
        assert len(validation_outputs) == 1
        val_result = validation_outputs[0]
        assert str(val_result.reason) not in ["Y", "N"]
        self.backend.default_to_constraint_checking_alora = True
        self.m.reset()

    def test_constraint_lora_override_does_not_override_alora(self) -> None:
        self.m.reset()
        self.backend.default_to_constraint_checking_alora = False  # type: ignore
        self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        validation_outputs = self.m.validate(
            ALoraRequirement(
                "The answer should mention that there is a b in the middle of one of the strings but not the other."
            )
        )
        assert len(validation_outputs) == 1
        non_alora_output = validation_outputs[0]
        assert "requirement_likelihood" in str(non_alora_output.reason)

        # Ensure the ValidationResult has its thunk and context set. Ensure the context has
        # the correct actions / results in it.
        assert isinstance(non_alora_output.context, Context)
        assert isinstance(non_alora_output.thunk, ModelOutputThunk)
        assert non_alora_output.context.previous_node is not None
        assert isinstance(
            non_alora_output.context.previous_node.node_data,
            ALoraRequirement,  # type: ignore
        )
        assert non_alora_output.context.node_data is non_alora_output.thunk

        self.backend.default_to_constraint_checking_alora = True
        self.m.reset()

    def test_llmaj_req_does_not_use_alora(self) -> None:
        self.m.reset()
        self.backend.default_to_constraint_checking_alora = True  # type: ignore
        self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        validation_outputs = self.m.validate(
            LLMaJRequirement(
                "The answer should mention that there is a b in the middle of one of the strings but not the other."
            )
        )
        assert len(validation_outputs) == 1
        non_alora_output = validation_outputs[0]
        assert str(non_alora_output.reason) not in ["Y", "N"]
        self.m.reset()

    def test_instruct(self) -> None:
        self.m.reset()
        result = self.m.instruct("Compute 1+1.")
        print(result)
        self.m.reset()

    def test_multiturn(self) -> None:
        self.m.instruct("Compute 1+1")
        self.m.instruct(
            "Let n be the result of the previous sum. Find the n-th letter in the greek alphabet."
        )
        words = self.m.instruct(
            "Now list five English words that start with that letter."
        )
        print(words)
        self.m.reset()

    def test_format(self) -> None:
        class Person(pydantic.BaseModel):
            name: str
            email_address: Annotated[
                str, pydantic.StringConstraints(pattern=r"[a-zA-Z]{5,10}@example\.com")
            ]

        class Email(pydantic.BaseModel):
            to: Person
            subject: str
            body: str

        output = self.m.instruct(
            "Write a short email to Olivia, thanking her for organizing a sailing activity. Her email server is example.com. No more than two sentences. ",
            format=Email,
            model_options={ModelOption.MAX_NEW_TOKENS: 2**8},
        )
        print("Formatted output:")
        email = Email.model_validate_json(
            output.value  # type: ignore
        )  # this should succeed because the output should be JSON because we passed in a format= argument...
        print(email)

        print("address:", email.to.email_address)
        assert "@" in email.to.email_address
        assert email.to.email_address.endswith("example.com")


if __name__ == "__main__":
    pytest.main([__file__])
