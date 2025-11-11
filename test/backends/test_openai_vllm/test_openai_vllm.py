# test/rits_backend_tests/test_openai_integration.py
from contextvars import Context
from mellea import MelleaSession
from mellea.stdlib.base import CBlock, ModelOutputThunk, ChatContext
from mellea.backends.openai import OpenAIBackend
from mellea.backends.aloras.openai.granite_aloras import add_granite_aloras
from mellea.stdlib.requirement import (
    Requirement,
    ALoraRequirement,
    LLMaJRequirement,
    req,
)
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.types import ModelOption

import pydantic
from typing_extensions import Annotated
import pytest
import os


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
        model_id="ibm-granite/granite-3.2-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-3.2-8b-instruct"),
        base_url="http://0.0.0.0:8000/v1",
        api_key="EMPTY",
    )
    m = MelleaSession(backend, ctx=ChatContext())

    def test_instruct(self):
        self.m.reset()
        result = self.m.instruct("Compute 1+1.")
        assert isinstance(result, ModelOutputThunk)
        assert "2" in result.value  # type: ignore
        self.m.reset()

    def test_multiturn(self):
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

    def test_format(self):
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
            output.value
        )  # this should succeed because the output should be JSON because we passed in a format= argument...
        print(email)

        print("address:", email.to.email_address)
        # this is not guaranteed, due to the lack of regexp pattern
        # assert "@" in email.to.email_address
        # assert email.to.email_address.endswith("example.com")
        pass

    def test_generate_from_raw(self):
        prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

        results = self.m.backend.generate_from_raw(
            actions=[CBlock(value=prompt) for prompt in prompts], ctx=self.m.ctx
        )

        assert len(results) == len(prompts)

    def test_generate_from_raw_with_format(self):
        prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

        class Answer(pydantic.BaseModel):
            name: str
            value: int

        results = self.m.backend.generate_from_raw(
            actions=[CBlock(value=prompt) for prompt in prompts],
            format=Answer,
            ctx=self.m.ctx,
        )

        assert len(results) == len(prompts)

        random_result = results[0]
        try:
            answer = Answer.model_validate_json(random_result.value)
        except pydantic.ValidationError as e:
            assert False, (
                f"formatting directive failed for {random_result.value}: {e.json()}"
            )


class TestOpenAIALoraStuff:
    backend = OpenAIBackend(
        model_id="ibm-granite/granite-3.2-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-4.0-tiny-preview"),
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    m = MelleaSession(backend, ctx=ChatContext())
    add_granite_aloras(backend)

    def test_system_prompt(self):
        self.m.reset()
        result = self.m.chat(
            "Where are we going?",
            model_options={ModelOption.SYSTEM_PROMPT: "Talk like a pirate."},
        )
        print(result)

    @pytest.mark.xfail
    def test_constraint_alora(self):
        self.m.reset()
        answer = self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        alora_output = self.backend.get_aloras()[
            0
        ].generate_using_strings(
            input="Find the difference between these two strings: aaaaaaaaaa aaaaabaaaa",
            response=str(answer),
            constraint="The answer mention that there is a b in the middle of one of the strings but not the other.",
            force_yn=False,  # make sure that the alora naturally output Y and N without constrained generation
        )
        assert alora_output in ["Y", "N"], alora_output
        self.m.reset()

    def test_constraint_lora_with_requirement(self):
        self.m.reset()
        answer = self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        validation_outputs = self.m.validate(
            ALoraRequirement(
                "The answer should mention that there is a b in the middle of one of the strings but not the other."
            )
        )
        assert len(validation_outputs) == 1
        val_result = validation_outputs[0]
        assert str(val_result.reason) in ["Y", "N"]
        self.m.reset()

    def test_constraint_lora_override(self):
        self.m.reset()
        self.backend.default_to_constraint_checking_alora = False  # type: ignore
        answer = self.m.instruct(
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

    def test_constraint_lora_override_does_not_override_alora(self):
        self.m.reset()
        self.backend.default_to_constraint_checking_alora = False  # type: ignore
        answer = self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        validation_outputs = self.m.validate(
            ALoraRequirement(
                "The answer should mention that there is a b in the middle of one of the strings but not the other."
            )
        )
        assert len(validation_outputs) == 1
        non_alora_output = validation_outputs[0]
        assert str(non_alora_output.reason) in ["Y", "N"]

        # Ensure the ValidationResult has its thunk and context set. Ensure the context has
        # the correct actions / results in it.
        assert isinstance(non_alora_output.context, Context)
        assert isinstance(non_alora_output.thunk, ModelOutputThunk)
        assert isinstance(
            non_alora_output.context.previous_node.node_data, ALoraRequirement
        )
        assert non_alora_output.context.node_data is non_alora_output.thunk

        self.backend.default_to_constraint_checking_alora = True
        self.m.reset()

    def test_llmaj_req_does_not_use_alora(self):
        self.m.reset()
        self.backend.default_to_constraint_checking_alora = True  # type: ignore
        answer = self.m.instruct(
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

    def test_instruct(self):
        self.m.reset()
        result = self.m.instruct("Compute 1+1.")
        print(result)
        self.m.reset()

    def test_multiturn(self):
        self.m.instruct("Compute 1+1")
        beta = self.m.instruct(
            "Let n be the result of the previous sum. Find the n-th letter in the greek alphabet."
        )
        words = self.m.instruct(
            "Now list five English words that start with that letter."
        )
        print(words)
        self.m.reset()

    def test_format(self):
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
            output.value
        )  # this should succeed because the output should be JSON because we passed in a format= argument...
        print(email)

        print("address:", email.to.email_address)
        assert "@" in email.to.email_address
        assert email.to.email_address.endswith("example.com")


if __name__ == "__main__":
    pytest.main([__file__])
