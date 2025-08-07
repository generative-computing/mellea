from mellea import MelleaSession
from mellea.stdlib.base import CBlock, LinearContext
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.aloras.huggingface.granite_aloras import add_granite_aloras
from mellea.stdlib.requirement import Requirement, ALoraRequirement, LLMaJRequirement
from mellea.backends.formatter import TemplateFormatter
from mellea.backends.cache import SimpleLRUCache
from mellea.backends.types import ModelOption
import pydantic

from typing_extensions import Annotated

import pytest


class TestHFALoraStuff:
    backend = LocalHFBackend(
        model_id="ibm-granite/granite-3.2-8b-instruct",
        formatter=TemplateFormatter(model_id="ibm-granite/granite-4.0-tiny-preview"),
        cache=SimpleLRUCache(5),
    )
    m = MelleaSession(backend, ctx=LinearContext())
    add_granite_aloras(backend)

    def test_system_prompt(self):
        self.m.reset()
        result = self.m.chat(
            "Where are we going?",
            model_options={ModelOption.SYSTEM_PROMPT: "Talk like a pirate."},
        )
        print(result)

    def test_constraint_alora(self):
        self.m.reset()
        answer = self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        alora_output = self.backend.get_aloras()[0].generate_using_strings(
            input="Find the difference between these two strings: aaaaaaaaaa aaaaabaaaa",
            response=str(answer),
            constraint="The answer mention that there is a b in the middle of one of the strings but not the other.",
            force_yn=False,  # make sure that the alora naturally output Y and N without constrained generation
        )
        print(alora_output)
        assert alora_output in ["Y", "N"], alora_output
        self.m.reset()

    def test_constraint_lora_with_requirement(self):
        self.m.reset()
        answer = self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        assert self.m.backend._cache is not None  # type: ignore
        assert self.m.backend._use_caches
        assert self.backend._cache.current_size() != 0
        validation_outputs = self.m.validate(
            "The answer should mention that there is a b in the middle of one of the strings but not the other.",
            return_full_validation_results=True,
        )
        assert len(validation_outputs) == 1
        alora_output, valuation_boolean = validation_outputs[0]
        assert str(alora_output) in ["Y", "N"]
        self.m.reset()

    def test_constraint_lora_override(self):
        self.m.reset()
        self.backend.default_to_constraint_checking_alora = False  # type: ignore
        answer = self.m.instruct(
            "Corporate wants you to find the difference between these two strings: aaaaaaaaaa aaaaabaaaa"
        )
        validation_outputs = self.m.validate(
            "The answer should mention that there is a b in the middle of one of the strings but not the other.",
            return_full_validation_results=True,
        )
        assert len(validation_outputs) == 1
        non_alora_output, _ = validation_outputs[0]
        assert str(non_alora_output) not in ["Y", "N"]
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
            ),
            return_full_validation_results=True,
        )
        assert len(validation_outputs) == 1
        non_alora_output, _ = validation_outputs[0]
        assert str(non_alora_output) in ["Y", "N"]
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
            ),
            return_full_validation_results=True,
        )
        assert len(validation_outputs) == 1
        non_alora_output, _ = validation_outputs[0]
        assert str(non_alora_output) not in ["Y", "N"]
        self.m.reset()

    def test_instruct(self):
        self.m.reset()
        result = self.m.instruct("Compute 1+1.")
        print(result)
        self.m.reset()

    def test_multiturn(self):
        self.m.instruct("Compute 1+1")
        beta = self.m.instruct(
            "Take the result of the previous sum and find the corresponding letter in the greek alphabet."
        )
        assert "β" in str(beta).lower()
        words = self.m.instruct(
            "Now list five English words that start with that letter."
        )
        print(words)
        self.m.reset()

    def test_format(self):
        class Person(pydantic.BaseModel):
            name: str
            email_address: Annotated[
                str,
                pydantic.StringConstraints(pattern=r"[a-zA-Z]{5,10}@example\.com"),
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
        assert (
            "@" in email.to.email_address
        ), "The @ sign should be in the meail address."
        assert email.to.email_address.endswith(
            "example.com"
        ), "The email address should be at example.com"

    def test_generate_from_raw(self):
        prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

        results = self.m.backend._generate_from_raw(
            actions=[CBlock(value=prompt) for prompt in prompts], generate_logs=None
        )

        assert len(results) == len(prompts)

    def test_generate_from_raw_with_format(self):
        prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

        class Answer(pydantic.BaseModel):
            name: str
            value: int

        results = self.m.backend._generate_from_raw(
            actions=[CBlock(value=prompt) for prompt in prompts],
            format=Answer,
            generate_logs=None,
        )

        assert len(results) == len(prompts)

        random_result = results[0]
        try:
            answer = Answer.model_validate_json(random_result.value)
        except pydantic.ValidationError as e:
            assert (
                False
            ), f"formatting directive failed for {random_result.value}: {e.json()}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
