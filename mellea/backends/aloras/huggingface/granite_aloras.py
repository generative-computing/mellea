"""Huggingface implementations for IBM's "starter pack" of Activated LoRAs."""

from copy import deepcopy

import torch

from mellea.backends.huggingface import HFAlora, HFAloraCacheInfo, LocalHFBackend
from mellea.backends.types import ModelOption
from mellea.helpers.fancy_logger import FancyLogger


class HFConstraintAlora(HFAlora):
    """The Requirement Checking ALora for Granite checks if the specified requirement was satisfied by the most recent model generation. Only one requirement is checked at a time.

    Currently supports [Granite 3.2 8B](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-requirement-check) and [Granite 3.3 8B](https://huggingface.co/ibm-granite/granite-3.3-8b-alora-requirement-check) by default.
    """

    def __init__(
        self,
        name: str,
        path_or_model_id: str,
        generation_prompt: str,
        backend: LocalHFBackend,
        *,
        constraint_prompt: str | None = None,
        include_constraint_in_alora_offset: bool = False,
    ):
        """Initialize after checking that the backend is correct.

        Args:
            constraint_prompt: a template that the constraint can be interpolated into; can only have a single `{}` slot.
            include_constraint_in_alora_offset: whether to include the constraint prompt in the alora offset
        """
        super().__init__(name, path_or_model_id, generation_prompt, backend)

        # Maintain default behavior.
        if constraint_prompt is None:
            constraint_prompt = "\nRequirement: {}<|end_of_text|>\n"

        self._constraint_prompt = constraint_prompt
        self._include_constraint_in_alora_offset = include_constraint_in_alora_offset

        # We do a lot of logging for ALoras because this is an experimental feature. Maybe we should tag these log messages?
        self._logger = FancyLogger.get_logger()

    def generate_using_strings(
        self, input: str, response: str, constraint: str, force_yn: bool = True
    ) -> str:
        """Generates a constraint response from the ALora."""
        assert self._backend.alora_model is not None
        # Go ahead and do runtime type-checking because passing CBlocks into this function is a common error.
        assert type(input) is str
        assert type(response) is str
        assert type(constraint) is str
        self._backend.alora_model.set_adapter(self.name)
        cache_hit = self._backend.cache_get(response)
        if cache_hit:
            self._logger.debug(
                f"using cache for alora {self.__class__} and response '{response}'"
            )
            return self._generate_using_cache(cache_hit, constraint, force_yn)
        else:
            self._logger.debug(
                f"not using cache for alora {self.__class__} and response '{response}'"
            )
            return self._generate_not_using_cache(input, response, constraint, force_yn)

    def _generate_using_cache(
        self, cache_hit: HFAloraCacheInfo, constraint: str, force_yn: bool
    ) -> str:
        assert self._backend.alora_model is not None

        # Must tokenize the constraint here since the requirement isn't known at initialization.
        constraint_tokens = self._backend._tokenizer(
            self._constraint_prompt.format(constraint), return_tensors="pt"
        ).to(self._backend._device)

        input_combined = {
            "input_ids": torch.cat(
                [
                    cache_hit.merged_token_ids.unsqueeze(0),
                    constraint_tokens["input_ids"],
                    self._generation_prompt_tokens["input_ids"],
                ],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [
                    cache_hit.merged_attention.unsqueeze(0),
                    constraint_tokens["attention_mask"],
                    self._generation_prompt_tokens["attention_mask"],
                ],
                dim=1,
            ),
        }

        if not self._include_constraint_in_alora_offset:
            alora_offsets = [self._generation_prompt_tokens["input_ids"].shape[1] - 1]
        else:
            alora_offsets = [
                constraint_tokens["input_ids"].shape[1]
                + self._generation_prompt_tokens["input_ids"].shape[1]
                - 2
            ]
        self._logger.debug(
            f"Prompt for cached aLoRA({self.name}):\n {self._backend._tokenizer.decode(input_combined['input_ids'][0])}"
        )

        if force_yn:
            output = self._backend.alora_model.generate(
                input_combined["input_ids"].to(self._backend._device),
                attention_mask=input_combined["attention_mask"].to(
                    self._backend._device
                ),
                max_new_tokens=1,
                past_key_values=deepcopy(cache_hit.kv_cache),
                return_dict_in_generate=True,
                alora_offsets=alora_offsets,
                output_scores=True,
            )
            last_logits = output.scores[-1].squeeze(0)
            token_Y = self._backend._tokenizer("Y", add_special_tokens=False)[
                "input_ids"
            ][0]
            token_N = self._backend._tokenizer("N", add_special_tokens=False)[
                "input_ids"
            ][0]
            logit_Y = last_logits[token_Y].item()
            logit_N = last_logits[token_N].item()
            return "Y" if logit_Y > logit_N else "N"
        else:
            output = self._backend.alora_model.generate(
                input_combined["input_ids"].to(self._backend._device),
                attention_mask=input_combined["attention_mask"].to(
                    self._backend._device
                ),
                max_new_tokens=1,
                past_key_values=deepcopy(cache_hit.kv_cache),
                return_dict_in_generate=True,
                alora_offsets=alora_offsets,
            )
            output_text = self._backend._tokenizer.decode(output.sequences[0])
            assert output_text[-1] in ["Y", "N"], (
                f"The constraint model card states the the Requirement Checker model will respond with 'Y' or 'N', but found: {output_text}"
            )
            constraint_satisfied = output_text.split(self._generation_prompt)[-1]
            return constraint_satisfied[0]

    def _generate_not_using_cache(
        self, input: str, response: str, constraint: str, force_yn: bool
    ) -> str:
        assert self._backend.alora_model is not None

        # Params aren't needed when just getting the backend args.
        backend_model_opts = self._backend._simplify_and_merge(None)
        sys_prompt = backend_model_opts.get(ModelOption.SYSTEM_PROMPT, None)

        chat = [
            *([{"role": "system", "content": sys_prompt}] if sys_prompt else []),
            {"role": "user", "content": input},
            {"role": "assistant", "content": response},
        ]

        templatized = self._backend._tokenizer.apply_chat_template(chat, tokenize=False)
        assert type(templatized) is str

        # Must tokenize the constraint here since the requirement isn't known at initialization.
        templatized = templatized + self._constraint_prompt.format(constraint)

        tokenized = self._backend._tokenizer(templatized, return_tensors="pt").to(
            self._backend._device
        )

        input_combined = {
            "input_ids": torch.cat(
                [tokenized["input_ids"], self._generation_prompt_tokens["input_ids"]],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [
                    tokenized["attention_mask"],
                    self._generation_prompt_tokens["attention_mask"],
                ],
                dim=1,
            ),
        }

        if not self._include_constraint_in_alora_offset:
            alora_offsets = [self._generation_prompt_tokens["input_ids"].shape[1] - 1]
        else:
            # Get the constraint tokens separately so that we can calculate the alora offsets.
            constraint_tokens = self._backend._tokenizer(
                self._constraint_prompt.format(constraint), return_tensors="pt"
            ).to(self._backend._device)

            alora_offsets = [
                constraint_tokens["input_ids"].shape[1]
                + self._generation_prompt_tokens["input_ids"].shape[1]
                - 2
            ]

        self._logger.debug(
            f"Prompt for non-cached aLoRA({self.name}):\n{self._backend._tokenizer.decode(input_combined['input_ids'][0])}"
        )

        if force_yn:
            output = self._backend.alora_model.generate(
                input_combined["input_ids"].to(self._backend._device),
                attention_mask=input_combined["attention_mask"].to(
                    self._backend._device
                ),
                max_new_tokens=1,
                return_dict_in_generate=True,
                alora_offsets=alora_offsets,
                output_scores=True,
            )
            last_logits = output.scores[-1].squeeze(0)
            token_Y = self._backend._tokenizer("Y", add_special_tokens=False)[
                "input_ids"
            ][0]
            token_N = self._backend._tokenizer("N", add_special_tokens=False)[
                "input_ids"
            ][0]
            logit_Y = last_logits[token_Y].item()
            logit_N = last_logits[token_N].item()
            return "Y" if logit_Y > logit_N else "N"
        else:
            output = self._backend.alora_model.generate(
                input_combined["input_ids"].to(self._backend._device),
                attention_mask=input_combined["attention_mask"].to(
                    self._backend._device
                ),
                max_new_tokens=1,
                return_dict_in_generate=True,
                alora_offsets=alora_offsets,
            )
            output_text = self._backend._tokenizer.decode(output.sequences[0])
            constraint_satisfied = output_text.split(self._generation_prompt)[-1]
            return constraint_satisfied[0]


def add_granite_aloras(backend: LocalHFBackend):
    """Adds the IBM Granite "starter pack" ALoras to a backend."""
    if backend._hf_model_id == "ibm-granite/granite-3.2-8b-instruct":
        backend.add_alora(
            HFConstraintAlora(
                name="constraint",
                path_or_model_id="ibm-granite/granite-3.2-8b-alora-requirement-check",
                generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
                backend=backend,
                constraint_prompt="\nRequirement: {}<|end_of_text|>\n",
                include_constraint_in_alora_offset=False,
            )
        )
    elif backend._hf_model_id == "ibm-granite/granite-3.3-8b-instruct":
        backend.add_alora(
            HFConstraintAlora(
                name="constraint",
                path_or_model_id="ibm-granite/granite-3.3-8b-alora-requirement-check",
                generation_prompt="<|start_of_role|>check_requirement<|end_of_role|>",
                backend=backend,
                constraint_prompt="\n<|start_of_role|>requirement<|end_of_role|>{}<|end_of_text|>\n",
                include_constraint_in_alora_offset=True,
            )
        )
    else:
        raise ValueError(
            f"cannot add_granite_aloras to unknown huggingface model_id / backend: {backend._hf_model_id}"
        )
