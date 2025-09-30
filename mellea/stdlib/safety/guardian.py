"""Risk checking with Guardian models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, Context
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import Requirement


class GuardianRisk:
    """Risk definitions as defined in https://github.com/ibm-granite/granite-guardian/blob/main/cookbooks/granite-guardian-3.2/quick_start_vllm.ipynb ."""

    HARM = "harm"
    GROUNDEDNESS = "groundedness"
    PROFANITY = "profanity"
    ANSWER_RELEVANCE = "answer_relevance"


class GuardianCheck(Requirement):
    """A Check for Risks based on local huggingface backend."""

    def __init__(
        self,
        risk: str = GuardianRisk.HARM,
        *,
        model_version: str = "ibm-granite/granite-guardian-3.2-3b-a800m",
        device: str | None = None,
    ):
        """Initializes a GuardianModel instance with the specified risk definition, model version, and device.

        Args:
            risk: The risk definition to check for, defaults to GuardianRisk.HARM.
            model_version:  The version of the model, defaults to "ibm-granite/granite-guardian-3.2-3b-a800m".
            device: The computational device to use ("cuda" for GPU, "mps" for Apple Silicon, or "cpu"), defaults to None. If not specified, the best available device will be automatically selected.
        """
        super().__init__(
            check_only=True, validation_fn=lambda c: self._guardian_validate(c)
        )
        self._risk = risk
        self._model_version = model_version

        # auto-device if not more specific
        self._device = device
        if device is None:
            device_name: str = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            assert device_name is not None
            self._device = torch.device(device_name)  # type: ignore

    @staticmethod
    def _parse_output(output, input_len, tokenizer):
        """Parse the output of a guardian model and determine whether if the risk is present or not.

        Args:
            output: The model's output containing sequences from which predictions are made.
            input_len: The length of the original input sequence used for alignment with the model's output.
            tokenizer: The tokenizer associated with the model, used to decode the tokens back into text.
        """
        safe_token = "No"
        unsafe_token = "Yes"

        label = None

        full_res = tokenizer.decode(
            output.sequences[:, input_len + 1 :][0], skip_special_tokens=True
        ).strip()
        FancyLogger.get_logger().debug(f"Full: {full_res}")
        confidence_level = (
            full_res.removeprefix("<confidence>").removesuffix("</confidence>").strip()
        )
        res = tokenizer.decode(
            output.sequences[:, input_len : input_len + 1][0], skip_special_tokens=True
        ).strip()
        FancyLogger.get_logger().debug(f"Res: {res}")
        if unsafe_token.lower() == res.lower():
            label = unsafe_token
        elif safe_token.lower() == res.lower():
            label = safe_token
        else:
            label = "Failed"

        return label, confidence_level

    def _guardian_validate(self, ctx: Context):
        """Validates the last turn of a conversation context using wrt given risk.

        Code is adopted from https://huggingface.co/ibm-granite/granite-guardian-3.2-3b-a800m#quickstart-example

        Args:
            ctx (LegacyContext): The context object containing the last turn of the conversation.

        Returns:
            bool: True if there is no identified risk, False otherwise.
        """

        messages: list[dict[str, str]] = []

        last_turn = ctx.last_turn()
        assert last_turn is not None

        # This requirement can handle incomplete turns with only a user message
        # or only an assistant message. Handle both.
        if last_turn.model_input:
            user_msg = last_turn.model_input

            # Handle the variety of possible user input.
            if isinstance(user_msg, CBlock) and user_msg.value is not None:
                messages.append({"role": "user", "content": user_msg.value})
            elif isinstance(user_msg, Message) and user_msg.content != "":
                messages.append({"role": user_msg.role, "content": user_msg.content})
            else:
                messages.append({"role": "user", "content": str(user_msg)})

        if last_turn.output and last_turn.output.value:
            messages.append({"role": "assistant", "content": last_turn.output.value})

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self._model_version, device_map="auto", torch_dtype=torch.bfloat16
        )
        model.to(self._device)  # type: ignore
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self._model_version)

        # Please note that the default risk definition is of `harm`. If a config is not specified, this behavior will be applied.
        guardian_config = {"risk_name": self._risk}

        input_ids = tokenizer.apply_chat_template(
            messages,
            guardian_config=guardian_config,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )

        label, confidence = GuardianCheck._parse_output(output, input_len, tokenizer)

        # valid if there is NO risk
        return label == "No"
