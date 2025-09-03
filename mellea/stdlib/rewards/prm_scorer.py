import torch

from mellea.stdlib.base import CBlock, Context
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import Requirement
from mellea.stdlib.rewards.prm import (
    GenerativePRMForInference,
    RegressionPRMForInference,
)

# class GenerativePRMScorer(Requirement):
#     """A generative process reward model scorer based on local huggingface backend."""

#     def __init__(
#         self,
#         *,
#         model_version: str = "ibm-granite/granite-3.3-8b-lora-math-prm",
#         device: str | None = None,
#         correct_token="Y",
#         generation_prompt="Is this response correct so far (Y/N)?",
#         step_splitter="\n\n",
#     ):
#         """

#         Args:
#             model_version:  The version of the model, defaults to "ibm-granite/granite-3.3-8b-lora-math-prm".
#             device: The computational device to use ("cuda" for GPU, "mps" for Apple Silicon, or "cpu"), defaults to None. If not specified, the best available device will be automatically selected.
#             correct_token: PRM generated token that indicates step is correct
#             generation_prompt: Generation prompt required for the PRM scorer
#             step_splitter: string on which assistant response is split into steps
#         """
#         super().__init__(check_only=True, validation_fn=lambda c: self._prm_validate(c))

#         self._model_version = model_version

#         # auto-device if not more specific
#         self._device = device
#         if device is None:
#             device_name: str = (
#                 "cuda"
#                 if torch.cuda.is_available()
#                 else "mps"
#                 if torch.backends.mps.is_available()
#                 else "cpu"
#             )
#             assert device_name is not None
#             self._device = torch.device(device_name)  # type: ignore

#         self.correct_token = correct_token
#         self.generation_prompt = generation_prompt
#         self.step_splitter = step_splitter

#     def _prm_validate(self, ctx: Context):
#         """
#         Returns PRM score of last turn of context
#         """
#         last_turn = ctx.last_turn()
#         assert last_turn is not None

#         # This requirement can handle only complete turns with both
#         # a user message and an assistant message

#         assert last_turn.model_input is not None and last_turn.output is not None
#         assert last_turn.output.value is not None

#         user_msg = last_turn.model_input

#         # Handle the variety of possible user input.
#         if isinstance(user_msg, CBlock) and user_msg.value is not None:
#             user_query = user_msg.value
#         elif isinstance(user_msg, Message) and user_msg.content != "":
#             user_query = user_msg.content
#         else:
#             user_query = str(user_msg)

#         assistant_content = last_turn.output.value

#         # convert assistant message into a list of steps
#         list_of_steps = [
#             step.strip()
#             for step in assistant_content.split(self.step_splitter)
#             if step.strip != ""
#         ]

#         # Load model
#         model = GenerativePRMForInference(
#             self._model_version,
#             self.correct_token,
#             self.generation_prompt,
#             load_in_bf16=True,
#             device=self._device,
#         )
#         model.to(self._device)

#         rewards, rewards_per_step = model(([user_query], [list_of_steps]))

#         # return single reward item for the response
#         assert len(rewards) == 1

#         # offload and delete model before returning rewards
#         del model

#         return rewards[0]


# class RegressionPRMScorer(Requirement):
#     """A regression process reward model scorer based on local huggingface backend."""

#     def __init__(
#         self,
#         *,
#         model_version: str,
#         device: str | None = None,
#         step_token="<end_of_step>",
#         step_splitter="\n\n",
#     ):
#         """
#         Args:
#             model_version:  The version of the model
#             device: The computational device to use ("cuda" for GPU, "mps" for Apple Silicon, or "cpu"), defaults to None. If not specified, the best available device will be automatically selected.
#             step_token: special token indicating the end of the step
#             step_splitter: string on which assistant response is split into steps
#         """
#         super().__init__(check_only=True, validation_fn=lambda c: self._prm_validate(c))

#         self._model_version = model_version

#         # auto-device if not more specific
#         self._device = device
#         if device is None:
#             device_name: str = (
#                 "cuda"
#                 if torch.cuda.is_available()
#                 else "mps"
#                 if torch.backends.mps.is_available()
#                 else "cpu"
#             )
#             assert device_name is not None
#             self._device = torch.device(device_name)  # type: ignore

#         self.step_token = step_token
#         self.step_splitter = step_splitter

#     def _prm_validate(self, ctx: Context):
#         """
#         Returns PRM score of last turn of context
#         """

#         last_turn = ctx.last_turn()
#         assert last_turn is not None

#         # This requirement can handle only complete turns with both
#         # a user message and an assistant message

#         assert last_turn.model_input is not None and last_turn.output is not None
#         assert last_turn.output.value is not None

#         user_msg = last_turn.model_input

#         # Handle the variety of possible user input.
#         if isinstance(user_msg, CBlock) and user_msg.value is not None:
#             user_query = user_msg.value
#         elif isinstance(user_msg, Message) and user_msg.content != "":
#             user_query = user_msg.content
#         else:
#             user_query = str(user_msg)

#         assistant_content = last_turn.output.value

#         # convert assistant message into a list of steps
#         list_of_steps = [
#             step.strip()
#             for step in assistant_content.split(self.step_splitter)
#             if step.strip != ""
#         ]

#         # Load model
#         model = RegressionPRMForInference(
#             self._model_version, self.step_token, load_in_bf16=True, device=self._device
#         )

#         rewards, rewards_per_step = model(([user_query], [list_of_steps]))

#         # return single reward item for the response
#         assert len(rewards) == 1

#         # offload and delete model before returning rewards
#         del model
#         return rewards[0]


class PRMScorer(Requirement):
    """A process reward model scorer based on local huggingface backend."""

    def __init__(
        self,
        *,
        model_version: str = "ibm-granite/granite-3.3-8b-lora-math-prm",
        device: str | None = None,
        step_splitter="\n\n",
        prm_type: str = "generative",
        **prm_kwargs,
    ):
        """

        Args:
            model_version:  The version of the model, defaults to "ibm-granite/granite-3.3-8b-lora-math-prm".
            device: The computational device to use ("cuda" for GPU, "mps" for Apple Silicon, or "cpu"), defaults to None. If not specified, the best available device will be automatically selected.
            correct_token: PRM generated token that indicates step is correct
            generation_prompt: Generation prompt required for the PRM scorer
            step_splitter: string on which assistant response is split into steps
            prm_type: type of prm tobe used. must be either `generative` or `regression`
            prm_kwargs: args for PRM. For Generative, pass `correct_token`, `generation_prompt`. For Regression, pass `step_token`
        """
        super().__init__(check_only=True, validation_fn=lambda c: self._prm_validate(c))

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

        self.step_splitter = step_splitter
        assert prm_type.lower() in ["generative", "regression"], (
            "prm_type must be either generative or regression"
        )
        self.prm_type = prm_type.lower()
        self.prm_kwargs = prm_kwargs

    def _prm_validate(self, ctx: Context):
        """
        Returns PRM score of last turn of context
        """
        last_turn = ctx.last_turn()
        assert last_turn is not None

        # This requirement can handle only complete turns with both
        # a user message and an assistant message

        assert last_turn.model_input is not None and last_turn.output is not None
        assert last_turn.output.value is not None

        user_msg = last_turn.model_input

        # Handle the variety of possible user input.
        if isinstance(user_msg, CBlock) and user_msg.value is not None:
            user_query = user_msg.value
        elif isinstance(user_msg, Message) and user_msg.content != "":
            user_query = user_msg.content
        else:
            user_query = str(user_msg)

        assistant_content = last_turn.output.value

        # convert assistant message into a list of steps
        list_of_steps = [
            step.strip()
            for step in assistant_content.split(self.step_splitter)
            if step.strip != ""
        ]

        # Load model
        model: GenerativePRMForInference | RegressionPRMForInference
        if self.prm_type == "generative":
            model = GenerativePRMForInference(
                model_path=self._model_version,
                load_in_bf16=True,
                device=self._device,
                **self.prm_kwargs,
            )
            model.to(self._device)
        elif self.prm_type == "regression":
            model = RegressionPRMForInference(
                model_path=self._model_version,
                load_in_bf16=True,
                device=self._device,
                **self.prm_kwargs,
            )  # type: ignore[no-redef]
        else:
            raise NotImplementedError

        rewards, rewards_per_step = model(([user_query], [list_of_steps]))

        # return single reward item for the response
        assert len(rewards) == 1

        # offload and delete model before returning rewards
        del model

        return rewards[0]
