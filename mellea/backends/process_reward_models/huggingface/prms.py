"""PRM Implementations for Local HuggingFace Backends."""

import torch
from transformers.tokenization_utils_base import BatchEncoding

from mellea.backends.huggingface import HFProcessRewardModel


class HFGenerativePRM(HFProcessRewardModel):
    """A Generative PRM that works with a huggingface backend."""

    def __init__(
        self,
        model_name_or_path: str = "ibm-granite/granite-3.3-8b-lora-math-prm",
        score_token: str = "Y",
        device: str | None = None,
        generation_prompt: str = "Is this response correct so far (Y/N)?",
        step_separator: str = "\n\n",
    ):
        """Initialize a Generative PRM that works with a huggingface backend. Currently supports and tested with IBM Process Reward Models.

        Args:
            model_name_or_path (str): A local path to PRM or a huggingface PRM
            score_token (str): token who's logits correspond to the PRM score. Usually is a correctness indicator (for generative PRMs)
            device (str): pointer to device
            generation_prompt (str): Optional prompt to be added before generation
            step_separator (str): string on which to separate the content into steps
        """
        super().__init__(model_name_or_path, score_token, device)
        self.generation_prompt = (
            generation_prompt if generation_prompt is not None else ""
        )
        self.step_separator = step_separator
        self.softmax = torch.nn.Softmax(dim=-1)

    def score(self, query: str, response: str) -> tuple[list[float], list[list[float]]]:
        """Returns a final and per-step score for a given input query and response.

        Args:
            query (str): User query
            response (str): Assistant Response to score
        """
        list_of_steps = self.stepify(response, self.step_separator)
        # get tokenized batch
        batches = self.prepare_inputs(query, list_of_steps)
        all_rewards = []
        all_rewards_per_step = []

        # find the chat turn where assistant message starts to find the correct placement of the score token
        # add empty system prompt to prevent model from adding its own system prompt
        chat_template_to_turn = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "assistant", "content": self._score_token},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        # removing the system prompt by finding the assistant turn, which usually starts like <|..|>assistant<|..>
        asst_text = chat_template_to_turn[chat_template_to_turn.find(">assistant<") :][
            1:
        ]
        asst_toks = self.tokenizer(
            asst_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        asst_toks_before_correct_token = asst_toks[
            : torch.where(asst_toks == self._score_token_id)[
                0
            ].item()  # type: ignore
        ].tolist()  # type : ignore

        # move each item of the batch to the device
        for i in batches:
            batches[i] = batches[i].to(self.model.device)

        with torch.no_grad():
            model_outputs = self.model(**batches)
            logits = model_outputs.logits  # (bsz, seq_len, vocab_size)

        for batch_idx in range(logits.shape[0]):
            per_input_rewards = []
            # for each element in the batch (i.e., each input)
            # we need to get logits for all tokens where the token is self._score_token (in assistant turn)
            # find batch index for **assistant** turn is self._score_token, not just the self._score_token_id
            correct_token_indices = torch.where(
                batches["input_ids"][batch_idx] == self._score_token_id
            )[0].tolist()
            prm_indices = []
            for t_idx in correct_token_indices:
                if (
                    batches["input_ids"][batch_idx][
                        t_idx - len(asst_toks_before_correct_token) : t_idx
                    ].tolist()
                    == asst_toks_before_correct_token
                ):
                    prm_indices.append(
                        t_idx - 1
                    )  # the logits for token i predict the token i+1: so, we need to look at the **previous** token logits

            assert len(prm_indices) > 0
            #  convert logits to probabilities and get the probability of the correct token id as reward
            for prm_idx in prm_indices:
                per_input_rewards.append(
                    self.softmax(logits[batch_idx, prm_idx, :])[
                        self._score_token_id
                    ].item()
                )

            # aggregate. return final rewards
            all_rewards_per_step.append(per_input_rewards)
            sum = 0
            for reward in per_input_rewards:
                sum += reward
            per_input_reward = sum / len(per_input_rewards)
            all_rewards.append(per_input_reward)

        return all_rewards, all_rewards_per_step

    def prepare_inputs(self, user_content: str, steps: list[str]) -> BatchEncoding:
        """Prepare the inputs for inference with the model.

        Args:
            user_content (str): the user query
            steps (List(str)): assistant response, broken down into steps
        """
        msgs = []
        for s_idx, step in enumerate(steps):
            # apply chat template as expected by the reward model
            # rewards are calculated from the logit of self._score_token as produced by the assistant
            if s_idx == 0:
                msgs.append(
                    {
                        "role": "user",
                        "content": user_content
                        + " "
                        + step
                        + " "
                        + self.generation_prompt,
                    }
                )
            else:
                # first add last assistant turn
                msgs.append({"role": "assistant", "content": self._score_token})
                msgs.append(
                    {"role": "user", "content": step + " " + self.generation_prompt}
                )

        # append last assistant turn
        msgs.append({"role": "assistant", "content": self._score_token})
        input_message = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, tokenize=False
        )
        return self.tokenizer(
            [input_message], return_tensors="pt", padding=True, truncation=True
        )


class HFRegressionPRM(HFProcessRewardModel):
    """A Regression PRM that works with a huggingface backend."""

    def __init__(
        self,
        model_name_or_path: str,
        score_token: str = "<end_of_step>",
        device: str | None = None,
        step_separator: str = "\n\n",
    ):
        """Initialize a Regression PRM that works with a huggingface backend. Currently supports and tested with IBM Process Reward Models.

        Args:
            model_name_or_path (str): A local path to PRM or a huggingface PRM
            score_token (str): token who's logits correspond to the PRM score. Usually is a step demarker (for non-generative PRMs)
            device (str): pointer to the device on which to run the model
            step_separator (str): string on which to separate the input content into steps
        """
        super().__init__(model_name_or_path, score_token, device)

        # initialize PRM head
        self.prm_head = torch.nn.Linear(
            self.model.config.hidden_size, 2, bias=False, dtype=self.model.dtype
        ).to(self.model.device)

        state = torch.load(model_name_or_path + "/added_params.bin")
        # need to do this-- we save model dict as `prm_head.weight` during training
        new_state_dict = {}
        for k, v in state.items():
            new_k = k.replace("prm_head.", "")
            new_state_dict[new_k] = v

        self.prm_head.load_state_dict(new_state_dict)
        self.prm_head.eval()

        self.step_separator = step_separator
        self.softmax = torch.nn.Softmax(dim=-1)

    def score(self, query: str, response: str) -> tuple[list[float], list[list[float]]]:
        """Returns a final and per-step score for a given input query and response.

        Args:
            query (str): User query
            response (str): Assistant Response to score
        """
        list_of_steps = self.stepify(response, self.step_separator)
        # tokenizes the batch and concatenates the list of steps into a single step-separated response
        batch = self.prepare_inputs(query, list_of_steps)
        # move each item of the batch to the device
        for i in batch:
            batch[i] = batch[i].to(self.model.device)

        with torch.no_grad():
            model_outputs = self.model(**batch, output_hidden_states=True)
            # all logits
            all_prm_logits = self.prm_head(model_outputs["hidden_states"][-1]).squeeze(
                -1
            )

        # get logits for each end of step i.e. logits for step_eos positions in the input
        prm_probs = []
        rewards = []
        for idx in range(all_prm_logits.shape[0]):
            prm_indices = torch.where(batch["input_ids"][idx] == self._score_token_id)[
                0
            ]
            assert prm_indices.shape[0] > 0
            # head produces two logits, the second one is the logit for the correct answer
            # convert logits to probabilities using softmax
            # return list of floats instead of list of tensors
            prm_probs_per_sample = [
                t.item() for t in self.softmax(all_prm_logits[idx][prm_indices])[:, 1]
            ]
            prm_probs.append(prm_probs_per_sample)

            reward = sum(prm_probs_per_sample) / len(prm_probs_per_sample)
            rewards.append(reward)

        return rewards, prm_probs

    def prepare_inputs(self, user_content: str, steps: list[str]) -> BatchEncoding:
        """Prepare the inputs for inference with the model.

        Args:
            user_content (str): the user query
            steps (List(str)): assistant response, broken down into steps
        """
        text_with_steps_marked = ""

        for step in steps:
            text_with_steps_marked += f"{step} {self._score_token}"

        message = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": text_with_steps_marked},
        ]
        input_message = self.tokenizer.apply_chat_template(message, tokenize=False)

        return self.tokenizer(
            [input_message], return_tensors="pt", padding=True, truncation=True
        )
