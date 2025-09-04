import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerativePRMForInference(torch.nn.Module):
    """
    Class for Generative Process Reward Models for Inference
    Uses Huggingface backend to load the model (which is trained using LoRA adapters)
    """

    def __init__(
        self,
        model_path="ibm-granite/granite-3.3-8b-lora-math-prm",
        correct_token="Y",
        generation_prompt="Is this response correct so far (Y/N)?",
        load_in_bf16=True,
        device=None,
    ) -> None:
        super().__init__()

        if not load_in_bf16:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )

        if device is not None:
            self.model.to(device)
        self.device = self.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.truncation_side = "left"  # prevents truncation from right (default): needed since we always want to have the last step and last generation prompt from the context.
        self.correct_token = correct_token
        self.correct_token_id = self.tokenizer.encode(
            self.correct_token, add_special_tokens=False
        )[0]
        self.generation_prompt = generation_prompt
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, raw_inputs):
        """
        Expects a raw_batch of (questions: List[str], steps: List[List[str]])
        Return the aggregated score for each problem (i.e., the average of the per-step scores), along with the per-step scores
        """

        # get un-tokenized batch
        batches = self.prepare_batch(raw_inputs)
        # each element of the batch consists of a list of num_steps messages corresponding to a single input, which we need to handle
        all_rewards = []
        all_rewards_per_step = []

        chat_template_to_turn = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": self.correct_token}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if "system" in chat_template_to_turn:
            if "granite" in self.model.config.model_type.lower():
                # for granite, apply_chat_template also adds system prompt
                asst_text = (
                    "<|start_of_role|>assistant<|end_of_role|>"
                    + self.correct_token
                    + "<|end_of_text|>"
                )
            elif "phi" in self.model.config.model_type.lower():
                # phi reasoning also applies the system prompt
                asst_text = (
                    "<|im_start|>assistant<|im_sep|>"
                    + self.correct_token
                    + "<|im_end|>'"
                )
        else:
            asst_text = chat_template_to_turn
        asst_toks = self.tokenizer(
            asst_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        asst_toks_before_correct_token = asst_toks[
            : torch.where(asst_toks == self.correct_token_id)[0].item()
        ].tolist()

        # each element in batch contains a question and the response
        for i in batches:
            batches[i] = batches[i].to(self.model.device)

        with torch.no_grad():
            model_outputs = self.model(**batches)
            logits = model_outputs.logits  # (bsz, seq_len, vocab_size)

        for batch_idx in range(logits.shape[0]):
            per_input_rewards = []
            # for each element in the batch (i.e., each input)
            # we need to get logits for all tokens where the token in "Y" (in assistant turn)
            # find batch index for assistant turn "Y", not just the correct_token_id
            correct_token_indices = torch.where(
                batches["input_ids"][batch_idx] == self.correct_token_id
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
                    )  # the logits for token i predict the token i+1: so, we need to look at the PREVIOUS token logits

            assert len(prm_indices) > 0
            #  convert logits to probabilities and get the probability of the correct token id as reward
            for prm_idx in prm_indices:
                per_input_rewards.append(
                    self.softmax(logits[batch_idx, prm_idx, :])[
                        self.correct_token_id
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

    def prepare_batch(self, raw_batch):
        """
        Expects a raw_batch of (question, list_of_steps). The list of steps is joined with the step_eos token
        prepare_batch() function splits each step into an individual response, and prepares an input batch
        prepare batch for forward pass
        """

        questions, list_of_steps = raw_batch
        assert len(questions) == len(list_of_steps)

        inputs = []
        for i in range(len(questions)):
            user_content = questions[i]
            steps = list_of_steps[i]
            msgs = []
            for s_idx, step in enumerate(steps):
                #  apply chat template as expected by RM input
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
                    msgs.append({"role": "assistant", "content": self.correct_token})
                    msgs.append(
                        {"role": "user", "content": step + " " + self.generation_prompt}
                    )

            # append the last asst turn
            msgs.append({"role": "assistant", "content": self.correct_token})

            input_message = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=False, tokenize=False
            )

            inputs.append(input_message)

        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )


class RegressionPRMForInference(torch.nn.Module):
    """
    Class for Regression (non-generative) Process Reward Models for Inference
    Uses Huggingface backend to load the model
    All regression process reward models trained by the GMA team at IBM research use a special step token, <end_of_step>
    """

    def __init__(
        self,
        model_path: str,
        step_eos: str = "<end_of_step>",
        load_in_bf16: bool = True,
        device=None,
    ) -> None:
        super().__init__()

        # Load the model
        self.model: AutoModelForCausalLM
        if not load_in_bf16:
            self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                model_path, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
        self.device = self.model.device
        self.config = self.model.config

        # get the token IDs for the step separator token
        self.step_eos = step_eos
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.tokenizer.add_tokens(self.step_eos)
        self.step_eos_id = self.tokenizer.encode(
            self.step_eos, add_special_tokens=False
        )[0]

        # load the PRM head
        self.prm_head = torch.nn.Linear(
            self.model.config.hidden_size, 2, bias=False, dtype=self.model.dtype
        ).to(self.model.device)
        state = torch.load(model_path + "/added_params.bin")
        self.load_state_dict(state, strict=False)
        self.model.eval()

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, raw_batch):
        """
        Expects a raw_batch of (questions: List[str], steps: List[List[str]])
        Return the aggregated score for each problem (i.e., the average of the per-step scores), along with the per-step scores
        """

        # tokenizes the batch and concatenates the list of steps into a single step-separated response
        batch = self.prepare_batch(raw_batch).to(self.device)

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
            prm_indices = torch.where(batch["input_ids"][idx] == self.step_eos_id)[0]
            if prm_indices.shape[0] == 0:
                # no match found-- model did not produce outputs in correct step-wise format
                prm_probs.append([None])
                reward = None
            else:
                # head produces two logits, the second one is the logit for the correct answer
                # convert logits to probabilities using softmax
                # return list of floats instead of list of tensors
                prm_probs_per_sample = [
                    t.item()
                    for t in self.softmax(all_prm_logits[idx][prm_indices])[:, 1]
                ]
                prm_probs.append(prm_probs_per_sample)

                reward = sum(prm_probs_per_sample) / len(prm_probs_per_sample)
                rewards.append(reward)

        return rewards, prm_probs

    def prepare_batch(self, raw_batch):
        """
        Tokenize and prepare batch for forward pass
        Expects a raw_batch of (question, list_of_steps). The list of steps is joined with the step_eos token
        """

        questions, list_of_steps = raw_batch
        assert len(questions) == len(list_of_steps)

        inputs = []
        for i in range(len(questions)):
            text_with_steps_marked = ""

            for step in list_of_steps[i]:
                text_with_steps_marked += f"{step} {self.step_eos}"

            message = [
                {"role": "user", "content": questions[i]},
                {"role": "assistant", "content": text_with_steps_marked},
            ]
            input = self.tokenizer.apply_chat_template(message, tokenize=False)
            inputs.append(input)

        # tokenize data for the RM
        batch = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        return batch
