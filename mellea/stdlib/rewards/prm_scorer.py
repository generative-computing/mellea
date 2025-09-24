from mellea.backends.huggingface import HFProcessRewardModel
from mellea.stdlib.base import CBlock, LegacyContext
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import ScorerRequirement, ValidationResult


class PRMScorer(ScorerRequirement):
    """A process reward model scorer based on local huggingface backend."""

    def __init__(
        self, *, prm_model: HFProcessRewardModel, preference_ordering: str = "max"
    ):
        """

        Args:
            prm_model:  The PRM model
            preference_ordering: indicates whether the goal is to maximize or minimize the score. must be either "max" or "min"
        """
        super().__init__(
            check_only=True,
            validation_fn=lambda c: self._prm_validate(c),
            preference_ordering=preference_ordering,
        )

        self.model: HFProcessRewardModel = prm_model

    def _prm_validate(self, ctx: LegacyContext):
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

        rewards, rewards_per_step = self.model.score(user_query, assistant_content)

        # return single reward item for the response
        assert len(rewards) == 1

        return ValidationResult(result=True, reason=None, score=rewards[0])
