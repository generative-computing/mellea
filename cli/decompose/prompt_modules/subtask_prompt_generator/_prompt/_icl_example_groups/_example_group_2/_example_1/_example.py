from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "1. Understand the prompt and constraints, ensuring clarity on the sequence of actions and the narrative tone.",
    "subtask_prompt_template": "",
}

example["execution_plan"] = [
    "1. Understand the prompt and constraints, ensuring clarity on the sequence of actions and the narrative tone. - Variable: PROMPT_UNDERSTANDING",
    "2. Brainstorm the narrative elements, focusing on the introspective tone and the sequence of actions described in the prompt. - Variable: NARRATIVE_BRAINSTORMING",
    "3. Incorporate the bad weather element into the story, ensuring it fits naturally within the narrative. - Variable: WEATHER_INCORPORATION",
    "4. Write the four-sentence story, ensuring it meets all the given constraints and maintains the introspective narrative tone. - Variable: STORY_WRITING",
    "5. Review the story to ensure it adheres to the constraints, make any necessary adjustment and output the four-sentence story text only without additional information as instructed by the task. - Variable: REVIEWED_STORY",
]

example["available_content_variables"] = []

example[
    "subtask_prompt_template"
] = r"""Your task is to analyze the given prompt to understand the scenario for writing a four-sentence story. Follow these steps to accomplish your task:

First, carefully read the prompt provided below:
<prompt>
Lost, found vodka, drank to forget.
</prompt>

Next, break down the prompt into its key components:
- Identify the sequence of events: being lost, finding vodka, and drinking to forget
- Understand the emotional state and motivations behind these actions

Consider the implications of each action:
- What does it mean to be lost? Is it physical, emotional, or both?
- Why was vodka found, and what led to the decision to drink it?
- What is the character trying to forget, and why?

Understand the constraints:
- The story should describe a character without using the word "man"
- The narrative tone should be introspective
- The story should include a description of bad weather

Ensure you grasp the introspective narrative tone, which involves reflecting on inner thoughts and feelings.

Finally, summarize your analysis to ensure you have a clear understanding of the scenario, the sequence of actions and the narrative tone. This summary will serve as the basis for the next steps in developing the story."""
