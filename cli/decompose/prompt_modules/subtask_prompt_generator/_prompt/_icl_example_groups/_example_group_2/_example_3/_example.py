from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "3. Incorporate the bad weather element into the story, ensuring it fits naturally within the narrative.",
    "subtask_prompt_template": "",
}

example["execution_plan"] = [
    "1. Understand the prompt and constraints, ensuring clarity on the sequence of actions and the narrative tone. - Variable: PROMPT_UNDERSTANDING",
    "2. Brainstorm the narrative elements, focusing on the introspective tone and the sequence of actions described in the prompt. - Variable: NARRATIVE_BRAINSTORMING",
    "3. Incorporate the bad weather element into the story, ensuring it fits naturally within the narrative. - Variable: WEATHER_INCORPORATION",
    "4. Write the four-sentence story, ensuring it meets all the given constraints and maintains the introspective narrative tone. - Variable: STORY_WRITING",
    "5. Review the story to ensure it adheres to the constraints, make any necessary adjustment and output the four-sentence story text only without additional information as instructed by the task. - Variable: REVIEWED_STORY",
]

example["available_content_variables"] = [
    r"{{PROMPT_UNDERSTANDING}}",
    r"{{NARRATIVE_BRAINSTORMING}}",
]

example[
    "subtask_prompt_template"
] = r"""Your task is to incorporate the bad weather element into the story, ensuring it fits naturally within the narrative. Follow these steps to accomplish your task:

First, review the brainstormed narrative elements from the previous step:
<narrative_brainstorming>
{{NARRATIVE_BRAINSTORMING}}
</narrative_brainstorming>

Next, consider how the bad weather can be integrated into the story. The weather should not just be mentioned but should play a role in the narrative, affecting the actions or emotions of the character.

Think about how the bad weather can enhance the introspective tone of the story. For example, the weather could reflect the character's inner turmoil or provide a backdrop that amplifies the character's feelings of being lost.

Ensure that the incorporation of the bad weather element is seamless and adds depth to the story. The weather should feel like a natural part of the narrative rather than an afterthought.

Finally, write a brief description of how you plan to incorporate the bad weather into the story. This description will serve as a guide for the next step, which is writing the four-sentence story."""
