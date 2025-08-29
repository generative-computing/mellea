from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "Review the story to ensure it adheres to the constraints, make any necessary adjustment and output the four-sentence story text only without additional information as instructed by the task.",
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
    r"{{WEATHER_INCORPORATION}}",
    r"{{STORY_WRITING}}",
]

example[
    "subtask_prompt_template"
] = r"""Your task is to review the story to ensure it adheres to the prompt and constraints, making any necessary adjustments and answering the four-sentence story text only without additional information. Follow these steps to accomplish your task:

First, review the original prompt for clarity:
<original_prompt>
Prompt: Lost, found vodka, drank to forget.

According to the above prompt, write a four-sentence story that describes a man. However, the word "man" should not appear in the story. Please write using a introspective narrative tone. You should also describe something about the bad weather.
</original_prompt>

Next, review the story written in the previous step:
<written_story>
{{STORY_WRITING}}
</written_story>

Ensure the story meets the following constraints:
1. The story should be four sentences long.
2. The word "man" should NOT appear in the story.
3. The narrative tone should be introspective.
4. The story should mention a bad weather.
5. The sequence of actions should follow the prompt: Lost, found vodka, drank to forget.

If the story does not meet any of the above constraints, make the necessary adjustments to ensure it adheres to the prompt and constraints.

Finally, provide the revised four-sentence story text only without additional information as your answer."""
