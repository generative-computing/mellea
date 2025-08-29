from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "2. Brainstorm the narrative elements, focusing on the introspective tone and the sequence of actions described in the prompt.",
    "subtask_prompt_template": "",
}

example["execution_plan"] = [
    "1. Understand the prompt and constraints, ensuring clarity on the sequence of actions and the narrative tone. - Variable: PROMPT_UNDERSTANDING",
    "2. Brainstorm the narrative elements, focusing on the introspective tone and the sequence of actions described in the prompt. - Variable: NARRATIVE_BRAINSTORMING",
    "3. Incorporate the bad weather element into the story, ensuring it fits naturally within the narrative. - Variable: WEATHER_INCORPORATION",
    "4. Write the four-sentence story, ensuring it meets all the given constraints and maintains the introspective narrative tone. - Variable: STORY_WRITING",
    "5. Review the story to ensure it adheres to the constraints, make any necessary adjustment and output the four-sentence story text only without additional information as instructed by the task. - Variable: REVIEWED_STORY",
]

example["available_content_variables"] = [r"{{PROMPT_UNDERSTANDING}}"]

example[
    "subtask_prompt_template"
] = r"""Your task is to brainstorm the narrative elements for a four-sentence story based on the given prompt. Focus on maintaining an introspective narrative tone and adhering to the sequence of actions described in the prompt.

<given_prompt>
Lost, found vodka, drank to forget.
</given_prompt>

First, review the understanding of the prompt and constraints from the previous step:
<prompt_understanding>
{{PROMPT_UNDERSTANDING}}
</prompt_understanding>

Next, consider the sequence of actions described in the prompt: "Lost, found vodka, drank to forget." Break down these actions into individual narrative elements that can be expanded upon.

Then, focus on the introspective narrative tone. Think about the internal thoughts, feelings, and reflections that can be included in the story to create a deep and personal narrative.

Finally, brainstorm how these elements can be woven together to create a cohesive and engaging four-sentence story. Ensure that the narrative flows naturally and stays true to the introspective tone."""
