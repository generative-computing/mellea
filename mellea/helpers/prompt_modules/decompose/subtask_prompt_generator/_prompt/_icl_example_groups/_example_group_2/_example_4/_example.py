from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "4. Write the four-sentence story, ensuring it meets all the given constraints and maintains the introspective narrative tone.",
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
]

example[
    "subtask_prompt_template"
] = r"""Your task is to write a four-sentence story that describes a character without using the word "man" anywhere. The story should be written in an introspective narrative tone and must mention a bad weather.

To accomplish this, follow these steps:

1. **Understand the Constraints and Tone**:
    Ensure you have a clear understanding of the constraints and the introspective narrative tone required for the story. You can refer to the understanding of the prompt from the previous step:
    <prompt_understanding>
    {{PROMPT_UNDERSTANDING}}
    </prompt_understanding>

2. **Incorporate Narrative Elements**:
    Use the brainstormed narrative elements to guide your writing. These elements should focus on the sequence of actions described in the prompt and maintain the introspective tone:
    <narrative_brainstorming>
    {{NARRATIVE_BRAINSTORMING}}
    </narrative_brainstorming>

3. **Include the Weather Element**:
    Ensure that the bad weather is naturally incorporated into the story. Refer to the weather incorporation notes from the previous step:
    <weather_incorporation>
    {{WEATHER_INCORPORATION}}
    </weather_incorporation>

4. **Write the Story**:
    Craft a four-sentence story that meets all the given constraints. The story should describe a character without using the word "man," maintain an introspective narrative tone, and include a description of bad weather.

Here is an example structure to guide your writing:
- **Sentence 1**: Introduce the character and the introspective tone.
- **Sentence 2**: Describe the sequence of actions related to losing something.
- **Sentence 3**: Incorporate the element of finding vodka and the bad weather.
- **Sentence 4**: Conclude with the action of drinking to forget, maintaining the introspective tone.

Ensure that each sentence flows naturally and adheres to the constraints and tone specified in the prompt. You should write only the story, do not include the guidance structure."""
