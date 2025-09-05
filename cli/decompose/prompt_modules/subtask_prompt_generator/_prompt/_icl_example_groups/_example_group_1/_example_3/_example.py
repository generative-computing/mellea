from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "3. Rewrite the generated personas using the Markdown format and respecting the provided constraints.",
    "subtask_prompt_template": "",
}

example["execution_plan"] = [
    "1. Receive and validate the input data, then compile all relevant information for generating personas. - Variable: COMPILED_DATA",
    "2. Analyze the extracted and compiled information and generate at least two personas with the required properties. - Variable: PERSONAS_GENERATION",
    "3. Rewrite the generated personas using the Markdown format and respecting the provided constraints. - Variable: FORMATTED_PERSONAS",
    "4. Extract only the generated personas asked on the task and answer the user without any additional explanation information. - Variable: TASK_ANSWER",
]

example["available_content_variables"] = [
    r"{{INPUT_DATA}}",
    r"{{COMPILED_DATA}}",
    r"{{PERSONAS_GENERATION}}",
]

example[
    "subtask_prompt_template"
] = r"""To format the generated personas in Markdown and present them in a clear and concise manner, you must use the personas generated in the previous step, which can be found below:
<generated_personas>
{{PERSONAS_GENERATION}}
</generated_personas>

You can also use as reference the compiled input data from the first step, which can be found below:
<compiled_data>
{{COMPILED_DATA}}
</compiled_data>

You must provide a comprehensive understanding of the target audience by including the following properties for each persona: name; age; occupation; demographics; goals; behaviors; pain points; motivations.

Use Markdown formatting to present the personas in a clear and organized way. You can use headings, lists, and bold text to make the personas easy to read and understand.

For each persona, you must include the following information:
- **Name**: The name of the persona
- **Age**: The age of the persona
- **Occupation**: The occupation of the persona
- **Demographics**: The demographics of the persona, including location, social class, and education level
- **Goals**: The goals and objectives of the persona
- **Behaviors**: The behaviors and habits of the persona
- **Pain Points**: The challenges and pain points of the persona
- **Motivations**: The motivations and desires of the persona

Follow all instructions above to rewrite the generated personas using the Markdown format and respecting the provided constraints."""
