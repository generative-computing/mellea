from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "4. Extract only the generated personas asked on the task and answer the user without any additional explanation information.",
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
    r"{{FORMATTED_PERSONAS}}",
]

example[
    "subtask_prompt_template"
] = r"""Your task is to extract only the generated personas as required by the task and provide them as the final answer without any additional explanation or information.

To accomplish this, follow these steps:

1. Review the Formatted Personas:
Carefully review the formatted personas generated in the previous step. These personas should be presented in Markdown format and include all the required properties: name, age, occupation, demographics, goals, behaviors, pain points, and motivations:
<formatted_personas>
{{FORMATTED_PERSONAS}}
</formatted_personas>

2. Extract the Personas:
Identify and extract only the personas from the formatted content. Ensure that you do not include any additional explanations, introductions, or concluding remarks. The output should contain only the personas in Markdown format.

3. Provide the Final Answer:
Present the extracted personas as the final answer. Make sure the output is clear, concise, and adheres to the Markdown formatting guidelines provided in the original task prompt.

Remember, your goal is to provide a straightforward and clean output that includes only the generated personas."""
