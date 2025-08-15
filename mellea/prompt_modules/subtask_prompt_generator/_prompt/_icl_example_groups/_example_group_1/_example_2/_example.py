from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "2. Analyze the extracted information and generate at least two personas with the required properties.",
    "subtask_prompt_template": "",
}

example["execution_plan"] = [
    "1. Receive and validate the input data, then compile all relevant information for generating personas. - Variable: COMPILED_DATA",
    "2. Analyze the extracted and compiled information and generate at least two personas with the required properties. - Variable: PERSONAS_GENERATION",
    "3. Rewrite the generated personas using the Markdown format and respecting the provided constraints. - Variable: FORMATTED_PERSONAS",
    "4. Extract only the generated personas asked on the task and answer the user without any additional explanation information. - Variable: TASK_ANSWER",
]

example["available_content_variables"] = [r"{{INPUT_DATA}}", r"{{COMPILED_DATA}}"]

example[
    "subtask_prompt_template"
] = r"""Your task is to analyze the extracted and compiled information to generate at least two personas with the required properties.
Follow these steps to accomplish your task:

First, review the validated and compiled input data from the previous step:
<compiled_data>
{{COMPILED_DATA}}
</compiled_data>

Use the compiled data information to identify patterns, trends, and correlations that can help you create realistic personas.

Next, consider the required properties that each persona should have, including:
- **Name**
- **Age**
- **Occupation**
- **Demographics**
- **Goals**
- **Behaviors**
- **Pain Points**
- **Motivations**

Analyze the compiled data to determine the goals, behaviors, pain points, and motivations of the target audience. Identify common characteristics, such as age, occupation, and demographics, that can be used to create distinct personas.

Create at least two personas that reflect the diversity of the target audience. Ensure that each persona is fictional, yet realistic, and includes all the required properties.

Use the analyzed information to generate at least two personas that provide a comprehensive understanding of the target audience."""
