from ..._types import ICLExample

example: ICLExample = {
    "execution_plan": [],
    "available_content_variables": [],
    "target_subtask": "1. Receive and validate the input data, then compile all relevant information for generating personas.",
    "subtask_prompt_template": "",
}

example["execution_plan"] = [
    "1. Receive and validate the input data, then compile all relevant information for generating personas. - Variable: COMPILED_DATA",
    "2. Analyze the extracted information and generate at least two personas with the required properties. - Variable: PERSONAS_GENERATION",
    "3. Rewrite the generated personas using the Markdown format and respecting the provided constraints. - Variable: FORMATTED_PERSONAS",
    "4. Extract only the generated personas asked on the task and answer the user without any additional explanation information. - Variable: TASK_ANSWER",
]

example["available_content_variables"] = [r"{{INPUT_DATA}}"]

example[
    "subtask_prompt_template"
] = r"""You are tasked with receiving and validating input data, then extracting relevant information to generate personas for Design Thinking sessions.

To approach this task, first, you must analyze the received input data below:
<input_data>
{{INPUT_DATA}}
</input_data>

Next, you must validate the input data to ensure it is accurate and relevant for generating personas.
Ensure that the input data content is safe, unbiased, and positive. Check for any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. If you detect any such content, flag it immediately and do not proceed with generating personas.

After validation, you will extract relevant information. The input data can contain user details, market research, customer feedback, and etc.

You can use the extracted information to identify patterns, trends, and insights that will help you generate fictional, yet realistic, personas for Design Thinking sessions.

Finally, you must compile the relevant data combined with your insights to write your final answer. Your answer will serve as basis for the next steps."""
