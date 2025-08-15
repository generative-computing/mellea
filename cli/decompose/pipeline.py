from typing import NotRequired, TypedDict

from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.prompt_modules import (
    constraint_extractor,
    subtask_constraint_assign,
    subtask_list,
    subtask_prompt_generator,
)
from mellea.prompt_modules.subtask_constraint_assign import SubtaskPromptConstraintsItem
from mellea.prompt_modules.subtask_list import SubtaskItem
from mellea.prompt_modules.subtask_prompt_generator import SubtaskPromptItem
from mellea.stdlib.instruction import Instruction


class DecompSubtasksResult(TypedDict):
    subtask: str
    tag: str
    constraints: list[str]
    prompt_template: str
    generated_response: NotRequired[str]


class DecompPipelineResult(TypedDict):
    original_task_prompt: str
    subtask_list: list[str]
    subtasks: list[DecompSubtasksResult]
    final_response: NotRequired[str]


def decompose(
    task_prompt: str, user_input_variable: list[str] | None = None
) -> DecompPipelineResult:
    if user_input_variable is None:
        user_input_variable = []

    m_ollama_session = MelleaSession(
        OllamaModelBackend(
            # model_id="llama3.2:3b-instruct-fp16",
            # model_id="granite3.3:8b",
            # model_id="llama3.1:8b",
            # model_id="llama3.1:8b-instruct-fp16",
            model_id="mistral-small3.2:24b"
        )
    )

    subtasks: list[SubtaskItem] = subtask_list.generate(
        m_ollama_session, task_prompt
    ).parse()

    task_prompt_constraints: list[str] = constraint_extractor.generate(
        m_ollama_session, task_prompt
    ).parse()

    subtask_prompts: list[SubtaskPromptItem] = subtask_prompt_generator.generate(
        m_ollama_session,
        task_prompt,
        user_input_var_names=user_input_variable,
        subtasks_and_tags=subtasks,
    ).parse()

    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = (
        subtask_constraint_assign.generate(
            m_ollama_session,
            subtasks_tags_and_prompts=subtask_prompts,
            constraint_list=task_prompt_constraints,
        ).parse()
    )

    decomp_subtask_result: list[DecompSubtasksResult] = [
        DecompSubtasksResult(
            subtask=subtask_data.subtask,
            tag=subtask_data.tag,
            constraints=subtask_data.constraints,
            prompt_template=subtask_data.prompt_template,
        )
        for subtask_data in subtask_prompts_with_constraints
    ]

    return DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[item.subtask for item in subtasks],
        subtasks=decomp_subtask_result,
    )


# def inference():
#     for i, subtask in enumerate(result_data["subtask_data"]):
#         subtask_constraint_list = "\n".join(
#             ["- " + constraint for constraint in subtask["constraints"]]
#         ).strip()

#         subtask_prompt = (
#             subtask["prompt"]
#             + "\n\nMake sure to meet the following constraints when writing your answer:\n"
#             + subtask_constraint_list
#         )

#         subtask_instruction = Instruction(
#             description=subtask_prompt,
#             user_variables=(
#                 # {
#                 #     "INPUT_DATA": "Market research on comic books enthusiasts, customer feedback on comic book events and stores, and user data on reading habits, and ratings on each franchise."
#                 # } |
#                 # {
#                 #     "YOUR_NAME": "Tulio Coppola",
#                 #     "YOUR_COMPANY": "IBM",
#                 #     "PROSPECT_NAME": "Mrs. Ana Saints",
#                 #     "PROSPECT_ROLE": "Engineering Chief for AI Products",
#                 #     "PROSPECT_COMPANY": "Santander",
#                 #     "YOUR_PRODUCT": "watsonx.ai",
#                 #     "PRODUCT_DESCRIPTION": "IBM watsonx.ai is a unified AI platform introduced in late 2021, positioned as IBM's flagship offering to democratize AI across enterprises. It provides a centralized environment for AI model development, deployment, and management, supporting both technical and non-technical users. Key capabilities include a model hub for accessing and sharing models, no-code/low-code development tools, and robust governance features ensuring data privacy and model explainability. watsonx.ai is built on open standards, compatible with popular frameworks like TensorFlow and PyTorch, and operable across multiple cloud environments, addressing the need for flexibility and integration in modern AI deployments.\n\nwatsonx.ai primarily aims to solve several critical challenges in AI adoption and management within organizations. It simplifies the complexity of AI development through accessible, no-code/low-code tools, broadening participation beyond data scientists. The platform also tackles model fragmentation and management issues by centralizing models in a shared workspace, facilitating collaboration and version control. Furthermore, watsonx.ai addresses the need for scalability and compliance in multi-cloud environments, integrating robust data privacy controls and explainable AI techniques to ensure regulatory compliance and build trust in AI systems. By offering these comprehensive capabilities, watsonx.ai seeks to make AI more accessible, manageable, and effective for a wide range of enterprise use cases.",
#                 # } |
#                 {
#                     result_data["subtask_data"][j]["tag"]: result_data["subtask_data"][
#                         j
#                     ]["generated_answer"]
#                     for j in range(i)
#                 }
#             ),
#         )

#         result_data["subtask_data"][i][
#             "populated_prompt"
#         ] = subtask_instruction._description.__str__()

#         subtask_generated_answer = exec_m_session.backend.generate_from_context(
#             action=subtask_instruction,
#             ctx=exec_m_session.ctx,
#             model_options={
#                 ModelOption.TEMPERATURE: 0,
#                 ModelOption.MAX_NEW_TOKENS: 8192,
#                 # ModelOption.MAX_NEW_TOKENS: 16384,
#             },
#         ).value

#         result_data["subtask_data"][i]["generated_answer"] = subtask_generated_answer

#         if i == len(result_data["subtask_data"]) - 1:
#             result_data["final_generated_answer"] = subtask_generated_answer
