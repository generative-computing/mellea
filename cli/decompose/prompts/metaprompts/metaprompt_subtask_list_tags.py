# flake8: noqa

metaprompt_subtask_list__system = """You will break down a long, complex task into a list of subtasks for a helpful but inexperienced AI assistant to execute in order. Your goal is to make the task easier to complete by structuring it into logically ordered, actionable subtasks.

You will generate:
1. A <Reasoning> section to reflect on the structure and needs of the task.
2. A <Draft Subtask List> where you write an initial set of subtasks, each tagged with a classification type from the list below.
3. A <Final Subtask List> with fewer, cleaner steps, each labeled with a variable name related to its purpose.

Use the following classification types in the draft list:
- "write"
- "extract"
- "present or show"
- "reason and reflect"
- "research"
- "assert and verify"
- "constraints and conditions"
- "format"
- "safety and security"

For the <Final Subtask List>:
- Use numbered steps (no subitems or multiline descriptions).
- Each step must describe a single, self-contained action.
- Assign each step a variable name using **UPPERCASE_WITH_UNDERSCORES** enclosed in `Variable: ...`.
- These variable names should reflect the intent or output of that step.

Here is an example:

<Final Subtask List>
1. Gather and understand the task input requirements. - Variable: INPUT_DATA
2. Conduct research or analysis using the task input. - Variable: RESEARCH
3. Write a structured result that satisfies the task requirements. - Variable: OUTPUT
</Final Subtask List>

Now see some full examples:

<Example>
  <Long Complex Task>
    Create a short, informative summary from a document that contains paragraphs of raw text. The summary should capture key ideas and tone of the original. Write the output in plain language and return it as a paragraph.
  </Long Complex Task>
  <Reasoning>
    The task requires understanding the document, extracting core content, rephrasing the core content in plain language, and writing a concise summary. We must also ensure that the output respects the tone and content focus of the original document.
  </Reasoning>
  <Draft Subtask List>
    1. Validate that the input document does not contain inappropriate or harmful content. - Category "safety and security"
    2. Extract the main themes and important ideas from the text. - Category "extract"
    3. Reflect on the tone and style of the input. - Category "reason and reflect"
    4. Write a short summary paragraph in plain language. - Category "write"
    5. Format the output for clean paragraph structure. - Category "format"
    6. Verify that the summary preserves original tone and covers key points. - Category "assert and verify"
  </Draft Subtask List>
  <Final Subtask List>
    1. Extract and reflect on the main ideas and tone from the document. - Variable: CONTENT_OVERVIEW
    2. Write a short, plain-language summary using the extracted information. - Variable: SUMMARY
  </Final Subtask List>
</Example>

<Example>
  <Long Complex Task>
    Analyze a dataset and create a visualization that shows the distribution of numerical values for a selected feature. Include descriptive labels and a brief caption explaining the visual.
  </Long Complex Task>
  <Reasoning>
    To complete this task, we need to first load and verify the dataset. Then we analyze the selected feature and generate a visualization like a histogram or boxplot. The visual must be labeled properly, and a clear caption must be added to interpret it.
  </Reasoning>
  <Draft Subtask List>
    1. Load the dataset and check its format and integrity. - Category "assert and verify"
    2. Extract the values of the selected numerical feature. - Category "extract"
    3. Generate a visual distribution chart. - Category "present or show"
    4. Write a descriptive caption that explains what the chart shows. - Category "write"
    5. Add labels and format the chart. - Category "format"
  </Draft Subtask List>
  <Final Subtask List>
    1. Extract the values for the selected feature from the dataset. - Variable: FEATURE_DATA
    2. Generate a chart and caption showing the feature’s distribution. - Variable: VISUALIZATION
  </Final Subtask List>
</Example>

That concludes the examples."""

metaprompt_subtask_list__user = """Now, here is the prompt of the long complex task for which I would like you to break down and write both subtask lists:

<Long Complex Task>
{{TASK}}
</Long Complex Task>

To write your subtask lists, follow THESE instructions:
1. In <Reasoning> tags, reason and think about the provided task and plan out how you will structure your subtasks in a correct order of execution. Always close the reasoning section with the </Reasoning> tag.
2. In <Draft Subtask List> tags, write your proposed draft subtask list. This draft subtask list should be similarly structured as the ones in the examples above. Always close the draft subtask list section with the </Draft Subtask List> tag.
3. Remember to classify each step on the <Draft Subtask List> under categories (types): "write", "extract", "present or show", "reason and reflect", "research", "assert and verify", "constraints and conditions", "format", "safety and security".
4. In <Final Subtask List> tags, write the subtask list. The subtask list should be based on the draft subtask list, but must have FEWER steps, you can have less steps by either grouping steps together or by omitting steps of specific categories that do not describe an actual action. This final subtask list should be similarly structured as the ones in the examples above. Always close the subtask list section with the </Final Subtask List> tag.
5. When writing the <Final Subtask List> try to group steps and remove non-actionable steps, but maintain the step's text very descriptive of its action.
6. Do not forget to always close each section with its corresponding close tag.
7. It is extremely important to always make sure both subtask lists are NUMBERED lists and that each item is a single-line, do not use new lines when writing the subtask lists and do not add subitems.

Note: This is probably obvious to you already, but you are not *completing* the task here. You are just writing a subtask list for an AI to follow and complete the task.
Note: The final <Final Subtask List> must omit steps of the categories such as "assert and verify", "present or show", "constraints and conditions", and "safety and security".
Note: When writing your <Final Subtask List>, you must try *MINIMIZE* the number of steps in the final list.
Note: The final <Final Subtask List> must NOT include the categories in the list, but it should include the "Variable" name based on the step's content.

Important: Do not forget to always close the <Final Subtask List> tag with "</Final Subtask List>" in the last line of your answer.
"""
