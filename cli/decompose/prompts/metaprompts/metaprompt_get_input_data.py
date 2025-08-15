# flake8: noqa

metaprompt_get_input_data__system = """You will analyze a task prompt to determine whether it requires user-provided input data (variables) in order to be executed effectively. Not all tasks need input data — some may be general knowledge questions or self-contained instructions.

The user input data / variables might be:
- Explicitly indicated using a templating format like {{VARIABLE_NAME}}
- Implicitly required, based on task understanding

Your objective is to identify all the necessary user input data required to complete the task and present them in a specific format.

Key Instructions:
- If input variables are present, list them **inside curly braces**, using **UPPERCASE_WITH_UNDERSCORES** format and **comma-separated**, **no spaces**.
- If no input data is needed, output `"N/A"` inside the `<User Input Data>` tags.
- Do not include anything else in your response except the tags and the input list.

---

<Example>
  <Task>
    Generate a formal letter for a recipient based on their name, address, and the purpose of the letter.
  </Task>
  <User Input Data>
    {{RECIPIENT_NAME}},{{RECIPIENT_ADDRESS}},{{LETTER_PURPOSE}}
  </User Input Data>
</Example>

<Example>
  <Task>
    Explain the concept of Newton’s Laws of Motion with examples.
  </Task>
  <User Input Data>
    N/A
  </User Input Data>
</Example>

<Example>
  <Task>
    Produce a summary from a provided article text.
  </Task>
  <User Input Data>
    {{ARTICLE_TEXT}}
  </User Input Data>
</Example>

That concludes the examples.
"""

metaprompt_get_input_data__user = """Now, here is the task prompt for which I would like you to identify and list all the user input data (variables) names:

<Task>
{{TASK}}
</Task>

To write your user input data (variables) list, follow THESE instructions:
1. Use your best judgement to create the user input data (variable) names.
2. The user input data (variable) names must be written surrounded by curly braces.
3. The user input data (variable) names must be written in uppercase letters and if the name is composed by multiple words, the words must be separated by a underscore character (_) instead o spaces.
4. In <User Input Data> tags, write your user input data (variables) list. This user input data (variables) list should be similarly structured as the ones in the examples above. Always close the user input data (variables) list section with the </User Input Data> tag.
5. If you judge that the provided task does not need user input data to be completed, then you must write "N/A" (without the quotes) inside the <User Input Data> tags and nothing else, just close the tags.
6. Do not forget to always close each section with its corresponding close tag.

Note: This is probably obvious to you already, but you are not *completing* the task here. You are just writing the user input data list for an AI to complete the task later.
Note: Remember that not all tasks need user input data, usually smaller tasks and tasks that are just asking for information don't require user input data, use your best judgment to identify those.
Note: You must write ONLY the user input data list, do not repeat the long complex task provided.
"""
