# SPDX-License-Identifier: Apache-2.0

"""Constants used in code that is specific to the Granite 3.3 family of models."""

# Complete set of special tokens from the Granite 3.3 tokenizer
ALL_SPECIAL_TOKENS = [
    "<|end_of_text|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>",
    "<|start_of_role|>",
    "<|end_of_role|>",
    "<|tool_call|>",
    "<|start_of_cite|>",
    "<|end_of_cite|>",
    "<|start_of_plugin|>",
    "<|end_of_plugin|>",
]

# Delimiters for chain of thought output of Granite 3.3
COT_START = "<think>"
COT_END = "</think>"
RESPONSE_START = "<response>"
RESPONSE_END = "</response>"


# Delimiters for hallucination and citation output of Granite 3.3
CITATIONS_START = '{"id": "citation"}'
HALLUCINATIONS_START = '{"id": "hallucination"}'
CITE_START = "<|start_of_cite|>"
CITE_END = "<|end_of_cite|>"


# List of variations of the chain of thought start/end strings that the model is known
# to produce.
COT_START_ALTERNATIVES = [COT_START]
COT_END_ALTERNATIVES = [COT_END]

# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are both tools and RAG documents in the current request.
TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART = """\
 You are a helpful assistant with access to the following tools. When a tool is \
required to answer the user's query, respond only with <|tool_call|> followed by a \
JSON list of tools used. If a tool does not exist in the provided list of tools, \
notify the user that you do not have the ability to fulfill the request. \

Write the response to the user's input by strictly aligning with the facts in the \
provided documents. If the information needed to answer the question is not available \
in the documents, inform the user that the question cannot be answered based on the \
available data."""


# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are no tools or documents in the current request and the "thinking" flag is
# set to `True`.
NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART = f"""\
 You are a helpful AI assistant.
Respond to every user query in a comprehensive and detailed way. You can write down \
your thoughts and reasoning process before responding. In the thought process, engage \
in a comprehensive cycle of analysis, summarization, exploration, reassessment, \
reflection, backtracing, and iteration to develop well-considered thinking process. \
In the response section, based on various attempts, explorations, and reflections from \
the thoughts section, systematically present the final solution that you deem correct. \
The response should summarize the thought process. Write your thoughts between \
{COT_START}{COT_END} and write your response between \
{RESPONSE_START}{RESPONSE_END} \
for each user query."""

# String that a Granite 3.3 model must receive immediately after either
# TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE  (if there are tools) or
# NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are no tools) in the system prompt
# if the "citations" flag is `True` and there are documents.
DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART = """ \

Use the symbols <|start_of_cite|> and <|end_of_cite|> to indicate when a fact comes \
from a document in the search result, e.g <start_of_cite> {document_id: 1}my fact \
<end_of_cite> for a fact from document 1. Afterwards, \
list all the citations with their corresponding documents in an ordered list."""

# String that a Granite 3.3 model must receive immediately after either
# TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are tools and no citations) or
# NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are no tools or citations) or
# DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART in the system prompt
# if the "hallucinations" flag is `True` and there are documents.
# Note that a list of zero documents counts as "having documents".
DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART = """\

Finally, after the response is written, include a numbered list of sentences from the \
response with a corresponding risk value that are hallucinated and not based in the \
documents."""

# String that a Granite 3.3 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are tools in the current request but there are no documents in the current
# request.
TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART = """\
 You are a helpful assistant with access to the following tools. When a tool is \
required to answer the user's query, respond only with <|tool_call|> followed by a \
JSON list of tools used. If a tool does not exist in the provided list of tools, \
notify the user that you do not have the ability to fulfill the request."""

MODEL_NAME = "Granite 3.3"
MODEL_HF_PATH_2B = "ibm-granite/granite-3.3-2b-instruct"
