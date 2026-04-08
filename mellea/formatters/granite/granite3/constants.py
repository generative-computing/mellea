# SPDX-License-Identifier: Apache-2.0

"""Constants for the Granite 3 family of models.

Constants used in code that is specific to the Granite 3 family of models, but not
specific to a particular point release.
"""

# String that a Granite 3.x model must receive immediately after _SYSTEM_MESSAGE_START
# if there are documents in the current request but there are no tools in the current
# request.
NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART = """\
 Write the response to the user's input by strictly aligning with the facts in the \
provided documents. If the information needed to answer the question is not available \
in the documents, inform the user that the question cannot be answered based on the \
available data."""


# String that a Granite 3.x model must receive immediately after _SYSTEM_MESSAGE_START
# if there are no tools or documents in the current request and the "thinking" flag is
# set to `False`.
NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART = """\
 You are a helpful AI assistant."""
