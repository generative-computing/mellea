# ruff: noqa E402
# Example: Rich Documents and Templating
# Lets look at how to pass documents to a model using `Mellea`.

# 1. We'll start by initializing our `Mellea` session.
from docling_core.types.doc.document import DoclingDocument

import mellea
from mellea.core import ModelOutputThunk, TemplateRepresentation

# Use a `SimpleContext` so that each LLM call is independent.
m = mellea.start_session(backend_name="hf")

# 2. Let's import docling so that we can process pdf documents.

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

source = "https://arxiv.org/pdf/1906.04043"
pipeline_options = PdfPipelineOptions(images_scale=2.0, generate_picture_images=True)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

result = converter.convert(source)
doc = result.document

# 3. Now that we have the document, we can pass it to the model to generate
# from it.
msg_text = doc.export_to_markdown() + "\nSummarize the provided document."
response = m.chat(msg_text)
print(response.content)  # > The paper introduces...

# 4. `Mellea` also provides a basic wrapper around this functionality to make
# basic processing of documents easier.
from mellea.stdlib.components.docs import RichDocument

# This creates a new `Mellea` RichDocument component that encapsulates all
# the logic above along with some convenient helpers.
rd = RichDocument.from_document_file(source)

# Note: Because the template for a RichDocument just outputs it as markdown,
# the model doesn't really know what to do with it in this context. However, this
# is a useful pattern if you want to use a component with a specified template.
thunk = m.act(action=rd)
print(thunk.value)  # > - user: What is the primary goal of the GLTR tool...

# 5. The class is opinionated and outputs the document as markdown to the model (like in the initial example).
# However, we can define our own class that modifies that behavior by redefining the template and template args.
from pathlib import Path

from docling_core.types.doc.document import SectionHeaderItem
from docling_core.types.io import DocumentStream


class RichDocumentSections(RichDocument):
    def format_for_llm(self) -> TemplateRepresentation:
        titles = [t.text for t in self._doc.texts if isinstance(t, SectionHeaderItem)]

        # Usually templates are defined in the `templates/` directory for the default TemplateFormatter,
        # but we can also define them at the class (and even object) level. Here's an example jinja template:
        template = (
            "{%- if titles|length > 0 -%}"
            "Guess the topic of and the type of the document from the following titles:\n"
            "{% for title in titles %}"
            "\n* {{ title }}"
            "{% endfor %}"
            "{%- endif -%}"
        )

        return TemplateRepresentation(
            obj=self, args={"titles": titles}, template=template
        )

    @classmethod
    def from_document_file(
        cls, source: str | Path | DocumentStream
    ) -> "RichDocumentSections":
        rd = RichDocument.from_document_file(source)
        return RichDocumentSections(rd.docling())


rds = RichDocumentSections.from_document_file(source)
print(
    rds.format_for_llm().args
)  # > {'titles': ['GLTR: Statistical Detection and Visualization of Generated Text', 'Abstract', ..., 'References']}

thunk = m.act(action=rds)
print(thunk.value)  # > The document appears to be an academic research paper...

# 6. We can also pass this document as grounding context to an instruction.
# When outputting the instruction to the model, the default TemplateFormatter for the
# backend will output the templated version of the RichDocumentSections component as part of the
# grounding context, using the RichDocumentSections' template and the Instruction's template.
# For this specific example, the model will get a user message looking like:
#   Write the abstract for [GLTR_Paper] based on its document section titles.
#   Here is some grounding context:
#   *   [GLTR_Paper] =  Guess the topic of and the type of the document from the following titles:\n\n* GLTR: Statistical Detection and Visualization of Generated Text\n* Abstract\n* Human-Written\n* 1 Introduction\n* Generated\n* 2 Method\n* 3 GLTR: Visualizing Outliers\n* 4 Empirical Validation\n* 5 Human-Subjects Study\n* 6 Related Work\n* 7 Discussion and Conclusion\n* Acknowledgments\n* References\n\n
instr_thunk = m.instruct(
    "Write the abstract for [GLTR_Paper] based on its document section titles.",
    grounding_context={"GLTR_Paper": rds},
)

# Since we aren't passing a sampling strategy, we know that the returned output is of type ModelOutputThunk.
if isinstance(instr_thunk, ModelOutputThunk):
    print(
        instr_thunk.value
    )  # > Based on the provided section titles of [GLTR_Paper]...
