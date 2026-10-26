# pytest: skip_always
"""RAG over a PDF document using RichDocument and grounding context.

Demonstrates loading a remote PDF into a `RichDocument` component and using it
as `grounding_context` in `m.instruct()` for document-grounded question answering.

Prerequisites:
    uv sync --extra backends
    pip install docling
    ollama pull granite4.2:3b
    ollama serve

Usage:
    uv run python docs/examples/rag/mellea_pdf.py
"""

import mellea
from mellea.stdlib.components.docs.richdocument import RichDocument

m = mellea.start_session()

doc = RichDocument.from_document_file(
    "https://www.fs.usda.gov/foresthealth/docs/fidls/FIDL-78-ArmillariaRootDisease.pdf"
)

print(
    m.instruct(
        "Determine whether the mellea fungus only feeds on dead or weakened trees, thereby improving forest health",
        grounding_context={"usfs": doc},
    ).value
)
