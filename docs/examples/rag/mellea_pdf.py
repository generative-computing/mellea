# pytest: e2e, ollama, slow

"""Answer a question grounded in W3C's historical WCAG 1.0 checklist."""

import mellea
from mellea.stdlib.components.docs.richdocument import RichDocument

doc = RichDocument.from_document_file(
    "https://www.w3.org/TR/1999/WAI-WEBCONTENT-19990505/full-checklist.pdf",
    do_ocr=False,
)

m = mellea.start_session()

answer = m.instruct(
    "Using only the provided WCAG 1.0 checklist, explain what Priority 1, "
    "Priority 2, and Priority 3 mean for a Web content developer. "
    "Answer with one sentence per priority level.",
    grounding_context={"wcag_checklist": doc},
).value

assert isinstance(answer, str)
assert answer.strip()
print(answer)
