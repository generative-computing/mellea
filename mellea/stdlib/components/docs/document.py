"""``Document`` component for grounding model inputs with text passages.

``Document`` wraps a text passage with an optional ``title`` and ``doc_id``, and
renders them inline as a formatted citation string for the model. Documents are
typically attached to a ``Message`` via its ``documents`` parameter, enabling
retrieval-augmented generation (RAG) workflows.
"""

import collections.abc
import warnings

from ....core import CBlock, Component, ModelOutputThunk, TemplateRepresentation


class Document(Component[str]):
    """A text passage with optional metadata for grounding model inputs.

    Documents are typically attached to a ``Message`` via its ``documents``
    parameter to enable retrieval-augmented generation (RAG) workflows.

    Args:
        text (str): The text content of the document.
        title (str | None): An optional human-readable title for the document.
        doc_id (str | None): An optional unique identifier for the document.

    """

    def __init__(self, text: str, title: str | None = None, doc_id: str | None = None):
        """Initialize Document with text content and optional title and ID."""
        self.text = text
        self.title = title
        self.doc_id = doc_id

    def parts(self) -> list[Component | CBlock]:
        """Returns the constituent parts of this document.

        Returns:
            list[Component | CBlock]: An empty list by default since the base
            ``Document`` class has no constituent parts. Subclasses may override
            this method to return meaningful parts.
        """
        return []

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the `Document` as a ``TemplateRepresentation``.

        Returns: a TemplateRepresentation with text, title, and doc_id args.
        """
        return TemplateRepresentation(
            obj=self,
            args={"text": self.text, "title": self.title, "doc_id": self.doc_id},
            template_order=["*", "Document"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""


def _coerce_to_documents(
    documents: collections.abc.Iterable[str | Document] | None,
    *,
    auto_doc_id: bool = False,
) -> list[Document] | None:
    """Convert an iterable of strings or Documents into a list of Documents.

    Args:
        documents: Strings, Document objects, a mix, or None.
        auto_doc_id: When True, assign sequential string doc_id values
            ("0", "1", ...) to Documents created from strings and warn about
            existing Document objects that have no ``doc_id`` set.

    Returns:
        A list of Document objects, or None if the input was None.
    """
    if documents is None:
        return None
    result: list[Document] = []
    for i, d in enumerate(documents):
        if isinstance(d, str):
            doc_id = str(i) if auto_doc_id else None
            result.append(Document(text=d, doc_id=doc_id))
        else:
            if auto_doc_id and d.doc_id is None:
                warnings.warn(
                    f"Document at index {i} has no doc_id; results may omit "
                    "document identification. Set doc_id on the Document or "
                    "pass a plain string to auto-generate one.",
                    UserWarning,
                    stacklevel=3,
                )
            result.append(d)
    return result


def _coerce_to_document(document: str | Document) -> Document:
    """Convert a single string or Document into a Document."""
    if isinstance(document, str):
        return Document(text=document)
    return document
