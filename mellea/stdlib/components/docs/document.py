"""``Document`` component for grounding model inputs with text passages.

``Document`` wraps a text passage with an optional ``title`` and ``doc_id``, and
renders them inline as a formatted citation string for the model. Documents are
typically attached to a ``Message`` via its ``documents`` parameter, enabling
retrieval-augmented generation (RAG) workflows.
"""

from ....core import CBlock, Component, ModelOutputThunk


# TODO: Add support for passing in docs as model options.
class Document(Component[str]):
    """A text passage with optional metadata for grounding model inputs.

    Documents are typically attached to a ``Message`` via its ``documents``
    parameter to enable retrieval-augmented generation (RAG) workflows.

    Args:
        text (str): The text content of the document.
        title (str | None): An optional human-readable title for the document.
        doc_id (str | None): An optional unique identifier for the document.

    Attributes:
        text (str): The text content of this document.
        title (str | None): Human-readable title, or ``None`` if not set.
        doc_id (str | None): Unique identifier, or ``None`` if not set.
    """

    def __init__(self, text: str, title: str | None = None, doc_id: str | None = None):
        """Create a document object. Should typically be used as a list in the ``_docs`` field of ``Message``.

        Args:
            text (str): The text content of the document.
            title (str | None): An optional human-readable title.
            doc_id (str | None): An optional unique identifier.
        """
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

    def format_for_llm(self) -> str:
        """Formats the `Document` into a string.

        Returns: a string
        """
        doc = ""
        if self.doc_id is not None:
            doc += f"document ID '{self.doc_id}': "
        if self.title is not None:
            doc += f"'{self.title}': "
        doc += f"{self.text}"

        return doc

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
