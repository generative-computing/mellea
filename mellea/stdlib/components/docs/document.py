"""Document component."""

from ....core import CBlock, Component, ModelOutputThunk


# TODO: Add support for passing in docs as model options.
class Document(Component[str]):
    """Documents should typically be used in a Message object."""

    def __init__(self, text: str, title: str | None = None, doc_id: str | None = None):
        """Create a document object. Should typically be used as a list in the `_docs` field of Message."""
        self.text = text
        self.title = title
        self.doc_id = doc_id

    def parts(self) -> list[Component | CBlock]:
        """The set of all the constituent parts of the `Component`."""
        # TODO: JAL. Text should be a cblock here. And it should be able to passed in as one too during initialization.
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
