"""Classes and functions for working with document-like objects."""

from .document import Document

# Note: RichDocument, Table, TableQuery, TableTransform are not imported here
# by default to avoid heavy docling/torch/transformers imports at module load time.
# Import them explicitly from mellea.stdlib.components.docs.richdocument when needed.

__all__ = ["Document"]
