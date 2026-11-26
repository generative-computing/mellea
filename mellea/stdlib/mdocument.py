"""Representations of Docling Documents."""

from __future__ import annotations

import io
from pathlib import Path

from mellea.stdlib.base import CBlock, Component, TemplateRepresentation
from mellea.stdlib.mobject import MObject, Query, Transform

# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import *

def convert_into_mdocuments(path: str | Path | list[str] | list[Path], recursive: bool = False) -> list[MDocument]:
    raise NotImplementedError("convert_into_mdocuments")


class MDocument(Component):
    """A `MDocument` is a block of content with an underlying DoclingDocument.
    
    It has helper functions for loading, saving, writing, chunking, searching
    and editing the document.
    """

    def __init__(self, *, name: str, doc: None | DoclingDocument):
        """A `RichDocument` is a block of content with an underlying DoclingDocument."""

        self.name = name
        
        if doc is None:
            self._doc = DoclingDocument()    

    def load(self, filename: str | Path) -> bool:
        """Load the underlying DoclingDocument."""
        if type(filename) is str:
            filename = Path(filename)        
        self._doc.load_from_json(filename)
        
    def save(self, filename: str | Path) -> None:
        """Save the underlying DoclingDocument for reuse later."""
        if type(filename) is str:
            filename = Path(filename)
        self._doc.save_as_json(filename)
    
    def parts(self) -> list[Component | CBlock]: # Inherited from the Component protocol
        """The set of all the constituent parts of the document."""
        raise NotImplementedError("parts isn't implemented")

    def format_for_llm(self) -> TemplateRepresentation | str: # Inherited from the Component protocol
        """Formats the `Component` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` or string
        """
        raise NotImplementedError("format_for_llm isn't implemented")

    def table_of_contents(self) -> MDocument:
        raise NotImplementedError("isn't implemented")

    def insert(self, cref:RefItem, item: DocItem):
        raise NotImplementedError("isn't implemented")

    def replace(self, cref:RefItem, item: DocItem):
        raise NotImplementedError("isn't implemented")

    def append(self, cref:RefItem, item: DocItem):
        raise NotImplementedError("isn't implemented")    
