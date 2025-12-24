"""Representations of Docling Documents."""

from __future__ import annotations

import io
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument, TableItem
from docling_core.types.io import DocumentStream

from mellea.stdlib.base import CBlock, Component, TemplateRepresentation
from mellea.stdlib.mobject import MObject, Query, Transform


class RichDocument(Component):
    """A `RichDocument` is a block of content with an underlying DoclingDocument.

    It has helper functions for working with the document and extracting parts of it.
    """

    def __init__(self, doc: DoclingDocument):
        """A `RichDocument` is a block of content with an underlying DoclingDocument."""
        self._doc = doc

    def parts(self) -> list[Component | CBlock]:
        # TODO: we could separate a DoclingDocument into chunks and then treat those chunks as parts.
        # for now, do nothing.
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Return Document content as Markdown.

        No template needed here.
        """
        return self.to_markdown()

    def docling(self) -> DoclingDocument:
        """Get the underlying Docling Document."""
        return self._doc

    def to_markdown(self):
        """Get the full text of the document as markdown."""
        return self._doc.export_to_markdown()

    def get_tables(self) -> list[Table]:
        """Return the `Table`s that are a part of this document."""
        return [Table(x, self.docling()) for x in self.docling().tables]

    def save(self, filename: str | Path) -> None:
        """Save the underlying DoclingDocument for reuse later."""
        if type(filename) is str:
            filename = Path(filename)
        self._doc.save_as_json(filename)

    @classmethod
    def load(cls, filename: str | Path) -> RichDocument:
        """Load a DoclingDocument from a file. The file must already be a DoclingDocument."""
        if type(filename) is str:
            filename = Path(filename)
        doc_doc = DoclingDocument.load_from_json(filename)
        return cls(doc_doc)

    @classmethod
    def from_document_file(cls, source: str | Path | DocumentStream) -> RichDocument:
        """Process a document with Docling."""
        pipeline_options = PdfPipelineOptions(
            images_scale=2.0, generate_picture_images=True
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(source)
        return cls(result.document)


class TableQuery(Query):
    """Table-specific query."""

    def __init__(self, obj: Table, query: str) -> None:
        """Initializes a new instance of the `TableQuery` class.

        Args:
            obj : The table object to which the query applies.
            query : The query string.
        """
        super().__init__(obj, query)
    
    def parts(self):
        return [ self.obj ]

    def format_for_llm(self) -> TemplateRepresentation:
        """Template arguments for Formatter."""
        assert isinstance(self._obj, Table)
        tbl_repr = self._obj.format_for_llm()
        assert isinstance(tbl_repr, TemplateRepresentation)
        return TemplateRepresentation(
            args={"query": self._query, "table": self._obj.to_markdown()},
            obj=self,
            tools=tbl_repr.tools,
            fields=tbl_repr.fields,
            template_order=["TableQuery", "Query"],
        )


class TableTransform(Transform):
    """Table-specific transform."""

    def __init__(self, obj: Table, transformation: str) -> None:
        """Initializes a new instance of the `TableTransform` class.

        Args:
            obj : The table object to which the transform applies.
            transformation : The transformation description string.
        """
        super().__init__(obj, transformation)

    def format_for_llm(self) -> TemplateRepresentation:
        """Template arguments for Formatter."""
        assert isinstance(self._obj, Table)
        tbl_repr = self._obj.format_for_llm()
        assert isinstance(tbl_repr, TemplateRepresentation)
        return TemplateRepresentation(
            args={
                "transformation": self._transformation,
                "table": self._obj.to_markdown(),
            },
            obj=self,
            tools=tbl_repr.tools,
            fields=tbl_repr.fields,
            template_order=["TableTransform", "Transform"],
        )


class Table(MObject):
    """A `Table` represents a single table within a larger Docling Document."""

    def __init__(self, ti: TableItem, doc: DoclingDocument):
        """If you pass doc=None, the underlying docling functions to extract data from tables may fail due to lack of context and docling deprecations."""
        super().__init__(query_type=TableQuery, transform_type=TableTransform)
        self._ti = ti
        self._doc = doc

    @classmethod
    def from_markdown(cls, md: str) -> Table | None:
        """Creates a fake document from the markdown and attempts to extract the first table found."""
        fake_doc = f"# X\n\n{md}\n"
        bs = io.BytesIO(fake_doc.encode("utf-8"))
        doc = RichDocument.from_document_file(DocumentStream(name="x.md", stream=bs))
        if len(doc.get_tables()) > 0:
            return doc.get_tables()[0]
        else:
            return None

    def to_markdown(self) -> str:
        """Get the `Table` as markdown."""
        return self._ti.export_to_markdown(self._doc)

    def transpose(self) -> Table | None:
        """Transposes the table. Will return a new transposed `Table` if successful."""
        t = self._ti.export_to_dataframe().transpose()
        return Table.from_markdown(t.to_markdown())

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Return Table representation for Formatter."""
        return TemplateRepresentation(
            args={"table": self.to_markdown()},
            obj=self,
            tools=self._get_all_members(),
            fields=[],
            template="{{table}}",
            template_order=None,
        )
