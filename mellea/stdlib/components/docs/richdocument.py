"""``RichDocument``, ``Table``, and related helpers backed by Docling.

``RichDocument`` wraps a ``DoclingDocument`` (e.g. produced by converting a PDF or
Markdown file) and renders it as Markdown for a language model. ``Table`` represents a
single table within a Docling document and provides ``transpose``, ``to_markdown``, and
query/transform helpers. Use ``RichDocument.from_document_file`` to convert a PDF or
other supported format, and ``get_tables()`` to extract structured table data for
downstream LLM-driven Q&A or transformation tasks.
"""

from __future__ import annotations

import io
from pathlib import Path

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc.document import DoclingDocument, TableItem
    from docling_core.types.io import DocumentStream
except ImportError as e:
    raise ImportError(
        "RichDocument requires extra dependencies. "
        'Please install them with: pip install "mellea[docling]"'
    ) from e

from ....backends.tools import MelleaTool
from ....core import CBlock, Component, ModelOutputThunk, TemplateRepresentation
from ..mobject import MObject, Query, Transform


class RichDocument(Component[str]):
    """A ``RichDocument`` is a block of content backed by a ``DoclingDocument``.

    Provides helper functions for working with the document and extracting parts
    such as tables. Use ``from_document_file`` to convert PDFs or other formats,
    and ``save``/``load`` for persistence.

    Args:
        doc (DoclingDocument): The underlying Docling document to wrap.
    """

    def __init__(self, doc: DoclingDocument):
        """Initialize RichDocument by wrapping the provided DoclingDocument."""
        self._doc = doc

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this document.

        Currently always returns an empty list. Future versions may support
        chunking the document into constituent parts.

        Returns:
            list[Component | CBlock]: Always an empty list.
        """
        # TODO: we could separate a DoclingDocument into chunks and then treat those chunks as parts.
        # for now, do nothing.
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Return the document content as a Markdown string.

        No template is needed; the full document is exported to Markdown directly.

        Returns:
            TemplateRepresentation | str: The full document rendered as Markdown.
        """
        return self.to_markdown()

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""

    def docling(self) -> DoclingDocument:
        """Return the underlying ``DoclingDocument``.

        Returns:
            DoclingDocument: The wrapped Docling document instance.
        """
        return self._doc

    def to_markdown(self):
        """Get the full text of the document as markdown."""
        return self._doc.export_to_markdown()

    def get_tables(self) -> list[Table]:
        """Return all tables found in this document.

        Returns:
            list[Table]: A list of ``Table`` objects extracted from the document.
        """
        return [Table(x, self.docling()) for x in self.docling().tables]

    def save(self, filename: str | Path) -> None:
        """Save the underlying ``DoclingDocument`` to a JSON file for later reuse.

        Args:
            filename (str | Path): Destination file path for the serialized
                document.
        """
        if type(filename) is str:
            filename = Path(filename)
        self._doc.save_as_json(filename)

    @classmethod
    def load(cls, filename: str | Path) -> RichDocument:
        """Load a ``RichDocument`` from a previously saved ``DoclingDocument`` JSON file.

        Args:
            filename (str | Path): Path to a JSON file previously created by
                ``RichDocument.save``.

        Returns:
            RichDocument: A new ``RichDocument`` wrapping the loaded document.
        """
        if type(filename) is str:
            filename = Path(filename)
        doc_doc = DoclingDocument.load_from_json(filename)
        return cls(doc_doc)

    @classmethod
    def from_document_file(
        cls, source: str | Path | DocumentStream, do_ocr: bool = True
    ) -> RichDocument:
        """Convert a document file to a ``RichDocument`` using Docling.

        Args:
            source (str | Path | DocumentStream): Path or stream for the
                source document (e.g. a PDF or Markdown file).
            do_ocr (bool): Whether to run OCR on the document. Disable for
                text-based PDFs to avoid downloading OCR model weights.

        Returns:
            RichDocument: A new ``RichDocument`` wrapping the converted document.
        """
        pipeline_options = PdfPipelineOptions(
            images_scale=2.0, generate_picture_images=True, do_ocr=do_ocr
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(source)
        return cls(result.document)


class TableQuery(Query):
    """A ``Query`` component specialised for ``Table`` objects.

    Formats the table as Markdown alongside the query string so the LLM receives
    both the structured table content and the natural-language question.

    Args:
        obj (Table): The table to query.
        query (str): The natural-language question to ask about the table.
    """

    def __init__(self, obj: Table, query: str) -> None:
        """Initialize TableQuery for the given table and natural-language query."""
        super().__init__(obj, query)

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this table query.

        Returns:
            list[Component | CBlock]: A list containing the wrapped ``Table``
            object.
        """
        cs: list[Component | CBlock] = [self._obj]
        return cs

    def format_for_llm(self) -> TemplateRepresentation:
        """Format this table query for the language model.

        Renders the table as Markdown alongside the query string, and forwards
        any tools and fields from the table's own representation.

        Returns:
            TemplateRepresentation: Template args containing the query string
            and the Markdown-rendered table.
        """
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
    """A ``Transform`` component specialised for ``Table`` objects.

    Formats the table as Markdown alongside the transformation instruction so the
    LLM receives both the structured table content and the mutation description.

    Args:
        obj (Table): The table to transform.
        transformation (str): Natural-language description of the desired mutation.
    """

    def __init__(self, obj: Table, transformation: str) -> None:
        """Initialize TableTransform for the given table and transformation description."""
        super().__init__(obj, transformation)

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this table transform.

        Returns:
            list[Component | CBlock]: A list containing the wrapped ``Table``
            object.
        """
        cs: list[Component | CBlock] = [self._obj]
        return cs

    def format_for_llm(self) -> TemplateRepresentation:
        """Format this table transform for the language model.

        Renders the table as Markdown alongside the transformation description,
        and forwards any tools and fields from the table's own representation.

        Returns:
            TemplateRepresentation: Template args containing the transformation
            description and the Markdown-rendered table.
        """
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
    """A ``Table`` represents a single table within a larger Docling Document.

    Args:
        ti (TableItem): The Docling ``TableItem`` extracted from the document.
        doc (DoclingDocument): The parent ``DoclingDocument``. Passing ``None``
            may cause downstream Docling functions to fail.
    """

    def __init__(self, ti: TableItem, doc: DoclingDocument):
        """Initialize Table by wrapping a Docling TableItem and its parent document."""
        super().__init__(query_type=TableQuery, transform_type=TableTransform)
        self._ti = ti
        self._doc = doc

    @classmethod
    def from_markdown(cls, md: str) -> Table | None:
        """Create a ``Table`` from a Markdown string by round-tripping through Docling.

        Wraps the Markdown in a minimal document, converts it with Docling, and
        returns the first table found.

        Args:
            md (str): A Markdown string containing at least one table.

        Returns:
            Table | None: The first ``Table`` extracted from the Markdown, or
            ``None`` if no table could be found.
        """
        fake_doc = f"# X\n\n{md}\n"
        bs = io.BytesIO(fake_doc.encode("utf-8"))
        doc = RichDocument.from_document_file(DocumentStream(name="x.md", stream=bs))
        if len(doc.get_tables()) > 0:
            return doc.get_tables()[0]
        else:
            return None

    def parts(self):
        """Return the constituent parts of this table component.

        The current implementation always returns an empty list because the
        table is rendered entirely through ``format_for_llm``.

        Returns:
            list[Component | CBlock]: Always an empty list.
        """
        return []

    def to_markdown(self) -> str:
        """Export this table as a Markdown string.

        Returns:
            str: The Markdown representation of this table.
        """
        return self._ti.export_to_markdown(self._doc)

    def transpose(self) -> Table | None:
        """Transpose this table and return the result as a new ``Table``.

        Returns:
            Table | None: A new transposed ``Table``, or ``None`` if the
            transposed Markdown cannot be parsed back into a ``Table``.
        """
        t = self._ti.export_to_dataframe().transpose()
        return Table.from_markdown(t.to_markdown())

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Return the table representation for the Formatter.

        Returns:
            TemplateRepresentation | str: A ``TemplateRepresentation`` that
            renders the table as its Markdown string using a ``{{table}}``
            template.
        """
        return TemplateRepresentation(
            args={"table": self.to_markdown()},
            obj=self,
            tools={
                k: MelleaTool.from_callable(c)
                for k, c in self._get_all_members().items()
            },
            fields=[],
            template="{{table}}",
            template_order=None,
        )
