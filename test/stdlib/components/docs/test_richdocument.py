import os
import tempfile

import pytest
from docling_core.types.doc.document import DoclingDocument

import mellea
from mellea.core import TemplateRepresentation
from mellea.stdlib.components.docs.richdocument import RichDocument, Table


@pytest.fixture(scope="module")
def rd() -> RichDocument:
    # Use a specific document so we can test some of the functionality
    # related to extracting and transforming text.
    return RichDocument.from_document_file("https://arxiv.org/pdf/1906.04043")


def test_richdocument_basics(rd: RichDocument):
    # Ensure basic parts of a rich document are as expected.
    # assert len(rd.parts()) == 0, "rich documents should have no parts"
    assert isinstance(rd.docling(), DoclingDocument), (
        "rich documents should have docling documents"
    )

    repr = rd.format_for_llm()
    assert isinstance(repr, str), "rich document template args should be a dict"


def test_richdocument_markdown(rd: RichDocument):
    mkd = rd.to_markdown()
    assert isinstance(mkd, str), "rich document `to_markdown` should be a string"
    assert "Bag of Words" in mkd, "expected string not in rd `to_markdown` output"


def test_richdocument_save(rd: RichDocument):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        path = os.path.join(temp_dir, "temp_rd.json")
        rd.save(path)

        loaded_rd = RichDocument.load(path)
        assert loaded_rd.docling().export_to_markdown(
            to_element=10
        ) == rd.docling().export_to_markdown(to_element=10), (
            "saved and loaded rich document text don't match."
        )


def test_table(rd: RichDocument):
    # Getting the tables technically tests the functionality of richdocument,
    # but we do it here to make it easier. The provided document has one table.
    tables = rd.get_tables()
    assert all(isinstance(t, Table) for t in tables), (
        f"rich document `get_tables` returned a non-table value: {tables}"
    )
    assert len(tables) > 0, (
        "rich document `get_tables` returned an empty array for a document known to have one table"
    )

    table = tables[0]
    repr = table.format_for_llm()
    assert isinstance(repr, TemplateRepresentation), (
        "table template args should be a dict"
    )
    assert "table" in repr.args.keys() and len(repr.args.keys()) == 1, (
        "table's should have a single `as_markdown` key"
    )

    mkd_table = table.to_markdown()
    assert isinstance(mkd_table, str), "table `to_markdown` should return a string"

    loaded_table = Table.from_markdown(mkd_table)
    assert loaded_table is not None, (
        "loaded table should not be None when loading from a known source"
    )

    # Use `in` since there might be some slight changes like the
    # title missing.
    assert loaded_table.to_markdown() in mkd_table, (
        "loaded table and original table don't match"
    )

    transposed_table = table.transpose()
    assert transposed_table is not None, (
        "transposed table should not be None for known table"
    )
    mkd_transposed = transposed_table.to_markdown()

    expected_first_row = "|         | 0            | 1                                    | 2                              | 3                                   | 4                             |"
    assert expected_first_row in mkd_transposed, (
        "transposed table is not in the expected form"
    )


def test_empty_table():
    table = Table.from_markdown("")
    assert table is None, "table should be empty when supplied string is empty"


@pytest.mark.skip  # Test requires too much memory for smaller machines.
def test_richdocument_generation(rd: RichDocument):
    m = mellea.start_session(backend_name="hf")
    response = m.chat(rd.to_markdown()[:500] + "\nSummarize the provided document.")
    assert response.content != "", (
        "response content should not be empty when summarizing a rich document"
    )
    assert "paper" in response.content.lower() or "gltr" in response.content.lower(), (
        "response content should include the word paper or gltr when summarizing"
    )
