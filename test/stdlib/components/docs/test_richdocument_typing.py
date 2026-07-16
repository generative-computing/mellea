# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip(
    "docling_core", reason="docling_core not installed — install mellea[docling]"
)

from mellea.stdlib.components.docs.richdocument import RichDocument


@pytest.mark.parametrize(
    "bad_value",
    ["path/to/file.pdf", "/absolute/path.pdf", 42, None],
    ids=["str-path", "absolute-str-path", "int", "none"],
)
def test_richdocument_init_rejects_non_docling_document(bad_value):
    with pytest.raises(TypeError, match="DoclingDocument"):
        RichDocument(bad_value)


def test_richdocument_init_rejects_path_object():
    from pathlib import Path

    with pytest.raises(TypeError, match="from_document_file"):
        RichDocument(Path("some/file.pdf"))
