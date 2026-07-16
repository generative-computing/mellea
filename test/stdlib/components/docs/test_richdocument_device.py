"""Unit tests for Docling accelerator device selection in `RichDocument`.

These tests mock out Docling's `DocumentConverter` so they exercise only the
device-selection logic in `from_document_file` — no network access or model
downloads. The heavier end-to-end conversion coverage lives in
`test_richdocument.py`.
"""

from unittest import mock

import pytest

pytest.importorskip(
    "docling_core", reason="docling_core not installed — install mellea[docling]"
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice
from docling_core.types.doc.document import DoclingDocument

from mellea.stdlib.components.docs.richdocument import RichDocument

_CONVERTER_PATH = "mellea.stdlib.components.docs.richdocument.DocumentConverter"


def _mock_converter() -> mock.MagicMock:
    """Return a mocked DocumentConverter whose `convert().document` is valid.

    `from_document_file` wraps `convert(...).document` in a `RichDocument`, so
    the mocked conversion must yield a real (empty) `DoclingDocument` to pass
    the constructor's type check.
    """
    converter_cls = mock.MagicMock()
    converter_cls.return_value.convert.return_value.document = DoclingDocument(
        name="test"
    )
    return converter_cls


def _captured_pipeline_options(converter_cls: mock.MagicMock):
    """Return the PdfPipelineOptions passed to the mocked DocumentConverter."""
    _, kwargs = converter_cls.call_args
    pdf_format_option = kwargs["format_options"][InputFormat.PDF]
    return pdf_format_option.pipeline_options


def test_from_document_file_forces_cpu_on_mps():
    """On Apple Silicon (MPS available), Docling is pinned to the CPU device."""
    converter_cls = _mock_converter()
    with (
        mock.patch.object(RichDocument, "_mps_is_available", return_value=True),
        mock.patch(_CONVERTER_PATH, new=converter_cls),
    ):
        RichDocument.from_document_file("dummy.pdf", do_ocr=False)

    pipeline_options = _captured_pipeline_options(converter_cls)
    assert pipeline_options.accelerator_options.device == AcceleratorDevice.CPU, (
        "Expected the CPU accelerator to be forced when MPS is available"
    )


def test_from_document_file_keeps_default_device_without_mps():
    """Without MPS (CUDA or CPU-only), Docling keeps its auto-selected device."""
    converter_cls = _mock_converter()
    with (
        mock.patch.object(RichDocument, "_mps_is_available", return_value=False),
        mock.patch(_CONVERTER_PATH, new=converter_cls),
    ):
        RichDocument.from_document_file("dummy.pdf", do_ocr=False)

    pipeline_options = _captured_pipeline_options(converter_cls)
    # The default AcceleratorOptions device is AUTO; we must not have overridden it.
    assert pipeline_options.accelerator_options.device != AcceleratorDevice.CPU, (
        "Should not force CPU when MPS is unavailable"
    )


def test_mps_is_available_false_without_torch():
    """`_mps_is_available` returns False when PyTorch cannot be imported."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=_fake_import):
        assert RichDocument._mps_is_available() is False
