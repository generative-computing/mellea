# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mellea.formatters.granite.retrievers.util download helpers."""

import http.client
import urllib.error
from unittest.mock import patch

import pytest

pytest.importorskip(
    "pyarrow", reason="pyarrow not installed — install mellea[granite_retriever]"
)

from mellea.formatters.granite.retrievers.util import (  # type: ignore[reportAttributeAccessIssue]
    download_mtrag_corpus,
    download_mtrag_embeddings,
)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://example.com",
        code=code,
        msg=str(code),
        hdrs=http.client.HTTPMessage(),
        fp=None,
    )


# ---------------------------------------------------------------------------
# download_mtrag_corpus
# ---------------------------------------------------------------------------


def test_corpus_raises_on_invalid_name(tmp_path):
    with pytest.raises(ValueError, match="cloud"):
        download_mtrag_corpus(str(tmp_path), "invalid_corpus")


def test_corpus_skips_download_if_file_exists(tmp_path):
    target = tmp_path / "cloud.jsonl.zip"
    target.write_bytes(b"dummy")
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        download_mtrag_corpus(str(tmp_path), "cloud")
    mock_retrieve.assert_not_called()


def test_corpus_downloads_when_file_missing(tmp_path):
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        download_mtrag_corpus(str(tmp_path), "fiqa")
    mock_retrieve.assert_called_once()
    assert "fiqa" in mock_retrieve.call_args[0][0]


def test_corpus_propagates_http_error(tmp_path):
    with patch("urllib.request.urlretrieve", side_effect=_http_error(429)):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            download_mtrag_corpus(str(tmp_path), "cloud")
    assert exc_info.value.code == 429


# ---------------------------------------------------------------------------
# download_mtrag_embeddings — 404 breaks cleanly, other errors propagate
# ---------------------------------------------------------------------------


def test_embeddings_404_on_first_part_raises_value_error(tmp_path):
    """A 404 on part_001 means no embeddings exist — ValueError, not HTTPError."""
    with patch("urllib.request.urlretrieve", side_effect=_http_error(404)):
        with pytest.raises(ValueError, match="No precomputed embeddings"):
            download_mtrag_embeddings("model", "cloud", str(tmp_path))


def test_embeddings_stops_after_404(tmp_path):
    """Downloads part_001, then 404 on part_002 → returns cleanly with one part."""
    call_count = 0

    def side_effect(_url, dest):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            open(dest, "wb").close()  # create the file
            return
        raise _http_error(404)

    with patch("urllib.request.urlretrieve", side_effect=side_effect):
        download_mtrag_embeddings("model", "cloud", str(tmp_path))

    assert call_count == 2


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_embeddings_propagates_non_404_errors(tmp_path, code):
    """429/5xx must propagate — never silently treated as end-of-parts."""
    with patch("urllib.request.urlretrieve", side_effect=_http_error(code)):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            download_mtrag_embeddings("model", "cloud", str(tmp_path))
    assert exc_info.value.code == code
