# SPDX-License-Identifier: Apache-2.0

"""Various utility functions relating to the MTRAG benchmark data set."""

# Standard
import os
import pathlib
import urllib.error
import urllib.request
import zipfile

# Third Party
import pyarrow as pa
import pyarrow.json as pj


def download_mtrag_corpus(target_dir: str, corpus_name: str) -> pathlib.Path:
    """Download a corpus file from the MTRAG benchmark if the file hasn't already present.

    Args:
        target_dir: Location where the file should be written if not already present.
        corpus_name: Should be one of ``"cloud"``, ``"clapnq"``, ``"fiqa"``,
            or ``"govt"``.

    Returns:
        Path to the downloaded (or cached) file.
    """
    corpus_names = ("cloud", "clapnq", "fiqa", "govt")
    if corpus_name not in corpus_names:
        raise ValueError(f"Corpus name must be one of {corpus_names}")

    target_file = pathlib.Path(target_dir) / f"{corpus_name}.jsonl.zip"
    if not os.path.exists(target_file):
        source_url = (
            f"https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/"
            f"corpora/{corpus_name}.jsonl.zip"
        )
        urllib.request.urlretrieve(source_url, target_file)
    return target_file


MB = 1048576  # 1MB in bytes


def read_mtrag_corpus(corpus_file: str | pathlib.Path) -> pa.Table:
    """Read the documents from one of the MTRAG benchmark's corpora.

    Args:
        corpus_file: Location of the corpus data file.

    Returns:
        Documents from the corpus as a PyArrow table, with schema
        ``["id", "url", "title", "text"]``.
    """
    if not isinstance(corpus_file, pathlib.Path):
        corpus_file = pathlib.Path(corpus_file)
    tables = []
    with zipfile.ZipFile(corpus_file, "r") as z:
        for name in z.namelist():
            tables.append(
                # pylint: disable=consider-using-with
                pj.read_json(z.open(name), pj.ReadOptions(block_size=100 * MB))
            )
    t = pa.concat_tables(tables)

    # Normalize schema
    # Guess at ID column
    id_candidates = [n for n in t.schema.names if "id" in n]
    if len(id_candidates) != 1:
        raise TypeError(f"Couldn't guess ID column; candidates are {id_candidates}")
    t = t.rename_columns(["id" if n == id_candidates[0] else n for n in t.schema.names])
    if "text" not in t.schema.names:
        raise TypeError(f"No text column (columns are {t.schema.names})")

    if "title" not in t.schema.names and "url" in t.schema.names:
        # Use URL as title if there is no title
        t = t.append_column("title", t.column("url"))

    # Fill in missing columns
    for name in ("title", "url"):
        if name not in t.schema.names:
            t = t.append_column(name, pa.array([None] * t.num_rows, type=pa.string()))

    # Remove extra columns
    t = t.select(["id", "url", "title", "text"])
    return t


def download_mtrag_embeddings(embedding_name: str, corpus_name: str, target_dir: str):
    """Download precomputed embeddings for a corpus in the MTRAG benchmark.

    Args:
        embedding_name: Name of the SentenceTransformers embedding model used to
            create the embeddings.
        corpus_name: Should be one of ``"cloud"``, ``"clapnq"``, ``"fiqa"``,
            or ``"govt"``.
        target_dir: Location where Parquet files named ``"part_001.parquet"``,
            ``"part_002.parquet"``, etc. will be written.
    """
    corpus_names = ("cloud", "clapnq", "fiqa", "govt")
    if corpus_name not in corpus_names:
        raise ValueError(f"Corpus name must be one of {corpus_names}")

    target_root = pathlib.Path(target_dir) / f"{corpus_name}_{embedding_name}.parquet"
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    part_num = 1
    repo_root = "https://github.com/frreiss/mt-rag-embeddings"
    while True:
        # Download part_001.parquet, part_002.parquet, etc. until a download fails.
        parquet_file_name = f"part_{part_num:03d}.parquet"
        source_url = (
            f"{repo_root}/raw/refs/heads/main/{embedding_name}/{corpus_name}.parquet"
            f"/{parquet_file_name}"
        )
        target_file = target_root / parquet_file_name
        try:
            urllib.request.urlretrieve(source_url, target_file)
            part_num += 1
        except urllib.error.HTTPError:
            # Found all the parts; flow through
            break

    if part_num == 1:
        raise ValueError(
            f"No precomputed embeddings for MTRAG corpus '{corpus_name}' "
            f"with embedding model '{embedding_name}' found at {repo_root}"
        )
