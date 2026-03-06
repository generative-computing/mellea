# SPDX-License-Identifier: Apache-2.0

"""Classes and functions that implement the InMemoryRetriever."""

# Standard
import os
import pathlib
import shutil

# Third Party
import numpy as np

# First Party
from ..base.optional import nltk_check


def _sliding_windows(
    token_offsets: list[tuple[int, int]],
    sentence_offsets: list[tuple[int, int]],
    window_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Compute window boundaries for slicing a document into chunks.

    :param token_offsets: Character offsets of the tokens, as begin, end pairs
    :param sentence_offsets: Character offsets of sentence boundaries, as begin, end
    :param window_size: Target maximum size of the windows, in tokens
    :param overlap: How many tokens each window should overlap with next
     and previous

    :returns: Tuples of begin and end offsets of windows
    """
    result: list[tuple[int, int]] = []
    cur_start = 0
    doc_len = len(token_offsets)

    # Create some lookup tables
    sentence_begins = {t[0] for t in sentence_offsets}
    sentence_ends = {t[1] for t in sentence_offsets}

    while len(result) == 0 or cur_start < doc_len - overlap:
        # Compute boundaries between the three regions of the window
        cur_end = min(cur_start + window_size, doc_len)
        begin_overlap_end = min(cur_start + overlap, doc_len)
        end_overlap_begin = min(cur_start + window_size - overlap, cur_end)

        # Advance start of window until we cross a sentence boundary, but don't go past
        # the overlap area.
        cur_start_char = token_offsets[cur_start][0]
        max_start_char = token_offsets[begin_overlap_end - 1][0]
        adjusted_start_char = cur_start_char
        while (
            adjusted_start_char not in sentence_begins
            and adjusted_start_char < max_start_char
        ):
            adjusted_start_char += 1
        if adjusted_start_char == max_start_char:
            adjusted_start_char = cur_start_char
        adjusted_start = cur_start
        while token_offsets[adjusted_start][0] < adjusted_start_char:
            adjusted_start += 1

        # print(f"Adjusted start {cur_start} => {adjusted_start}")

        # Move end of window back until we cross a sentence boundary, but don't go past
        # the overlap area
        cur_end_char = token_offsets[cur_end - 1][1]
        min_end_char = token_offsets[end_overlap_begin - 1][1]
        adjusted_end_char = cur_end_char
        while (
            adjusted_end_char not in sentence_ends and adjusted_end_char > min_end_char
        ):
            adjusted_end_char -= 1
        if adjusted_end_char == min_end_char:
            adjusted_end_char = cur_end_char
        adjusted_end = cur_end
        while token_offsets[adjusted_end - 1][1] > adjusted_end_char:
            adjusted_end -= 1

        # print(f"Adjusted end {cur_end} => {adjusted_end}")

        result.append((adjusted_start, adjusted_end))
        cur_start = adjusted_end - overlap
    return result


def compute_embeddings(
    corpus,  #: pa.Table,
    embedding_model_name: str,
    chunk_size: int = 512,
    overlap: int = 128,
):  # "-> pa.Table:"
    # pylint: disable=too-many-locals
    """Split documents into windows and compute embeddings for each of the the windows.

    :param corpus: PyArrow Table of documents as returned by :func:`read_corpus()`.
        Should have the columns ``["id", "url", "title", "text"]``
    :param embedding_model_name: Hugging Face model name for the model that computes
        embeddings. Also used for tokenizing.
    :param chunk_size: Maximum size of chunks to split documents into, in embedding
        model tokens; must be less than or equal to the embedding model's maximum
        sequence length.
    :param overlap: Target overlap between adjacent chunks, in embedding model tokens.
        Actual begins and ends of chunks will be on sentence boundaries.

    :returns: PyArrow Table of chunks of the corpus, with schema
        ````["id", "url", "title", "begin", "end", "text", "embedding"]``
    """
    # Third Party
    import pyarrow as pa
    import sentence_transformers  # type: ignore[import-not-found]

    with nltk_check("generation of document embeddings"):
        # Third Party
        import nltk

    embedding_model = sentence_transformers.SentenceTransformer(embedding_model_name)
    sentence_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer()

    # Corpora currently fit in memory, so just iterate with a for loop
    id_list = []
    url_list = []
    title_list = []
    text_list = []
    begin_list = []
    end_list = []

    for row in corpus.to_pylist():
        doc_text = row["text"]
        tokenizer_output = embedding_model.tokenizer(
            doc_text, return_offsets_mapping=True, return_attention_mask=False
        )

        # doc_tokens = tokenizer_output["input_ids"]
        token_offsets = tokenizer_output["offset_mapping"]
        sentence_offsets = list(sentence_splitter.span_tokenize(doc_text))

        # For some reason, the tokenizer likes to give the last token a character
        # range from zero to zero.
        if token_offsets[-1] == (0, 0) and len(token_offsets) > 1:
            last_char = token_offsets[-2][1]
            token_offsets[-1] = (last_char, last_char)

        # Pick sliding windows that start and end on sentence boundaries
        offsets = _sliding_windows(token_offsets, sentence_offsets, chunk_size, overlap)

        for begin_tok, end_tok in offsets:
            # Use the token offsets to find the appropriate ranges of characters
            # for the windows. Note that we cannot do this by detokenizing, as the
            # tokenizer may be case-insensitive.
            begin_char = token_offsets[begin_tok][0]
            end_char = token_offsets[end_tok - 1][1]

            id_list.append(row["id"])
            url_list.append(row["url"])
            title_list.append(row["title"])
            text_list.append(doc_text[begin_char:end_char])
            begin_list.append(begin_char)
            end_list.append(end_char)
    result = pa.table(
        {
            "id": pa.array(id_list),
            "url": pa.array(url_list),
            "title": pa.array(title_list),
            "begin": pa.array(begin_list),
            "end": pa.array(end_list),
            "text": pa.array(text_list),
        }
    )
    embeddings = embedding_model.encode(result.column("text").to_pylist())
    embeddings_list = list(embeddings)  # Retain dtype
    result = result.append_column(
        "embedding",
        pa.array(embeddings_list, type=pa.list_(pa.from_numpy_dtype(embeddings.dtype))),
    )
    return result


def write_embeddings(
    target_dir: str,
    corpus_name: str,
    embeddings,  #: pa.Table,
    chunks_per_partition: int = 10000,
) -> pathlib.Path:
    """Write embeddings.

    Write the embeddings produced by :func:`compute_embeddings()` to a directory of
    Parquet files on local disk.

    :param target_dir: Location where the files should be written (in a subdirectory)
    :param corpus_name: Corpus name to use in generating a location for writing files
    :param chunks_per_partition: Number of document chunks to write to each partition
      of the Parquet file

    :param returns: Path where data was written
    """
    # Third Party
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_root = pathlib.Path(target_dir) / f"{corpus_name}.parquet"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)
    # Divide into bite-sized chunks
    for batch_num, batch in enumerate(embeddings.to_batches(chunks_per_partition)):
        pq.write_table(
            pa.Table.from_batches([batch]),
            output_root / f"part_{batch_num + 1:03d}.parquet",
        )
    return output_root


class InMemoryRetriever:
    """Simple retriever that keeps docs and embeddings in memory."""

    def __init__(
        self,
        data_file_or_table,  #: pathlib.Path | str | pa.Table,
        embedding_model_name: str,
    ):
        """Simple retriever that keeps docs and embeddings in memory.

        :param data_file_or_table: Parquet file of document snippets and embeddings,
         or an equivalent in-memory PyArrow Table object.
         Should have columns `id`, `begin`, `end`, `text`, and `embedding`.
        :param embedding_model_name: Name of Sentence Transformers model to use for
         embeddings. Must be the same model that was used to compute embeddings in the
         data file.
        """
        # Third Party
        import pyarrow as pa
        import pyarrow.parquet as pq
        import sentence_transformers
        import torch

        if isinstance(data_file_or_table, pa.Table):
            self._data_table = data_file_or_table
        else:
            self._data_table = pq.read_table(data_file_or_table)
        self._is_float16 = pa.types.is_float16(
            self._data_table.schema.field("embedding").type.value_type
        )
        self._embedding_model = sentence_transformers.SentenceTransformer(
            embedding_model_name,
            model_kwargs={"dtype": "float16" if self._is_float16 else "float32"},
        )
        embeddings_array = np.array(
            list(self._data_table.column("embedding").to_numpy())
        )
        self._embeddings = torch.tensor(embeddings_array)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Run a query and return results."""
        # Third Party
        import pyarrow as pa
        import sentence_transformers

        query_embeddings = self._embedding_model.encode(query)
        raw_result = sentence_transformers.util.semantic_search(
            query_embeddings, self._embeddings, top_k=top_k
        )

        # raw_result is a list of lists of {corpus_id, score}
        row_nums = [r["corpus_id"] for r in raw_result[0]]
        scores = [r["score"] for r in raw_result[0]]
        result = self._data_table.take(row_nums)
        result = result.append_column("score", pa.array(scores))
        rows = result.select(["id", "title", "url", "begin", "end", "text", "score"])

        # TODO: Wrap this in the Retriever Processor
        documents = []
        for row in rows.to_pylist():
            document = {
                "doc_id": str(row["id"]),  # Non-string values crash vLLM
                "text": row["text"],
                "score": str(row["score"]),  # Non-string values crash vLLM
            }
            documents.append(document)

        return documents
