from bs4 import BeautifulSoup
import bz2
import json
import trafilatura
from typing import AsyncGenerator, Any, Dict

from utils.data import BaseDatasetLoader
from utils.logger import logger, DefaultProgressLogger, BaseProgressLogger
from utils.utils import parse_timestamp

class MovieDatasetLoader(BaseDatasetLoader):
    
    def __init__(self,
                 data_path: str, 
                 config: Dict[str, Any], 
                 mode: str = "doc",
                 logger: BaseProgressLogger = DefaultProgressLogger(),
                 **kwargs):
        super().__init__(config, mode, **kwargs)

        self.data_path = data_path
        self.logger = logger
        self.data_generator = load_data_in_batches(
            data_path,
            batch_size=1,
            domain="movie"
        )
    
    async def load_doc(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            try:
                batch = next(self.data_generator)
            except StopIteration:
                break  # Exit the loop when there is no more data.

            # Transform each record into a document item with necessary fields
            for group_id, search_results, query_time in zip(
                batch['id'],
                batch["search_results"],
                batch["query_time"],
            ):
                for page_id, page in enumerate(search_results):
                    doc = trafilatura.extract(page["page_result"], include_formatting=True)
                    doc_id = f"{group_id}_{page_id}"
                    if doc_id in self.logger.processed_docs:
                        continue
                    
                    modified_at = parse_timestamp(page["page_last_modified"])
                    created_at = parse_timestamp(query_time)
                    ref = json.dumps({doc_id: {"name": page['page_name'], "link": page["page_url"]}})
                    yield {
                        "id": doc_id,
                        "doc": doc,
                        "created_at": created_at,
                        "modified_at": modified_at,
                        "ref": ref
                    }
    
    async def load_query(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            try:
                batch = next(self.data_generator)
            except StopIteration:
                break  # Exit the loop when there is no more data.

            for idx in range(len(batch['id'])):
                group_id = batch['id'][idx]
                interaction_id = batch["interaction_id"][idx]
                query = batch["query"][idx]
                query_time = batch["query_time"][idx]
                ans = batch["answer"][idx]

                docs = []
                for page in batch["search_results"][idx]:
                    html_source = page["page_result"]
                    soup = BeautifulSoup(html_source, "lxml")
                    text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces
                    docs.append(text)
                
                query_id = f"{group_id}"
                if query_id in self.logger.processed_questions:
                    continue

                query_time = parse_timestamp(query_time)
                yield {
                    "id": query_id,
                    "interaction_id": interaction_id,
                    "docs": docs,
                    "query": query,
                    "query_time": query_time,
                    "ans": ans
                }

def load_data_in_batches(dataset_path, batch_size, domain=None, start_idx=None):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"id": [], "interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        cur = -1
        # Handle both compressed (.bz2) and uncompressed files
        if dataset_path.endswith('.bz2'):
            file = bz2.open(dataset_path, "rt", encoding='utf-8')
        else:
            file = open(dataset_path, "rt", encoding='utf-8')

        with file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    if domain and item['domain'] != domain:
                        continue
                    cur += 1
                    if start_idx and cur < start_idx:
                        continue
                    # if cur == 8:
                    #     return
                    item['id'] = cur
                    for key in batch:
                        batch[key].append(item[key])
                    
                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e