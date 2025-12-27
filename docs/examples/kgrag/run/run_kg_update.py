import argparse
import functools
import os

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend, TemplateFormatter
import openai
import torch

from kg.kg_updater import KG_Updater
from dataset.movie_dataset import MovieDatasetLoader
from utils.logger import KGProgressLogger
from utils.utils import always_get_an_event_loop, token_counter
from constants import (
    DATASET_PATH,
    API_BASE, API_KEY, TIME_OUT, MODEL_NAME,
    EVAL_API_BASE, EVAL_API_KEY, EVAL_TIME_OUT, EVAL_MODEL_NAME,
    EMB_API_BASE, EMB_API_KEY, EMB_TIME_OUT, EMB_MODEL_NAME,
    RITS_API_KEY
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=64, help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=64, help="Queue size of data loading")
    parser.add_argument("--progress-path", type=str, default="results/update_movie_kg_progress.json", help="Progress log path")
    
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
    }

    # Create Mellea session for LLM
    session = MelleaSession(backend=
        OpenAIBackend(
        model_id=MODEL_NAME,
        formatter=TemplateFormatter(model_id=MODEL_NAME),
        base_url=API_BASE,
        api_key=API_KEY,
        timeout=TIME_OUT,
        default_headers={'RITS_API_KEY': RITS_API_KEY}
    ))

    # Create embedding session
    if EMB_API_BASE:
        emb_session = openai.AsyncOpenAI(
            base_url=EMB_API_BASE,
            api_key=EMB_API_KEY,
            timeout=EMB_TIME_OUT,
            default_headers={'RITS_API_KEY': RITS_API_KEY}
        )
    else:
        from sentence_transformers import SentenceTransformer
        emb_session = SentenceTransformer(
            EMB_MODEL_NAME,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            ),
        )

    logger = KGProgressLogger(progress_path=args.progress_path)
    updater = KG_Updater(session=session, emb_session=emb_session, config=config, logger=logger)
    domain = "movie"
    loader = MovieDatasetLoader(
        os.path.join(DATASET_PATH, "crag_movie_dev.jsonl.bz2"), 
        config, "doc", logger, 
        processor=functools.partial(updater.process_doc, domain=domain)
    )
    print(logger.processed_docs)

    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
        # parse_dataset(DATA_PATH, logger=logger)
    )

    logger.info(f"Done updating KG using provided corpus ✅")
    logger.info(f"Token usage: {token_counter.get_token_usage()}")