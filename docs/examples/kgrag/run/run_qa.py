import argparse
from datetime import datetime
from copy import deepcopy
import functools
import json
import os
import time

from mellea import MelleaSession
from dataset.movie_dataset import MovieDatasetLoader
from kg_model import KGModel
from utils.logger import BaseProgressLogger, DefaultProgressLogger, QAProgressLogger
from utils.utils import token_counter, always_get_an_event_loop
from eval import evaluate_predictions
from constants import (
    DATASET_PATH, 
    API_BASE, API_KEY, TIME_OUT, MODEL_NAME,
    EVAL_API_BASE, EVAL_API_KEY, EVAL_TIME_OUT, EVAL_MODEL_NAME,
    EMB_API_BASE, EMB_API_KEY, EMB_TIME_OUT, EMB_MODEL_NAME,
    RITS_API_KEY
)
from mellea.backends.openai import OpenAIBackend, TemplateFormatter
import openai
import torch

def parse_key_value(arg):
    """Parses key=value string into a (key, value) pair, converting value to int/float if needed."""
    if '=' not in arg:
        raise argparse.ArgumentTypeError(
            "Arguments must be in key=value format")
    key, value = arg.split('=', 1)
    try:
        # Try to cast to int or float
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  # Keep as string if it can't be converted
    return key, value


def _snapshot_token_usage():
    return deepcopy(token_counter.get_token_usage()) if token_counter else {}


def _compute_token_usage_delta(start_usage):
    if not token_counter:
        return {}
    end_usage = token_counter.get_token_usage()
    keys = set(start_usage.keys()) | set(end_usage.keys())
    return {key: end_usage.get(key, 0) - start_usage.get(key, 0) for key in keys}


async def generate_prediction(id: str = "",
                              query: str = "",
                              query_time: datetime = None,
                              ans: str = "",
                              logger: BaseProgressLogger = DefaultProgressLogger(),
                              **kwargs):
    start_time = time.perf_counter()
    token_usage_start = _snapshot_token_usage()

    prediction = await participant_model.generate_answer(query=query, query_time=query_time, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    token_usage_delta = _compute_token_usage_delta(token_usage_start)

    logger.add_stat({
        "id": id,
        "query": query,
        "query_time": query_time,
        "ans": ans,
        "prediction": prediction,
        "processing_time": round(elapsed_time, 2),
        "token_usage": token_usage_delta
    })
    print(len(logger.processed_questions))
    logger.update_progress({"last_question_total": round(elapsed_time, 2)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=128,
                        help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=128,
                        help="Queue size of data loading")
    parser.add_argument("--split", type=int, default=0,
                        help="Dataset split index")
    parser.add_argument("--prefix", type=str,
                        help="Prefix added to the result file name")
    parser.add_argument("--postfix", type=str,
                        help="Postfix added to the result file name")
    parser.add_argument("--keep", action='store_true',
                        help="Keep the progress file")
    parser.add_argument('--config', nargs='*', type=parse_key_value,
                        help="Override model config as key=value")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
        "split": args.split,
    }

    session = MelleaSession(backend=
        OpenAIBackend(
        model_id=MODEL_NAME,
        formatter=TemplateFormatter(model_id=MODEL_NAME),
        base_url=API_BASE,
        api_key=API_KEY,
        timeout=TIME_OUT,
        default_headers={'RITS_API_KEY': RITS_API_KEY}
    ))

    eval_session = MelleaSession(backend=
        OpenAIBackend(
        model_id=EVAL_MODEL_NAME,
        formatter=TemplateFormatter(model_id=EVAL_MODEL_NAME),
        base_url=EVAL_API_BASE,
        api_key=EVAL_API_KEY,
        timeout=EVAL_TIME_OUT,
        default_headers={'RITS_API_KEY': RITS_API_KEY} 
    ))

    if EMB_API_BASE:
        emb_session = openai.AsyncOpenAI(
            base_url=EMB_API_BASE,
            api_key=API_KEY,
            timeout=EMB_TIME_OUT,
            default_headers={'RITS_API_KEY': RITS_API_KEY}
        )
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        emb_session = SentenceTransformer(
            EMB_MODEL_NAME, #"sentence-transformers/all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            ),
        )

    progress_path = f"results/{f"_{args.prefix}" if args.prefix else ""}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
    result_path = f"results/{f"_{args.prefix}" if args.prefix else ""}_results{f"_{args.postfix}" if args.postfix else ""}.json"
    logger = QAProgressLogger(progress_path=progress_path)

    print("processed questions: ")
    print(logger.processed_questions)

    loader = MovieDatasetLoader(
        os.path.join(DATASET_PATH, "crag_movie_dev.jsonl"),
        config, "qa", logger,
        processor=functools.partial(generate_prediction, logger=logger)
    )
    domain = "movie"
    participant_model = KGModel(
        session,
        eval_session,
        emb_session,
        domain=domain,
        config=dict(args.config) if args.config else None,
        logger=logger
    )

    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
    )

    inf_token_usage = token_counter.get_token_usage()
    token_counter.reset_token_usage()
    results = [
        {"id": int(stat["id"]), "query": stat["query"], "query_time": stat["query_time"],
         "ans": stat["ans"], "prediction": stat["prediction"], "processing_time": stat["processing_time"],
         "token_usage": stat.get("token_usage", {})}
        for stat in logger.progress_data["stats"]
    ]
    results = sorted(results, key=lambda x: x["id"])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    queries = [item["query"] for item in results]
    ground_truths_list = [[str(item["ans"])] for item in results]
    predictions = [str(item["prediction"]) for item in results]

    stats, history = evaluate_predictions(
        queries, ground_truths_list, predictions, 'llama', batch_size=64
    )
    eval_token_usage = token_counter.get_token_usage()
    stats.update({
        "inf_prompt_tokens": inf_token_usage.get("prompt_tokens"),
        "inf_completion_tokens": inf_token_usage.get("completion_tokens"),
        "inf_total_tokens": inf_token_usage.get("total_tokens"),
        "eval_prompt_tokens": eval_token_usage.get("prompt_tokens"),
        "eval_completion_tokens": eval_token_usage.get("completion_tokens"),
        "eval_total_tokens": eval_token_usage.get("total_tokens")
    })
    for idx in range(len(results)):
        id = results[idx]['id']
        results[idx]['score'] = history[idx]['score']
        results[idx]['explanation'] = history[idx]['explanation']
    results.insert(0, stats)
    # Save to a JSON file
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if not args.keep:
        os.remove(progress_path)

    logger.info(
        f"Done inference in {args.dataset} dataset on {args.model}_model ✅")
    logger.info(
        f"Inference token usage: {inf_token_usage}; Eval token usage: {eval_token_usage}")