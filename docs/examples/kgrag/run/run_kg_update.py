import argparse

from kg.kg_updater import KG_Updater 
from dataset.movie_dataset import MovieDatasetLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=64, help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=64, help="Queue size of data loading")
    parser.add_argument("--progress-path", type=str, default="results/update_{dataset}_kg_progress.json", help="Progress log path")
    
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
    }
    logger = KGProgressLogger(progress_path=args.progress_path.format(dataset=args.dataset))
    updater = KG_Updater(config=config, logger=logger)
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