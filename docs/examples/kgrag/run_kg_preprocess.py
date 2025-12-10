import argparse
import asyncio

from kg.kg_preprocessor import *

if __name__ == "__main__":
    preprocessors = [MovieKG_Preprocessor()]

    # Async Route
    async def main():
        for preprocessor in preprocessors:
            await preprocessor.preprocess()
            await preprocessor.close()

    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)  # Set it as the current loop
    loop.run_until_complete(main())
    
    logger.info("Data imported to Neo4j ✅")
