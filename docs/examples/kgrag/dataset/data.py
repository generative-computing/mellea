import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any, Dict

# Base loader with shared interface
class BaseDatasetLoader(ABC):
    """
    Dataset dependent loader
    """
    def __init__(self, config: Dict[str, Any], 
                 mode: str, 
                 processor: Any):
        self.config = config
        self.mode = mode
        self.queue = asyncio.Queue(maxsize=config.get("queue_size", 64))
        self.processor = processor

    @abstractmethod
    async def load_doc(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Load a documents from the dataset.
           Return None when there are no more documents."""
        pass

    @abstractmethod
    async def load_query(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Load a query from the dataset.
           Return None when there are no more queries."""
        pass

    async def producer(self):
        """Continuously load data and put each item into the queue."""
        load = self.load_doc if self.mode.lower() == 'doc' else self.load_query

        async for item in load():
            await self.queue.put(item)
            
        # Signal termination for all consumers
        for _ in range(self.config.get("num_workers", 4)):
            await self.queue.put(None)

    async def consumer(self):
        """Consume items from the queue and process them."""
        task_name = asyncio.current_task().get_name()
        print(task_name)
        while True:
            item = await self.queue.get()
            if item is None:
                print("Stop!")
                break
            await self.processor(**item)
            print(task_name)

    async def run(self):
        """Run the producer-consumer pipeline."""
        producer_task = asyncio.create_task(self.producer(), name="Producer")
        consumer_tasks = [
            asyncio.create_task(self.consumer(), name=f"Consumer-{i}")
            for i in range(self.config.get("num_workers", 4))
        ]
        await asyncio.gather(producer_task, *consumer_tasks)
