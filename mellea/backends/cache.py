"""Cache abstractions and implementations for model state.

Defines the abstract ``Cache`` interface with ``put``, ``get``, and
``current_size`` methods, and provides a concrete ``SimpleLRUCache`` that evicts
the least-recently-used entry when capacity is exceeded — optionally calling an
``on_evict`` callback (e.g. to free GPU memory). Used by local HuggingFace backends
to store and reuse KV cache state across requests.
"""

import abc
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class Cache(abc.ABC):
    """A Cache for storing model state (e.g., kv cache)."""

    # Whenever PEP 695 generics are supported by mypy, we should use them here.

    @abc.abstractmethod
    def put(self, key: str | int, value: Any):
        """Insert a value into the cache under the given key.

        May trigger eviction of existing entries if the cache is at capacity.

        Args:
            key (str | int): The cache key to store the value under.
            value (Any): The value to store.
        """
        ...

    @abc.abstractmethod
    def get(self, key: str | int) -> Any | None:
        """Retrieve a value from the cache by key.

        May affect which entries are considered for future eviction (e.g. LRU ordering).

        Args:
            key (str | int): The cache key to look up.

        Returns:
            Any | None: The cached value, or ``None`` if ``key`` has no cached entry.
        """
        ...

    @abc.abstractmethod
    def current_size(self) -> int:
        """Return the number of entries currently stored in the cache.

        Returns:
            int: Count of items currently held in the cache; useful for debugging.
        """
        ...


class SimpleLRUCache(Cache):
    """A simple `LRU <https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_(LRU)>`_ cache.

    Evicts the least-recently-used entry when capacity is exceeded, optionally
    invoking an ``on_evict`` callback (e.g. to free GPU memory). Used by local
    HuggingFace backends to store and reuse KV cache state across requests.

    Args:
        capacity (int): Maximum number of items to store in the cache.
        on_evict (Callable[[Any], None] | None): Optional callback invoked with the
            evicted value whenever an entry is removed to make room for a new one.

    Attributes:
        capacity (int): Maximum number of items this cache will hold.
        cache (OrderedDict): Internal ordered dict used for LRU tracking.
        on_evict (Callable[[Any], None] | None): Eviction callback, or ``None`` if not set.
    """

    def __init__(self, capacity: int, on_evict: Callable[[Any], None] | None = None):
        """Initializes the LRU cache with a certain capacity.

        The `SimpleLRUCache` either contains a value or it doesn't. There is no cache hierarchy. Take care when choosing `capacity`. In practice usually a small value will be fine, but ideally you should try to choose a capacity based upon your available device memory and the context size of your model.

        Args:
            capacity: Maximum number of items to store in the cache.
            on_evict: Optional callback function called when an item is evicted from the cache.
                      This can be used to free resources (e.g., GPU memory) when items are removed.
        """
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.on_evict = on_evict

    def current_size(self) -> int:
        """Return the number of entries currently stored in the cache.

        Returns:
            int: Count of items currently held in the cache; useful for debugging.
        """
        return len(self.cache.keys())

    def get(self, key: str | int) -> Any | None:
        """Retrieve a value from the cache, promoting it to most-recently-used.

        Args:
            key (str | int): The cache key to look up.

        Returns:
            Any | None: The cached value, or ``None`` if ``key`` is not present.
        """
        if key not in self.cache:
            return None
        else:
            # Move the accessed item to the end (most recent)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

    def put(self, key: str | int, value: Any):
        """Insert or update a value in the cache.

        If the cache is at capacity and the key is new, the least-recently-used
        entry is evicted first, invoking the ``on_evict`` callback if set.

        Args:
            key (str | int): The cache key to store the value under.
            value (Any): The value to cache.
        """
        if self.capacity <= 0:
            return
        if key in self.cache:
            # If the key exists, move it to the end (most recent)
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # If the cache is full, remove the least recently used item
            _evicted_key, evicted_value = self.cache.popitem(last=False)
            # Call eviction callback if provided (e.g., to free GPU memory)
            if self.on_evict is not None:
                self.on_evict(evicted_value)
        # Add the new key-value pair to the end (most recent)
        self.cache[key] = value
