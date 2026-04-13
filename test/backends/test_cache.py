"""Unit tests for SimpleLRUCache — LRU eviction, callbacks, and capacity edge cases."""

from mellea.backends.cache import SimpleLRUCache

# --- basic put / get ---


def test_put_and_get():
    cache = SimpleLRUCache(capacity=3)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_get_missing_key_returns_none():
    cache = SimpleLRUCache(capacity=3)
    assert cache.get("missing") is None


def test_current_size_tracks_entries():
    cache = SimpleLRUCache(capacity=5)
    assert cache.current_size() == 0
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.current_size() == 2


# --- update existing key ---


def test_put_existing_key_updates_value():
    cache = SimpleLRUCache(capacity=2)
    cache.put("a", "old")
    cache.put("a", "new")
    assert cache.get("a") == "new"
    assert cache.current_size() == 1


def test_put_existing_key_does_not_evict():
    evicted: list = []
    cache = SimpleLRUCache(capacity=1, on_evict=evicted.append)
    cache.put("a", 1)
    cache.put("a", 2)
    assert evicted == []


# --- eviction ---


def test_evicts_least_recently_used():
    cache = SimpleLRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_on_evict_callback_receives_evicted_value():
    evicted: list = []
    cache = SimpleLRUCache(capacity=2, on_evict=evicted.append)
    cache.put("a", "val_a")
    cache.put("b", "val_b")
    cache.put("c", "val_c")  # evicts "a"
    assert evicted == ["val_a"]


def test_multiple_evictions_fire_callback_each_time():
    evicted: list = []
    cache = SimpleLRUCache(capacity=1, on_evict=evicted.append)
    cache.put("a", 1)
    cache.put("b", 2)  # evicts 1
    cache.put("c", 3)  # evicts 2
    assert evicted == [1, 2]


# --- LRU ordering: get promotes to most-recent ---


def test_get_promotes_key_preventing_eviction():
    cache = SimpleLRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # promote "a" — now "b" is LRU
    cache.put("c", 3)  # should evict "b", not "a"
    assert cache.get("a") == 1
    assert cache.get("b") is None


# --- capacity edge cases ---


def test_zero_capacity_put_is_noop():
    evicted: list = []
    cache = SimpleLRUCache(capacity=0, on_evict=evicted.append)
    cache.put("a", 1)
    assert cache.current_size() == 0
    assert cache.get("a") is None
    assert evicted == []


def test_capacity_one():
    cache = SimpleLRUCache(capacity=1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.current_size() == 1
