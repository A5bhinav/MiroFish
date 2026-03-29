"""
Simple thread-safe in-memory TTL cache.

Used across data fetchers to avoid redundant external API calls.
Each service creates its own module-level instance so caches remain
independent and don't interfere with each other.

Usage:
    from ..utils.cache import TTLCache
    _cache = TTLCache()

    # In a fetch function:
    cached = _cache.get(key)
    if cached is not None:
        return cached
    result = expensive_fetch()
    _cache.set(key, result, ttl=300)   # 5-minute TTL
    return result
"""

import time
import threading
from typing import Any, Optional


class TTLCache:
    """Thread-safe in-memory dictionary with per-key time-to-live (TTL)."""

    def __init__(self):
        self._store: dict = {}      # key → (value, expires_at)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired; None otherwise."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        """Store value under key; TTL is in seconds."""
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)

    def delete(self, key: str) -> None:
        """Remove a single key (no-op if absent)."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush the entire cache."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        return len(self._store)


def make_key(url: str, params: dict) -> str:
    """Build a deterministic cache key from URL + query params."""
    if not params:
        return url
    sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{url}?{sorted_params}"
