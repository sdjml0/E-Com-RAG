import time
import hashlib
import json
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("cache_service")

class LRUContextCache:
    """In-memory query context & vector embedding cache with TTL."""

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}

    def _hash_key(self, key_data: Any) -> str:
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key_data: Any) -> Any | None:
        key = self._hash_key(key_data)
        if key in self._cache:
            expires_at, value = self._cache[key]
            if time.time() < expires_at:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key_data: Any, value: Any, ttl_seconds: int | None = None) -> None:
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl
        key = self._hash_key(key_data)
        self._cache[key] = (expires_at, value)

    def clear(self) -> None:
        self._cache.clear()

cache_service = LRUContextCache()
