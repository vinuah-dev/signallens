"""Process-local, size-bounded storage for datasets and trained models.

Good enough for a single-instance deployment; state is lost on restart, and
the frontend handles the resulting 404s by asking the user to reload data."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

from .errors import NotFoundError

T = TypeVar("T")


class LRUStore(Generic[T]):
    def __init__(self, capacity: int, kind: str):
        self.capacity = capacity
        self.kind = kind
        self._items: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: str, value: T) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def get(self, key: str) -> T:
        with self._lock:
            if key not in self._items:
                raise NotFoundError(
                    f"This {self.kind} is no longer available (the server may have restarted). Please load your data again.",
                    code=f"{self.kind}_not_found",
                )
            self._items.move_to_end(key)
            return self._items[key]

    def values(self) -> list[T]:
        with self._lock:
            return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)
