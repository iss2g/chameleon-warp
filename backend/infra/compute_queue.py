"""
Global queue for CPU-heavy pipeline jobs (/run, /wrap).

Rationale: the deformation transfer math (correspondence iterations, sparse
solves) is single-threaded heavy CPU. Two concurrent jobs share one core
and both run roughly twice as slow as serial — net throughput is the same
but each user waits longer. So we let exactly one job execute at a time
and queue the rest. Other endpoints (markers, mesh fetches, uploads) are
NOT in this queue and continue running in parallel.

The queue is a counted semaphore + two atomic counters for "waiting" and
"running". When a request enters `occupy()`:
  1. Bump `waiting` (it's now visible in /api/queue as "queued").
  2. Block on the semaphore until a slot opens.
  3. Move from waiting → running.
  4. Run the job; release on exit.

Front-end polls `GET /api/queue` while their /run request is in flight,
so it can show position-in-queue feedback without changing the request/
response shape of /run itself.

This is process-local state. Horizontal scaling (multiple uvicorn workers)
would lose serialization — the math is CPU-bound so you wouldn't want
that anyway, but if you do, move queue state to Redis.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator, Dict


class _Counter:
    """Thread-safe integer counter."""

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()

    def bump(self, delta: int) -> int:
        with self._lock:
            self._n += delta
            return self._n

    def value(self) -> int:
        with self._lock:
            return self._n


class ComputeQueue:
    """
    Wraps a counted semaphore with queue-position bookkeeping.

    `concurrency` is how many jobs may run simultaneously. Default 1 means
    strict serialization; bump to e.g. 2 on a beefy multi-socket box.
    """

    def __init__(self, concurrency: int = 1):
        self._sem = threading.BoundedSemaphore(value=max(1, concurrency))
        self._waiting = _Counter()
        self._running = _Counter()
        self._concurrency = max(1, concurrency)

    @contextmanager
    def occupy(self) -> Iterator[None]:
        """
        Block until a slot is free, then yield. Exit releases the slot.

        Usage:
            with compute_queue.occupy():
                # CPU-heavy job

        Safe to call from a synchronous FastAPI handler: FastAPI runs sync
        handlers in a threadpool, so this `blocking` acquire doesn't block
        the asyncio loop — only the worker thread serving this one request.
        """
        self._waiting.bump(+1)
        try:
            self._sem.acquire()
        finally:
            self._waiting.bump(-1)
        self._running.bump(+1)
        try:
            yield
        finally:
            self._running.bump(-1)
            self._sem.release()

    def status(self) -> Dict[str, int]:
        """Snapshot of queue depth for /api/queue."""
        return {
            "waiting": self._waiting.value(),
            "running": self._running.value(),
            "concurrency": self._concurrency,
        }


# Single shared queue for the process. Initialized lazily by main.py once
# config has been loaded — we don't want to pin a concurrency value at
# import time.
_queue: ComputeQueue | None = None


def init_queue(concurrency: int) -> ComputeQueue:
    global _queue
    _queue = ComputeQueue(concurrency=concurrency)
    return _queue


def get_queue() -> ComputeQueue:
    if _queue is None:
        raise RuntimeError(
            "ComputeQueue not initialized. Call init_queue(concurrency) from your "
            "application bootstrap before any /run or /wrap handler runs."
        )
    return _queue
