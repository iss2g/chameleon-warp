"""
In-memory job registry for the async /run and /wrap pipeline.

Why: /run and /wrap take tens of seconds to minutes. Held open as a single
synchronous HTTP request they die behind any proxy with an idle timeout,
and if the user closes the tab the client goes blind even though the server
keeps computing. So the endpoints now return a `job_id` immediately, do the
work on a background thread, and the client polls GET /sessions/{id}/job for
`state` + `progress` + `queue_position`.

Design decisions:
  • One active job per SESSION. A session is a single user's workspace; there
    is never a good reason to run two heavy jobs for it at once, and keying on
    session id makes GET /sessions/{id}/job trivial. Submitting while a job is
    still queued/running is a 409.
  • The registry is pure bookkeeping — it does NOT own the compute or the
    threadpool. The endpoint spawns a daemon thread that acquires the shared
    ComputeQueue slot (so cross-session serialization is unchanged) and calls
    back into the job to publish progress.
  • Process-local, like the rest of the app (single uvicorn worker; the math
    is CPU-bound so horizontal scaling isn't wanted). Terminal snapshots are
    persisted into the session's state.json by the caller so a backend restart
    can still answer "your last job finished / errored".

Thread-safety: every field read/written from more than one thread goes
through the per-job lock (worker thread writes progress/state, request threads
read snapshots). The registry has its own lock for the id maps.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional

# Terminal states — a job in one of these will never change again.
TERMINAL = ("done", "error")


class Job:
    """A single background compute (run or wrap) for one session."""

    def __init__(self, job_id: str, session_id: str, kind: str):
        self.job_id = job_id
        self.session_id = session_id
        self.kind = kind  # "run" | "wrap"
        self.state = "queued"  # queued | running | done | error
        # progress: stage is a coarse phase label; iter/total drive a bar
        # during the iterative phases (0/0 = indeterminate).
        self._stage = "queued"
        self._iter = 0
        self._total = 0
        self.error: Optional[str] = None
        # The full endpoint response payload, published on success so GET /job
        # can hand it to the client without re-running anything.
        self.result: Optional[dict] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._lock = threading.Lock()

    # ---- worker-side mutators -------------------------------------------- #

    def set_progress(self, stage: str, iter: int = 0, total: int = 0) -> None:
        with self._lock:
            self._stage = stage
            self._iter = int(iter)
            self._total = int(total)

    def mark_running(self) -> None:
        with self._lock:
            self.state = "running"
            self.started_at = time.time()
            self._stage = "precompute"

    def mark_done(self, result: dict) -> None:
        with self._lock:
            self.state = "done"
            self.result = result
            self.finished_at = time.time()
            self._stage = "done"

    def mark_error(self, message: str) -> None:
        with self._lock:
            self.state = "error"
            self.error = message
            self.finished_at = time.time()
            self._stage = "error"

    # ---- read side ------------------------------------------------------- #

    @property
    def is_active(self) -> bool:
        return self.state not in TERMINAL

    def snapshot(self, queue_position: int = -1) -> dict:
        """Status payload for GET /sessions/{id}/job."""
        with self._lock:
            snap = {
                "job_id": self.job_id,
                "kind": self.kind,
                "state": self.state,
                "queue_position": queue_position,
                "progress": {
                    "stage": self._stage,
                    "iter": self._iter,
                    "total": self._total,
                },
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }
            # Only ship the (potentially large) result payload once done.
            if self.state == "done":
                snap["result"] = self.result
            return snap

    def persist_dict(self) -> dict:
        """Compact terminal snapshot stored in session state.json so a restart
        can still report the outcome of the last job. Excludes the heavy
        result payload — the session's own persisted results/wrap cover that."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "kind": self.kind,
                "state": self.state,
                "error": self.error,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
            }


class JobRegistry:
    """Tracks the latest job per session and their submission order (for
    queue-position math). Process-local."""

    def __init__(self) -> None:
        self._by_session: Dict[str, Job] = {}
        self._order: List[Job] = []  # creation order; pruned lazily
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[Job]:
        with self._lock:
            return self._by_session.get(session_id)

    def create(self, session_id: str, kind: str) -> Job:
        """Register a fresh job for the session. Raises RuntimeError if one is
        still active (queued/running) — the caller maps that to HTTP 409."""
        job = Job(uuid.uuid4().hex[:12], session_id, kind)
        with self._lock:
            existing = self._by_session.get(session_id)
            if existing is not None and existing.is_active:
                raise RuntimeError("a job is already in progress for this session")
            self._by_session[session_id] = job
            self._order.append(job)
            # Keep _order from growing unbounded: drop terminal jobs that are
            # no longer the session's current job.
            self._prune_locked()
        return job

    def forget(self, session_id: str) -> None:
        """Drop a session's job entirely (called on session delete)."""
        with self._lock:
            job = self._by_session.pop(session_id, None)
            if job is not None:
                try:
                    self._order.remove(job)
                except ValueError:
                    pass

    def queue_position(self, job: Job) -> int:
        """0 = currently running (or first in line), 1 = next, etc. -1 once the
        job is terminal. Position among all still-active jobs in submission
        order, which with concurrency=1 is exactly "how many ahead of you"."""
        with self._lock:
            if job.state in TERMINAL:
                return -1
            active = [j for j in self._order if j.is_active]
            try:
                return active.index(job)
            except ValueError:
                return -1

    def _prune_locked(self) -> None:
        current = set(id(j) for j in self._by_session.values())
        self._order = [
            j for j in self._order if j.is_active or id(j) in current
        ]


# Single process-wide registry, mirrors the ComputeQueue lifecycle.
_registry: Optional[JobRegistry] = None


def init_registry() -> JobRegistry:
    global _registry
    _registry = JobRegistry()
    return _registry


def get_registry() -> JobRegistry:
    if _registry is None:
        raise RuntimeError(
            "JobRegistry not initialized. Call infra.jobs.init_registry() from "
            "application bootstrap before any /run or /wrap handler runs."
        )
    return _registry
