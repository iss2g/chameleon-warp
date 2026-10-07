"""
Per-session upload serialization.

Fixes the race condition where two concurrent uploads (e.g. user kicks off a
slow FBX → Blender extract for source and clicks target before the first
finishes) both mutate `Session.poses` / `Session.source_ref_path` /
`Session.target_ref_path`, corrupting state.json and leaving the session
in a half-applied configuration that crashes subsequent requests.

The fix is conceptually simple: every endpoint that mutates session state
under upload must take a per-session lock. The Session dataclass already
has a `threading.Lock` for the /run pipeline; we expose a small context
manager so endpoints opt into serialization with a single `with` line.

Why not a single global lock? Uploads to different sessions (e.g. two
browser tabs) are independent — only same-session concurrent uploads are
the bug.

Why a non-blocking try-acquire? We don't want to hold an HTTP request for
30 seconds waiting on someone else's Blender extract. If a session is
already busy uploading, we return 409 (Conflict) immediately and ask the
client to retry.
"""
from __future__ import annotations

from contextlib import contextmanager

from fastapi import HTTPException


@contextmanager
def upload_lock(session, *, timeout: float = 0.0):
    """
    Acquire `session._lock` for the duration of an upload mutation.

    Raises HTTPException(409) immediately if another upload on the same
    session is in progress. With `timeout > 0` we'll wait up to that many
    seconds (useful for sequential client retries).
    """
    acquired = session._lock.acquire(timeout=timeout) if timeout > 0 else session._lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail=(
                "Another upload is in progress for this session. "
                "Wait for it to complete before starting the next one."
            ),
        )
    try:
        yield
    finally:
        session._lock.release()
