"""
Quota enforcement.

These checks raise HTTPException so they can be called inline from any
endpoint handler — no extra try/except in the caller. Each check is a
single function; the route decides which ones apply.

Quota policy lives here so it's auditable in one place. The numbers come
from `infra.config`, not from hardcoded constants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import HTTPException

from .config import get_config


def check_upload_size(size_bytes: int, filename: str = "") -> None:
    """Reject if a single upload exceeds the configured cap."""
    cap = get_config().max_upload_bytes
    if size_bytes > cap:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload too large: {_fmt(size_bytes)} > {_fmt(cap)} "
                f"({filename or 'unnamed'}). "
                f"Tune DT_MAX_UPLOAD_BYTES if you need a higher cap."
            ),
        )


def check_aggregate_upload_size(sizes: Iterable[int]) -> None:
    """Sum of a multi-file upload (e.g. /upload/poses) must also fit the cap.

    We apply the same per-request cap to the total — a request that uploads
    200 small files should still stay under the limit. Use a separate env
    var if you need an independent cap.
    """
    total = sum(int(s) for s in sizes)
    cap = get_config().max_upload_bytes
    if total > cap:
        raise HTTPException(
            status_code=413,
            detail=f"Upload batch too large: {_fmt(total)} > {_fmt(cap)} total.",
        )


def check_poses_count(current_count: int, adding: int = 0) -> None:
    """Reject if accepting `adding` more poses would exceed the per-session cap."""
    cap = get_config().max_poses_per_session
    if current_count + adding > cap:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Pose count cap exceeded: {current_count + adding} > {cap}. "
                f"Remove some poses or tune DT_MAX_POSES_PER_SESSION."
            ),
        )


def check_session_storage(workspace: Path, additional_bytes: int = 0) -> None:
    """
    Sum disk usage under `workspace` and reject if it would exceed the cap.

    Called before accepting a new upload — we want to fail fast and not
    write a 100 MB FBX only to then notice we're over budget.
    """
    cap = get_config().max_session_storage_bytes
    used = _dir_size(workspace)
    if used + additional_bytes > cap:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Session storage cap exceeded: "
                f"{_fmt(used + additional_bytes)} > {_fmt(cap)}. "
                f"Delete uploads or start a fresh session."
            ),
        )


def check_source_triangles(n_tris: int) -> None:
    """Reject a source reference denser than the configured cap."""
    cap = get_config().max_source_triangles
    if n_tris > cap:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Source mesh too dense: {n_tris:,} triangles > {cap:,} limit. "
                f"Decimate it in Blender (Decimate modifier, or "
                f"Mesh ▸ Cleanup) and re-upload. The source topology is what "
                f"the result inherits, so a lighter source is what you want anyway."
            ),
        )


def check_target_triangles(n_tris: int) -> bool:
    """Validate a target reference's triangle count.

    Returns True if DT transfer (/run) is available for this target, False if
    it's too dense for DT but still fine for Wrap. Raises 400 if it exceeds the
    absolute ceiling (too dense even for Wrap).
    """
    cfg = get_config()
    if n_tris > cfg.max_target_triangles:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target mesh too dense: {n_tris:,} triangles > "
                f"{cfg.max_target_triangles:,} limit. Decimate it in Blender "
                f"and re-upload."
            ),
        )
    return n_tris <= cfg.max_target_triangles_dt


def _dir_size(path: Path) -> int:
    """Recursive byte total under `path`. Returns 0 if missing."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            # Ignore races (file deleted mid-walk); the next call will catch up.
            continue
    return total


def _fmt(n: int) -> str:
    """Human-readable byte count for error messages."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"
