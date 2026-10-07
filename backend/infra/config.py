"""
Central place for runtime configuration.

All env-var lookups live here so deployments tune behavior via a single
documented surface, and the rest of the code never reads `os.environ`
directly.

To add a new knob:
  1. Add a field to `Config`
  2. Read it from env (with a sane default) in `Config.from_env`
  3. Document the env var name in docs/CONFIGURATION.md
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[config] {name}={raw!r} is not an int, using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[config] {name}={raw!r} is not a float, using default {default}")
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    """Comma-separated list. Empty values dropped, whitespace trimmed."""
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or list(default)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw)


@dataclass(frozen=True)
class Config:
    # ---- storage ----
    # Where session workspaces (uploads, results, caches) live.
    workspace_root: Path

    # ---- session lifecycle ----
    # Idle session age before the sweeper deletes its workspace.
    session_ttl_seconds: float
    # How often the sweeper wakes up.
    session_sweep_interval_seconds: float

    # ---- per-request quotas ----
    # Single upload body cap (FBX with many shape keys can reach 50–100 MB).
    max_upload_bytes: int
    # Hard cap on poses per session (one extracted FBX easily ≈ 50; 250
    # leaves headroom but prevents abuse).
    max_poses_per_session: int
    # Hard cap on the total bytes a single session may consume.
    max_session_storage_bytes: int
    # Hard cap on total bytes across ALL sessions. When exceeded, the store
    # evicts least-recently-used sessions early (before their TTL). 0 = off.
    max_total_storage_bytes: int

    # ---- mesh complexity caps (triangles, post-triangulation) ----
    # Source reference: solve cost + result quality both degrade past this.
    # Over the cap -> 400 with a "decimate in Blender" hint.
    max_source_triangles: int
    # Target reference: hard ceiling. Wrap tolerates dense targets (KD-tree of
    # centroids); above this it's rejected outright.
    max_target_triangles: int
    # Target reference: soft ceiling for DT transfer (/run). The DT result has
    # the target's topology, so a dense target makes /run huge. Above this the
    # upload still succeeds (Wrap works) but the session is flagged
    # DT-unavailable and /run refuses.
    max_target_triangles_dt: int

    # ---- compute queue ----
    # How many DT pipeline jobs (/run, /wrap) may execute concurrently.
    # The math is CPU-bound — two parallel jobs share a single CPU and
    # both run roughly twice as slow, so the throughput-optimal value is
    # 1 on most boxes. Bump if you have a beefy multi-socket machine and
    # want to trade per-user latency for global throughput.
    compute_concurrency: int

    # ---- CORS ----
    # Allowed origins for browser CORS. Only relevant when the UI is served
    # from a different origin than the API.
    cors_origins: List[str]

    @classmethod
    def from_env(cls, default_workspace_root: Path) -> "Config":
        return cls(
            workspace_root=_env_path("DT_WORKSPACE_ROOT", default_workspace_root),

            # 7 days — long enough that work left over a weekend is still
            # there. The sweeper still runs periodically so abandoned sessions
            # don't pile up on disk forever. Override via env.
            session_ttl_seconds=_env_float("DT_SESSION_TTL_SECONDS", 7 * 24 * 60 * 60),
            session_sweep_interval_seconds=_env_float("DT_SESSION_SWEEP_INTERVAL_SECONDS", 30 * 60),

            max_upload_bytes=_env_int("DT_MAX_UPLOAD_BYTES", 200 * 1024 * 1024),  # 200 MB
            max_poses_per_session=_env_int("DT_MAX_POSES_PER_SESSION", 250),
            max_session_storage_bytes=_env_int("DT_MAX_SESSION_STORAGE_BYTES", 1024 * 1024 * 1024),  # 1 GB
            # 50 GB across all sessions before LRU eviction kicks in.
            max_total_storage_bytes=_env_int("DT_MAX_TOTAL_STORAGE_BYTES", 50 * 1024 * 1024 * 1024),

            max_source_triangles=_env_int("DT_MAX_SOURCE_TRIANGLES", 100_000),
            max_target_triangles=_env_int("DT_MAX_TARGET_TRIANGLES", 1_500_000),
            max_target_triangles_dt=_env_int("DT_MAX_TARGET_TRIANGLES_DT", 100_000),

            compute_concurrency=_env_int("DT_COMPUTE_CONCURRENCY", 1),

            cors_origins=_env_list(
                "DT_CORS_ORIGINS",
                ["http://localhost:5173", "http://127.0.0.1:5173"],
            ),
        )


# Single global instance, populated by main.py at startup via Config.from_env.
# Modules that need config import this name. Tests can mutate it via the
# helper below to override values.
_config: Optional[Config] = None


def set_config(c: Config) -> None:
    global _config
    _config = c


def get_config() -> Config:
    if _config is None:
        raise RuntimeError(
            "Config not initialized. Call infra.config.set_config(Config.from_env(...)) "
            "from your application bootstrap before importing other infra modules."
        )
    return _config
