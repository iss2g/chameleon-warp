"""
Session store with disk-backed persistence. Each session owns a workspace
directory on disk that holds the uploaded reference meshes, poses, any
computed results, and a `state.json` describing the session for rehydration.

The session also holds an opportunistic multi-level cache of expensive
intermediates so the user can iterate (add poses, tweak smoothness, etc.)
without paying the full setup cost every time:

  source_setup  ← keyed on source_ref content        (saves ~setup time when target/markers change)
  target_setup  ← keyed on target_ref content
  mapping       ← keyed on (source, target, markers, iters, smoothness, ident_w)
  transformation← keyed on (mapping_key, smoothness)

The numpy/scipy cache slots are NOT serialized — they rebuild lazily on the
next Run. Everything else (file paths, markers, pose list, result paths,
fbx-source flags) is persisted to `state.json` after each mutation so a
backend restart preserves the user's work.
"""
from __future__ import annotations

import json
import secrets
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_FILENAME = "state.json"
STATE_VERSION = 1


def _dir_size_bytes(path: Path) -> int:
    """Recursive byte total under `path`; 0 if missing. Races (a file deleted
    mid-walk) are ignored — the next pass catches up."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total

# How long an idle session lives before the background sweeper deletes its
# workspace. Touched on every API access; heartbeat from the browser keeps
# active sessions warm. 30 min is long enough for short breaks / network
# blips but short enough that abandoned sessions don't pile up on disk.
DEFAULT_SESSION_TTL_SECONDS = 30 * 60
# How often the sweeper runs. Short interval = quicker reclamation, but it
# only walks the in-memory dict so the cost is trivial.
DEFAULT_SWEEP_INTERVAL_SECONDS = 5 * 60


@dataclass
class Pose:
    pose_id: str
    name: str
    path: str


@dataclass
class Session:
    session_id: str
    workspace: Path
    source_ref_path: Optional[str] = None
    target_ref_path: Optional[str] = None
    # Proxy basemesh rest pose (P) for the proxy-wardrobe path (refit D): the
    # topology a wrap deformed into the body, used to transport a bound garment.
    proxy_ref_path: Optional[str] = None
    source_uploaded_as_fbx: bool = False
    target_uploaded_as_fbx: bool = False
    # False when the uploaded target is too dense for DT transfer (/run) but
    # still fine for Wrap. Set at target upload from the triangle-count check;
    # /run refuses when False. True when no target, or target is DT-sized.
    target_dt_available: bool = True
    poses: Dict[str, Pose] = field(default_factory=dict)
    markers: List[Tuple[int, int]] = field(default_factory=list)
    results: Dict[str, str] = field(default_factory=dict)  # pose_id -> result_path
    # Wrap mode artifact: warped source mesh (source topology, target shape).
    # Produced by /wrap; downloaded by user as a customization shape key.
    wrap_path: Optional[str] = None
    # Which flavor produced wrap_path: "wrap" (full shrink-wrap), "fit"
    # (anchor-only accessory fit), or "region" (masked boundary-loop wrap).
    # Affects the download filename.
    wrap_mode: str = "wrap"
    # For "region" wraps: the boundary spec the user drew, so a reload can
    # restore the outline. {"waypoints": [...], "seed": int, "feather_width": float}
    wrap_region: Optional[dict] = None

    # Refit working state (gizmo placement, pins, frozen paint, preset, advanced
    # knobs) so a reload / view switch doesn't lose the user's setup. Opaque blob
    # written by PUT /refit_state; shape owned by the frontend. Includes the
    # source name it was captured against so a changed source discards it.
    refit_state: Optional[dict] = None

    # Unix timestamp of the last API touch. Updated by Session.touch() on
    # every request that hits this session. The store sweeper deletes
    # sessions whose last_seen is older than the configured TTL.
    last_seen: float = field(default_factory=time.time)

    # Compact snapshot of the last finished (done/error) background job, so a
    # backend restart can still answer GET /job after the in-memory registry
    # is gone. Populated by the job worker; see infra.jobs.Job.persist_dict.
    last_job: Optional[dict] = None

    # TTL applied to this session, stamped by the owning SessionStore. NOT
    # persisted (it's a store-level policy, not session data) — used only to
    # surface an expiry time in the summary. Defaults to the module constant so
    # a bare Session is still self-describing.
    ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS

    # ---- cache slots (single-entry caches, key + value) — NOT persisted ----
    source_setup_key: Optional[str] = None
    source_setup: Any = None         # dt_core.correspondence.SourceSetup
    target_setup_key: Optional[str] = None
    target_setup: Any = None         # dt_core.correspondence.TargetSetup
    mapping_key: Optional[str] = None
    mapping: Any = None              # np.ndarray (Nx2)
    transformation_key: Optional[str] = None
    transformation: Any = None       # dt_core.transformation.Transformation

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def invalidate_source_cache(self):
        """Source mesh changed -> everything downstream is stale."""
        self.source_setup_key = None
        self.source_setup = None
        self.mapping_key = None
        self.mapping = None
        self.transformation_key = None
        self.transformation = None
        self._drop_wrap()

    def invalidate_target_cache(self):
        """Target mesh changed -> downstream is stale."""
        self.target_setup_key = None
        self.target_setup = None
        self.mapping_key = None
        self.mapping = None
        self.transformation_key = None
        self.transformation = None
        self._drop_wrap()

    def _drop_wrap(self) -> None:
        """Delete the wrap OBJ on disk and clear the path."""
        if self.wrap_path:
            try:
                Path(self.wrap_path).unlink(missing_ok=True)
            except OSError:
                pass
            self.wrap_path = None
        self.wrap_mode = "wrap"
        self.wrap_region = None

    def touch(self) -> None:
        """Mark the session as recently used (keeps TTL sweeper at bay)."""
        self.last_seen = time.time()

    def delete_workspace(self) -> None:
        """Remove the entire workspace directory from disk."""
        try:
            shutil.rmtree(self.workspace, ignore_errors=True)
        except OSError as e:
            print(f"[session {self.session_id}] delete_workspace failed: {e}")

    def to_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "source_ref": self._summarize(self.source_ref_path),
            "target_ref": self._summarize(self.target_ref_path),
            "proxy_ref": self._summarize(self.proxy_ref_path),
            "source_uploaded_as_fbx": self.source_uploaded_as_fbx,
            "target_uploaded_as_fbx": self.target_uploaded_as_fbx,
            "target_dt_available": self.target_dt_available,
            "poses": [
                {"pose_id": p.pose_id, "name": p.name} for p in self.poses.values()
            ],
            "marker_count": len(self.markers),
            "result_pose_ids": list(self.results.keys()),
            "has_results": bool(self.results),
            "has_wrap": bool(self.wrap_path and Path(self.wrap_path).exists()),
            "wrap_mode": self.wrap_mode,
            "refit_state": self.refit_state,
            # Session lifetime, so the UI can show "expires in …". expires_at is
            # last_seen + ttl in Unix seconds; the browser compares to now.
            "last_seen": self.last_seen,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.last_seen + self.ttl_seconds,
            "cache": {
                "source_setup": self.source_setup_key is not None,
                "target_setup": self.target_setup_key is not None,
                "mapping": self.mapping_key is not None,
                "transformation": self.transformation_key is not None,
            },
        }

    @staticmethod
    def _summarize(p: Optional[str]) -> Optional[dict]:
        if not p:
            return None
        path = Path(p)
        return {"name": path.name}

    # ------------------------------------------------------------------ #
    # Disk persistence — survives backend restart                         #
    # ------------------------------------------------------------------ #

    def _state_path(self) -> Path:
        return self.workspace / STATE_FILENAME

    def to_state_dict(self) -> dict:
        """Serializable snapshot of everything we want to survive restart."""
        return {
            "version": STATE_VERSION,
            "session_id": self.session_id,
            "source_ref_path": self.source_ref_path,
            "target_ref_path": self.target_ref_path,
            "proxy_ref_path": self.proxy_ref_path,
            "source_uploaded_as_fbx": self.source_uploaded_as_fbx,
            "target_uploaded_as_fbx": self.target_uploaded_as_fbx,
            "target_dt_available": self.target_dt_available,
            "poses": [
                {"pose_id": p.pose_id, "name": p.name, "path": p.path}
                for p in self.poses.values()
            ],
            "markers": [[int(a), int(b)] for a, b in self.markers],
            "results": {pid: path for pid, path in self.results.items()},
            "wrap_path": self.wrap_path,
            "wrap_mode": self.wrap_mode,
            "wrap_region": self.wrap_region,
            "refit_state": self.refit_state,
            "last_seen": self.last_seen,
            "last_job": self.last_job,
        }

    def save(self) -> None:
        """Atomically write state.json. Safe to call after any mutation."""
        path = self._state_path()
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as fp:
                json.dump(self.to_state_dict(), fp, indent=2)
            # os.replace is atomic on Windows when target exists
            tmp.replace(path)
        except OSError as e:
            # Best-effort persistence — don't crash the request if disk is full
            # or the file is briefly locked.
            print(f"[session {self.session_id}] save failed: {e}")

    @classmethod
    def from_state_dict(cls, data: dict, workspace: Path) -> Optional["Session"]:
        """
        Rehydrate a Session from a state.json blob. Files that no longer exist
        on disk are dropped silently — the user will see the gaps reflected in
        the summary (e.g. poses they had uploaded are gone if the .obj files
        were manually deleted).
        """
        if data.get("version") != STATE_VERSION:
            return None  # forward-compat hook; bump VERSION on breaking changes
        sid = data.get("session_id")
        if not sid:
            return None

        def _check(p: Optional[str]) -> Optional[str]:
            if p and Path(p).exists():
                return p
            return None

        source_ref_path = _check(data.get("source_ref_path"))
        target_ref_path = _check(data.get("target_ref_path"))
        proxy_ref_path = _check(data.get("proxy_ref_path"))

        poses: Dict[str, Pose] = {}
        for entry in data.get("poses", []) or []:
            try:
                pose_id = entry["pose_id"]
                name = entry["name"]
                path = entry["path"]
            except (KeyError, TypeError):
                continue
            if Path(path).exists():
                poses[pose_id] = Pose(pose_id=pose_id, name=name, path=path)

        results: Dict[str, str] = {}
        for pid, path in (data.get("results") or {}).items():
            if Path(path).exists() and pid in poses:
                results[pid] = path

        wrap_path = _check(data.get("wrap_path"))
        try:
            last_seen = float(data.get("last_seen") or time.time())
        except (TypeError, ValueError):
            last_seen = time.time()

        markers_raw = data.get("markers", []) or []
        markers: List[Tuple[int, int]] = []
        for m in markers_raw:
            try:
                markers.append((int(m[0]), int(m[1])))
            except (KeyError, TypeError, IndexError, ValueError):
                continue

        return cls(
            session_id=sid,
            workspace=workspace,
            source_ref_path=source_ref_path,
            target_ref_path=target_ref_path,
            proxy_ref_path=proxy_ref_path,
            source_uploaded_as_fbx=bool(data.get("source_uploaded_as_fbx", False)),
            target_uploaded_as_fbx=bool(data.get("target_uploaded_as_fbx", False)),
            target_dt_available=bool(data.get("target_dt_available", True)),
            poses=poses,
            markers=markers,
            results=results,
            wrap_path=wrap_path,
            wrap_mode=str(data.get("wrap_mode") or "wrap"),
            wrap_region=data.get("wrap_region") if isinstance(data.get("wrap_region"), dict) else None,
            refit_state=data.get("refit_state") if isinstance(data.get("refit_state"), dict) else None,
            last_seen=last_seen,
            last_job=data.get("last_job") if isinstance(data.get("last_job"), dict) else None,
        )


class SessionStore:
    def __init__(
        self,
        workspace_root: Path,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
        max_total_storage_bytes: int = 0,
    ):
        self.root = workspace_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, Session] = {}
        self._global_lock = threading.Lock()
        self.ttl_seconds = ttl_seconds
        self.sweep_interval_seconds = sweep_interval_seconds
        # Global disk budget across all sessions; 0 disables LRU eviction.
        self.max_total_storage_bytes = max_total_storage_bytes
        self._rehydrate_from_disk()
        # Also run a sweep immediately on startup so anything that aged out
        # between restarts is cleaned up before the first request lands.
        self.sweep_stale()
        self._start_sweeper()

    # ------------------------------------------------------------------ #
    # Rehydrate (with TTL filter)                                        #
    # ------------------------------------------------------------------ #

    def _rehydrate_from_disk(self) -> None:
        """Scan workspace dirs and reload any sessions with a valid state.json.

        Sessions whose `last_seen` is older than the TTL are deleted from disk
        instead of being loaded — same effect as the periodic sweep but
        applied to the startup catch-up batch.
        """
        loaded = 0
        skipped = 0
        expired = 0
        now = time.time()
        for ws in self.root.iterdir():
            if not ws.is_dir():
                continue
            state_path = ws / STATE_FILENAME
            if not state_path.exists():
                # Workspace from before we added persistence, or a fresh dir.
                # Skip silently — it just won't be discoverable.
                skipped += 1
                continue
            try:
                with state_path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except (OSError, json.JSONDecodeError) as e:
                print(f"[session restore] {ws.name}: failed to read state.json: {e}")
                skipped += 1
                continue
            s = Session.from_state_dict(data, ws)
            if s is None:
                print(f"[session restore] {ws.name}: invalid state.json")
                skipped += 1
                continue
            if now - s.last_seen > self.ttl_seconds:
                # Aged out while the backend was down — drop it now.
                s.delete_workspace()
                expired += 1
                continue
            (ws / "poses").mkdir(exist_ok=True)
            (ws / "results").mkdir(exist_ok=True)
            s.ttl_seconds = self.ttl_seconds
            self._sessions[s.session_id] = s
            loaded += 1
        if loaded or skipped or expired:
            print(f"[session restore] loaded={loaded} expired={expired} skipped={skipped}")

    # ------------------------------------------------------------------ #
    # CRUD                                                                #
    # ------------------------------------------------------------------ #

    def create(self) -> Session:
        # The session_id is the ONLY credential guarding a session's meshes
        # (capability model, no separate auth). A short id could in principle
        # be guessed/scraped, so use a long, URL- and filesystem-safe random
        # token (~192 bits) instead of a truncated UUID. token_urlsafe yields
        # [A-Za-z0-9_-], all safe as a directory name on Windows/POSIX.
        session_id = secrets.token_urlsafe(24)
        ws = self.root / session_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "poses").mkdir(exist_ok=True)
        (ws / "results").mkdir(exist_ok=True)
        s = Session(session_id=session_id, workspace=ws, ttl_seconds=self.ttl_seconds)
        s.save()  # write initial state.json so it's discoverable
        with self._global_lock:
            self._sessions[session_id] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        with self._global_lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Explicit delete (e.g. user clicked 'New Session' or tab close beacon).

        Returns True if the session existed.
        """
        with self._global_lock:
            s = self._sessions.pop(session_id, None)
        if s is None:
            # Best-effort fallback: workspace dir may exist on disk even if
            # the session wasn't in memory (e.g. backend restarted just now).
            ws = self.root / session_id
            if ws.exists() and ws.is_dir():
                shutil.rmtree(ws, ignore_errors=True)
                print(f"[session delete] {session_id} (from disk only)")
                return True
            return False
        s.delete_workspace()
        print(f"[session delete] {session_id}")
        return True

    # ------------------------------------------------------------------ #
    # TTL sweeper                                                         #
    # ------------------------------------------------------------------ #

    def sweep_stale(self) -> int:
        """Delete sessions whose last_seen is older than TTL. Returns count."""
        now = time.time()
        with self._global_lock:
            stale_ids = [
                sid for sid, s in self._sessions.items()
                if now - s.last_seen > self.ttl_seconds
            ]
            stale = [self._sessions.pop(sid) for sid in stale_ids]
        for s in stale:
            s.delete_workspace()
        if stale:
            ages = [int(now - s.last_seen) for s in stale]
            print(f"[session sweep] dropped {len(stale)} idle sessions (ages s: {ages})")
        return len(stale)

    # ------------------------------------------------------------------ #
    # Global disk quota (LRU eviction)                                    #
    # ------------------------------------------------------------------ #

    def enforce_global_quota(self) -> int:
        """If total disk usage across all sessions exceeds the configured cap,
        evict least-recently-used sessions (oldest last_seen first) until back
        under budget. Returns the number evicted. No-op if the cap is 0.

        This is the disk-space backstop for the session TTL: a burst of large
        uploads can blow past the disk budget long before TTL would reclaim
        anything, so we drop the coldest sessions early.
        """
        cap = self.max_total_storage_bytes
        if cap <= 0:
            return 0
        with self._global_lock:
            sessions = list(self._sessions.values())
        sized = [(s, _dir_size_bytes(s.workspace)) for s in sessions]
        total = sum(sz for _, sz in sized)
        if total <= cap:
            return 0
        evicted = 0
        # Coldest first. Never evict a session touched in the last minute — it
        # may be mid-upload (its files on disk but last_seen just bumped).
        now = time.time()
        for s, sz in sorted(sized, key=lambda x: x[0].last_seen):
            if total <= cap:
                break
            if now - s.last_seen < 60:
                continue
            with self._global_lock:
                popped = self._sessions.pop(s.session_id, None)
            if popped is None:
                continue
            popped.delete_workspace()
            total -= sz
            evicted += 1
        if evicted:
            print(f"[session quota] evicted {evicted} LRU session(s); "
                  f"~{total / (1024**3):.2f} GB now in use (cap "
                  f"{cap / (1024**3):.2f} GB)")
        return evicted

    def _start_sweeper(self) -> None:
        """Background thread: every sweep_interval_seconds, drop idle sessions
        and enforce the global disk budget."""
        def loop():
            while True:
                time.sleep(self.sweep_interval_seconds)
                try:
                    self.sweep_stale()
                    self.enforce_global_quota()
                except Exception as e:  # noqa: BLE001
                    # The sweeper must not die — any exception is a bug we
                    # want logged but we keep the thread alive.
                    print(f"[session sweep] error: {e}")
        t = threading.Thread(target=loop, name="session-sweeper", daemon=True)
        t.start()
