"""
FastAPI backend for Chameleon Warp.

Routing + pipeline orchestration. The math lives in `dt_core/`, serving
concerns (config, quotas, jobs, logging) in `infra/`, session state in
`sessions.py`. The full endpoint list is in docs/API.md; the interactive
OpenAPI docs are served at /docs while the backend is running.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from dt_core import meshlib
from dt_core import regions
from dt_core.align import umeyama, apply_similarity
from dt_core.smoothing import taubin_smooth
from dt_core.correspondence import (
    SourceSetup, TargetSetup, RegionSpec,
    compute_correspondence_with_setup, get_vertex_normals,
    precompute_source, precompute_target,
)
from dt_core.surface import (
    build_surface, closest_points_on_surface, damp_folds, project_onto_surface,
)
from dt_core.transformation import Transformation
from dt_core.refit import run_refit, build_refit_field, PRESETS as REFIT_PRESETS
import fbx_io
import obj_io
from sessions import Pose, Session, SessionStore

# All serving / HTTP-concern config lives in infra/. Keep this file
# focused on routing + pipeline orchestration.
from infra.config import Config, set_config, get_config
from infra.compute_queue import init_queue, get_queue
from infra.jobs import init_registry, get_registry, Job
from infra.limits import (
    check_upload_size, check_aggregate_upload_size,
    check_poses_count, check_session_storage,
    check_source_triangles, check_target_triangles,
)
from infra.logging import setup_logging, RequestIdMiddleware
from infra.upload_guard import upload_lock

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_ROOT = BACKEND_DIR / "workspace"

# Initialize the global config from env. This must happen BEFORE any infra
# module reads it (the middleware constructor is fine because get_config
# is called lazily inside dispatch()).
set_config(Config.from_env(DEFAULT_WORKSPACE_ROOT))
_cfg = get_config()

# Structured logging — both for our own log.info() calls and for tagging
# uvicorn's access lines with the same request_id.
setup_logging()

store = SessionStore(
    _cfg.workspace_root,
    ttl_seconds=_cfg.session_ttl_seconds,
    sweep_interval_seconds=_cfg.session_sweep_interval_seconds,
    max_total_storage_bytes=_cfg.max_total_storage_bytes,
)

# Serialize CPU-heavy jobs (/run, /wrap). All other endpoints stay parallel.
init_queue(concurrency=_cfg.compute_concurrency)

# Async job bookkeeping: /run and /wrap return a job_id and compute on a
# background thread so the request doesn't sit open for minutes (proxies and
# browsers drop idle connections) and survives the client closing its tab.
init_registry()

app = FastAPI(title="Chameleon Warp")

# Middleware order matters — Starlette processes them outside-in in
# REVERSE add order. So the LAST add_middleware runs FIRST. We want:
#   1. RequestId (assign rid before anything else can log)
#   2. CORS
#   3. GZip (innermost: number-heavy mesh JSON shrinks ~5x)
# To get that order, add in the reverse sequence below.
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(RequestIdMiddleware)

# CORS — only matters when the UI is served from a different origin than the
# API (e.g. the Vite dev server without its proxy). Origins come from env.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

class MarkerPair(BaseModel):
    source: int = Field(..., ge=0)
    target: int = Field(..., ge=0)


class MarkersPayload(BaseModel):
    markers: List[MarkerPair]



class BodyMaskPayload(BaseModel):
    # The covered-body vertex indices to persist (paint edits). Validated
    # against the body vertex count on the endpoint.
    hidden_vertices: List[int]


class BodyMaskDilatePayload(BaseModel):
    # +N grows the mask by N adjacency rings, -N erodes by N.
    rings: int = Field(..., ge=-20, le=20)


class RefitStatePayload(BaseModel):
    # Opaque refit working-state blob (gizmo placement, pins, frozen, preset,
    # advanced knobs + the source name it was captured against). Shape owned by
    # the frontend; the backend only round-trips it. Bounded to keep state.json
    # sane — frozen-paint sets are the only thing that can get large.
    refit_state: Optional[dict] = None


class RunPayload(BaseModel):
    iterations: int = Field(default=8, ge=1, le=20)
    smoothness: float = Field(default=1.0, ge=0.0)
    identity_weight: float = Field(default=0.001, ge=0.0)


class RegionPayload(BaseModel):
    """Boundary-loop region for a partial (masked) wrap.

    waypoints     — source vertex indices the user clicked around the patch;
                    the backend stitches them into a dense loop via shortest
                    paths on the mesh edge graph.
    seed          — one source vertex strictly INSIDE the loop, used to flood-
                    fill the editable interior.
    feather_width — world-space width of the smooth transition band just
                    inside the loop (0 = hard seam).
    """
    waypoints: List[int] = Field(..., min_items=3)
    # Optional: a vertex on the side you want editable. If omitted, the
    # smaller side the loop carves out is used (usually the patch you drew
    # around). `invert` flips to the other side.
    seed: Optional[int] = Field(default=None, ge=0)
    invert: bool = False
    feather_width: float = Field(default=0.0, ge=0.0)
    # Seam sharpness: how steeply the feather weight falls off from the
    # boundary inward. 1 = gentle/linear; higher = the feather concentrates
    # near the seam (firmer seam, faster wrap freedom inside). Default 2.
    feather_exponent: float = Field(default=2.0, ge=1.0, le=6.0)


class WrapPayload(BaseModel):
    """
    Parameters for Wrap / Fit mode.

    Wrap uses only the correspondence (inflation) step from the DT pipeline.
    The output is the source mesh deformed to roughly match the target shape,
    keeping the source's vertex count and topology — exactly what the user
    needs as a character-customization shape key.

    use_closest_point=False switches to Fit mode: the shrink-wrap term is
    dropped, so only the marker vertices are pinned to their target positions
    and the rest of the source deforms minimally (smoothness + identity).
    Use for attaching accessories (beard -> head) where the bulk of the
    source must keep its shape instead of snapping onto the target surface.
    """
    iterations: int = Field(default=8, ge=1, le=20)
    smoothness: float = Field(default=1.0, ge=0.0)
    identity_weight: float = Field(default=0.001, ge=0.0)
    use_closest_point: bool = True
    # Taubin smoothing passes applied to the result after the solve, to remove
    # the high-frequency wobble the closest-point step injects. 0 = off. For a
    # regional wrap only the editable patch is smoothed (frozen stays exact).
    smooth_result: int = Field(default=0, ge=0, le=50)
    # Optional regional wrap: deform only inside a boundary loop drawn on the
    # source, freeze everything outside, feather the seam. None = whole-mesh.
    region: Optional[RegionPayload] = None
    # Bring the target into the source's coordinate frame using the marker
    # pairs (similarity: scale+rotation+translation) before wrapping. Needed
    # when the two meshes were authored at different scales / positions, which
    # otherwise makes the closest-point step collapse the result.
    align_to_source: bool = True
    # --- surface-matching robustness (see dt_core.surface) ---
    # Normal-compatibility limit for closest-point matches and projection.
    match_max_angle_deg: float = Field(default=60.0, ge=10.0, le=90.0)
    # Floor of the adaptive match-distance cut, as a fraction of the target's
    # bbox diagonal. Matches farther than max(floor, 3*median) are dropped so
    # source areas with no target coverage don't stretch toward a wrong patch.
    match_distance_frac: float = Field(default=0.02, gt=0.0, le=1.0)
    # Final shrink-wrap: snap the result onto the target surface after the
    # solve (then alternate smooth->re-project if smooth_result > 0). Only
    # applies in Wrap mode; Fit mode never projects.
    project_result: bool = True
    # Max snap distance for the final projection, fraction of bbox diagonal.
    project_distance_frac: float = Field(default=0.02, gt=0.0, le=1.0)
    # Soft markers: once the shrink-wrap weight schedule reaches its strong
    # phase, marker pins relax into WEAK penalty rows, so an imprecisely
    # placed marker (classic case: wrong side of an ear/fold) gets pulled
    # back onto the right surface instead of denting it. Accurate markers
    # are unaffected. False = classic hard pins. Ignored in Fit mode.
    soft_markers: bool = True
    soft_markers_weight: float = Field(default=10.0, gt=0.0)


class RefitPayload(BaseModel):
    """Parameters for Refit — fitting a garment / accessory (the source) onto a
    body (the target). Unlike Wrap, the garment keeps its own shape and only its
    interface grips the body; a preset picks the conform field + stiffness +
    collision strategy (see dt_core.refit).

    source_ref = the garment / accessory (deformed); target_ref = the body.
    """
    preset: str = Field(default="accessory")   # accessory | cloth | armor | skintight
    iterations: int = Field(default=10, ge=1, le=20)
    # Manual place / non-uniform resize applied to the garment before the solve
    # (the frontend gizmo). Row-major 4x4 (16 floats). None = identity.
    pre_transform: Optional[List[float]] = Field(default=None, min_items=16, max_items=16)
    # Seam grip width (world units); None = preset default (fraction of bbox diag).
    grip_width: Optional[float] = Field(default=None, gt=0.0)
    # Seam proximity gate (world units): an open boundary grips only if within
    # this distance of the body, so a far outer rim (double-wall collar) stays
    # free and keeps its standoff. None = preset default; 0 = gate off.
    seam_depth: Optional[float] = Field(default=None, ge=0.0)
    # Pin "preserve" verts: outer-wall source vertices a needle pierced. They
    # get released (never grip the body) so a double-wall standoff is kept; the
    # inner wall grips via a normal marker. Empty = none.
    preserve: List[int] = Field(default_factory=list, max_items=5000)
    # Frozen paint: source verts the user brushed as "do not deform at all".
    # They lose every grip (seam/contact/pins), ride the fit as a near-rigid
    # patch via the bind pass, and collision never moves them.
    frozen: List[int] = Field(default_factory=list, max_items=200000)
    # Pin patch radius (world units); None = 6% of the garment bbox diagonal.
    pin_radius: Optional[float] = Field(default=None, gt=0.0)
    # Collision offset (world units); None = the effective cloth thickness.
    offset: Optional[float] = Field(default=None, ge=0.0)
    # Contact band (world units): garment closer than `contact_tight` to the
    # body grips fully; the grip fades to 0 by `contact_free` (hanging cloth).
    # None = preset defaults (fractions of the body bbox diagonal).
    contact_tight: Optional[float] = Field(default=None, ge=0.0)
    contact_free: Optional[float] = Field(default=None, gt=0.0)
    # Cloth thickness (world units): fit targets sit this far above the skin.
    thickness: Optional[float] = Field(default=None, ge=0.0)
    # Advanced overrides of the preset (None = keep the preset's value).
    stiffness: Optional[float] = Field(default=None, ge=0.0)
    smoothness: Optional[float] = Field(default=None, ge=0.0)
    collision: Optional[str] = Field(default=None)   # none | push_out | hide_body
    # Auto-polish: a small trust-regioned pre-alignment of the placement so the
    # contact region sits at the cloth standoff. When it fires, the effective
    # placement rides back in stats.refit.polish.matrix (16 floats) so the
    # frontend can adopt it into the gizmo. Off by default.
    auto_polish: bool = Field(default=False)
    # Growth continuation: solve against a sequence of eroded bodies growing to
    # the true shape (deep-penetration case). None = preset default (1 = today's
    # single-body solve); auto-collapses to 1 when the placement barely
    # penetrates. Overrides the preset when set.
    growth_steps: Optional[int] = Field(default=None, ge=1, le=6)
    # Layer classification mode: "legacy" (per-vertex occlusion|opposed, default)
    # or "auto" (score-based body-side voting + fan-out occlusion, then
    # smoothing/hysteresis/component policy — see dt_core.layers). Auto adds a
    # per-vertex layer id to the /refit_field response for the overlay + the
    # "→ frozen paint" prefill.
    layer_mode: str = Field(default="legacy", pattern="^(legacy|auto)$")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

VALID_REF_ROLES = {"source_ref", "target_ref", "proxy_ref"}


def _get_session(session_id: str) -> Session:
    """Look up a session and bump its last_seen so it stays in the TTL window.

    Every endpoint that takes a session_id goes through here, which means
    any user activity (even idle reads like GET /session) postpones the
    sweeper from reclaiming the workspace.
    """
    s = store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    s.touch()
    return s


def _save_upload(upload: UploadFile, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fp:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            fp.write(chunk)


def _require_obj(filename: str) -> None:
    if not filename.lower().endswith(".obj"):
        raise HTTPException(status_code=400, detail=f"Only .obj files are accepted (got {filename!r})")


def _is_fbx(filename: str) -> bool:
    return filename.lower().endswith(".fbx")


def _require_obj_or_fbx(filename: str) -> None:
    f = filename.lower()
    if not (f.endswith(".obj") or f.endswith(".fbx")):
        raise HTTPException(status_code=400, detail=f"Only .obj or .fbx files are accepted (got {filename!r})")


def _drop_all_poses(s: Session) -> int:
    """Wipe all uploaded poses and their cached preview files. Returns count dropped."""
    n = 0
    for pose in list(s.poses.values()):
        try:
            Path(pose.path).unlink(missing_ok=True)
        except OSError:
            pass
        n += 1
    s.poses.clear()
    # Drop results too (they were generated against now-invalid poses)
    for r in list(s.results.values()):
        try:
            Path(r).unlink(missing_ok=True)
        except OSError:
            pass
    s.results.clear()
    return n


def _mesh_to_json(mesh: meshlib.Mesh) -> dict:
    m = mesh.to_third_dimension(copy=False)
    return {
        "vertices": m.vertices.astype(float).flatten().tolist(),
        "faces": m.faces.astype(int).flatten().tolist(),
        "vertex_count": int(len(m.vertices)),
        "face_count": int(len(m.faces)),
    }


# LRU cache of SERIALIZED mesh JSON, keyed by (path, mtime, size). A preview
# session re-requests the same files (viewport reloads, several clients, the
# playback scrubber) and each miss costs an OBJ re-parse + a megabytes-scale
# json.dumps — the two dominant costs of the preview endpoints. Byte budget
# instead of entry count because meshes vary from KBs to tens of MBs.
_MESH_JSON_BUDGET = 256 * 1024 * 1024
_mesh_json_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_mesh_json_cache_lock = threading.Lock()


def _mesh_json_response(path: str) -> Response:
    st_ = os.stat(path)
    key = (str(path), st_.st_mtime_ns, st_.st_size)
    with _mesh_json_cache_lock:
        payload = _mesh_json_cache.get(key)
        if payload is not None:
            _mesh_json_cache.move_to_end(key)
    if payload is None:
        data = _mesh_to_json(meshlib.Mesh.load(path))
        payload = json.dumps(data, separators=(",", ":")).encode()
        with _mesh_json_cache_lock:
            _mesh_json_cache[key] = payload
            total = sum(len(v) for v in _mesh_json_cache.values())
            while total > _MESH_JSON_BUDGET and len(_mesh_json_cache) > 1:
                _, evicted = _mesh_json_cache.popitem(last=False)
                total -= len(evicted)
    return Response(content=payload, media_type="application/json")


# -----------------------------------------------------------------------------
# Sessions
# -----------------------------------------------------------------------------

@app.post("/api/sessions")
def create_session():
    s = store.create()
    return s.to_summary()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    s = _get_session(session_id)
    return s.to_summary()


@app.get("/api/queue")
def get_queue_status():
    """
    Snapshot of the compute queue.

    Returns:
      • `waiting`     — how many requests are blocked waiting for a slot
      • `running`     — how many slots are currently occupied
      • `concurrency` — total number of slots (env: DT_COMPUTE_CONCURRENCY)

    The browser polls this while it has a /run or /wrap request in flight
    so it can render "you are #2 in queue" feedback. The endpoint is cheap
    (just reads two counters).
    """
    return get_queue().status()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Explicit session teardown.

    Used by:
      • the "New Session" button (delete old before creating new)
      • the tab-close sendBeacon (best-effort cleanup; may or may not arrive
        before the browser kills the request, but if it does we reclaim the
        workspace immediately instead of waiting for the TTL sweep).

    Always returns 200 — there's no useful client behavior on a 404 here,
    and the call is fire-and-forget on the browser side.
    """
    existed = store.delete(session_id)
    # Drop any job bookkeeping so a stale terminal job can't linger for a
    # reused id (ids are random, but be tidy).
    get_registry().forget(session_id)
    return {"deleted": existed, "session_id": session_id}


# -----------------------------------------------------------------------------
# Upload poses (one or many) — declared BEFORE the parametric /upload/{role}
# route so it doesn't get swallowed by it.
# -----------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/upload/poses")
def upload_poses(session_id: str, files: List[UploadFile] = File(...)):
    s = _get_session(session_id)
    if not s.source_ref_path:
        raise HTTPException(status_code=400, detail="Upload source_ref first so poses can be validated against it")

    # Quota gate: total bytes of this batch, and pose-count cap.
    sizes = [getattr(f, "size", 0) or 0 for f in files]
    check_aggregate_upload_size(sizes)
    check_poses_count(len(s.poses), adding=len(files))
    check_session_storage(s.workspace, additional_bytes=sum(sizes))

    with upload_lock(s):
        source_ref = meshlib.Mesh.load(s.source_ref_path)
        expected_verts = len(source_ref.vertices)
        expected_faces = len(source_ref.faces)

        accepted = []
        rejected = []
        for upload in files:
            _require_obj(upload.filename)
            pose_id = uuid.uuid4().hex[:10]
            dst = s.workspace / "poses" / f"{pose_id}.obj"
            _save_upload(upload, dst)
            try:
                mesh = meshlib.Mesh.load(str(dst))
            except Exception as e:  # noqa: BLE001
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
                rejected.append({"name": upload.filename, "reason": f"parse error: {e}"})
                continue
            if len(mesh.vertices) != expected_verts or len(mesh.faces) != expected_faces:
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
                rejected.append({
                    "name": upload.filename,
                    "reason": (
                        f"topology mismatch with source_ref "
                        f"(got {len(mesh.vertices)} verts / {len(mesh.faces)} faces, "
                        f"expected {expected_verts} / {expected_faces})"
                    ),
                })
                continue
            pose = Pose(pose_id=pose_id, name=upload.filename, path=str(dst))
            s.poses[pose_id] = pose
            accepted.append({"pose_id": pose_id, "name": upload.filename})

        s.save()
    # Uploads are the only thing that grows disk meaningfully — enforce the
    # global budget now so a burst can't outrun the periodic sweeper.
    store.enforce_global_quota()
    return {
        "session": s.to_summary(),
        "accepted": accepted,
        "rejected": rejected,
    }


# -----------------------------------------------------------------------------
# Upload reference meshes
# -----------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/upload/{role}")
def upload_reference(session_id: str, role: str, file: UploadFile = File(...)):
    """
    Upload a reference mesh. Accepts .obj or .fbx.

    If FBX uploaded as `source_ref`, the script also extracts every non-basis
    shape key as a separate pose, replacing any previously uploaded poses.
    For `target_ref`, FBX is accepted but only the basis mesh is used (target
    shape keys are not interesting for us — we generate them on output).
    """
    if role not in VALID_REF_ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of {sorted(VALID_REF_ROLES)}")
    s = _get_session(session_id)
    # proxy_ref (wardrobe basemesh) is a plain static mesh — no shape-key
    # extraction, so it is .obj only; everything else may be .obj or .fbx.
    if role == "proxy_ref":
        _require_obj(file.filename)
    else:
        _require_obj_or_fbx(file.filename)

    # Quota gate before we write anything to disk or fire up Blender.
    upload_size = getattr(file, "size", 0) or 0
    check_upload_size(upload_size, file.filename or "")
    check_session_storage(s.workspace, additional_bytes=upload_size)

    is_fbx = _is_fbx(file.filename)

    # Serialize the whole upload — fixes the race where the user starts a
    # slow FBX extract for source then clicks target before it finishes
    # and both handlers mutate s.poses / s.*_ref_path concurrently.
    with upload_lock(s):
        if is_fbx:
            # Save the raw .fbx into the session workspace, then call Blender
            # to extract basis + shape keys.
            if not fbx_io.blender_available():
                raise HTTPException(
                    status_code=400,
                    detail=("FBX upload requires a Blender installation. Install Blender "
                            "or set the BLENDER_PATH env var to its executable, then retry."),
                )
            fbx_dst = s.workspace / f"{role}_upload.fbx"
            _save_upload(file, fbx_dst)

            if role == "source_ref":
                extract_dir = s.workspace / "poses_extracted"
            else:
                extract_dir = s.workspace / "target_extracted"
            if extract_dir.exists():
                for f in extract_dir.iterdir():
                    try:
                        f.unlink()
                    except OSError:
                        pass
            extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = fbx_io.extract_shape_keys(str(fbx_dst), str(extract_dir))
            except fbx_io.BlenderError as e:
                try:
                    fbx_dst.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HTTPException(status_code=400, detail=str(e))

            basis_obj_path = s.workspace / f"{role}.obj"
            try:
                basis_obj_path.unlink(missing_ok=True)
            except OSError:
                pass
            Path(result.basis_path).replace(basis_obj_path)

            # Triangle-count caps on the extracted basis. Checked BEFORE any
            # session mutation so a too-dense source doesn't wipe existing
            # poses. On rejection, drop the raw FBX + extracted basis.
            basis_mesh = meshlib.Mesh.load(str(basis_obj_path))
            n_tris = len(basis_mesh.faces)
            dt_ok = True
            try:
                if role == "source_ref":
                    check_source_triangles(n_tris)
                else:
                    dt_ok = check_target_triangles(n_tris)
            except HTTPException:
                basis_obj_path.unlink(missing_ok=True)
                fbx_dst.unlink(missing_ok=True)
                raise

            if role == "source_ref":
                _drop_all_poses(s)
                # The FBX-extracted poses are produced atomically here, so
                # the per-session count cap also applies — enforce it.
                check_poses_count(0, adding=len(result.poses))
                for p in result.poses:
                    pose_id = uuid.uuid4().hex[:10]
                    pose_dst = s.workspace / "poses" / f"{pose_id}.obj"
                    Path(p.path).replace(pose_dst)
                    s.poses[pose_id] = Pose(
                        pose_id=pose_id,
                        name=p.name,
                        path=str(pose_dst),
                    )
                s.source_ref_path = str(basis_obj_path)
                s.source_uploaded_as_fbx = True
                s.invalidate_source_cache()
            else:
                s.target_ref_path = str(basis_obj_path)
                s.target_uploaded_as_fbx = True
                s.target_dt_available = dt_ok
                s.invalidate_target_cache()

            s.save()
            summary = s.to_summary()
            summary["fbx_extract"] = {
                "mesh_name": result.mesh_name,
                "vertex_count": result.vertex_count,
                "polygon_count": result.polygon_count,
                "extracted_poses": [p.name for p in result.poses],
            }
            if role == "target_ref" and not dt_ok:
                summary["target_warning"] = (
                    f"Target has {n_tris:,} triangles — too dense for DT transfer "
                    f"(Run), but Wrap works fine. Decimate it if you need Run."
                )
            store.enforce_global_quota()
            return summary

        # --- plain .obj path ---
        dst = s.workspace / f"{role}.obj"
        _save_upload(file, dst)
        try:
            mesh = meshlib.Mesh.load(str(dst))
        except Exception as e:  # noqa: BLE001
            try:
                dst.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=400, detail=f"Failed to parse .obj: {e}")

        # Triangle-count caps. On rejection, remove the just-saved file so a
        # too-dense upload doesn't linger against the storage quota.
        n_tris = len(mesh.faces)
        dt_ok = True
        try:
            if role == "target_ref":
                dt_ok = check_target_triangles(n_tris)
            else:                         # source_ref / proxy_ref — basemesh
                check_source_triangles(n_tris)
        except HTTPException:
            dst.unlink(missing_ok=True)
            raise

        if role == "source_ref":
            _drop_all_poses(s)
            s.source_ref_path = str(dst)
            s.source_uploaded_as_fbx = False
            s.invalidate_source_cache()
        elif role == "proxy_ref":
            s.proxy_ref_path = str(dst)
        else:
            s.target_ref_path = str(dst)
            s.target_uploaded_as_fbx = False
            s.target_dt_available = dt_ok
            s.invalidate_target_cache()
        s.save()
        summary = s.to_summary()
        if role == "target_ref" and not dt_ok:
            summary["target_warning"] = (
                f"Target has {n_tris:,} triangles — too dense for DT transfer "
                f"(Run), but Wrap works fine. Decimate it if you need Run."
            )
        store.enforce_global_quota()
        return summary


@app.delete("/api/sessions/{session_id}/poses/{pose_id}")
def delete_pose(session_id: str, pose_id: str):
    s = _get_session(session_id)
    pose = s.poses.pop(pose_id, None)
    if pose is None:
        raise HTTPException(status_code=404, detail="Pose not found")
    Path(pose.path).unlink(missing_ok=True)
    # Also drop any cached result for that pose
    res_path = s.results.pop(pose_id, None)
    if res_path:
        Path(res_path).unlink(missing_ok=True)
    s.save()
    return s.to_summary()


# -----------------------------------------------------------------------------
# Mesh JSON for viewports
# -----------------------------------------------------------------------------

@app.get("/api/sessions/{session_id}/mesh/{role}")
def get_mesh(session_id: str, role: str):
    # "wrap" is handled here too: a separate fixed route /mesh/wrap would be
    # shadowed by this dynamic one (registered first, FastAPI matches in
    # registration order), which used to 400 and silently kill the wrap/fit
    # viewport preview.
    s = _get_session(session_id)
    if role == "source_ref":
        path = s.source_ref_path
    elif role == "target_ref":
        path = s.target_ref_path
    elif role == "proxy_ref":
        path = s.proxy_ref_path
    elif role == "wrap":
        if not s.wrap_path:
            raise HTTPException(status_code=404, detail="No wrap result yet")
        path = s.wrap_path
    else:
        raise HTTPException(status_code=400,
                            detail=f"role must be one of {sorted(VALID_REF_ROLES | {'wrap'})}")
    if not path:
        raise HTTPException(status_code=404, detail=f"{role} not uploaded yet")
    if role != "wrap":
        return _mesh_json_response(path)
    # Wrap responses merge the residual sidecar, so they skip the byte cache
    # (fetched once per wrap anyway).
    mesh = meshlib.Mesh.load(path)
    data = _mesh_to_json(mesh)
    if role == "wrap":
        # Attach the per-vertex residual (distance to target surface, as a
        # fraction of its bbox diagonal) for the viewport heatmap, if this
        # wrap computed one. Length must match — a mismatch means the file
        # describes a different mesh, so silently skip it.
        rp = s.workspace / "wrap" / "residual.json"
        if rp.exists():
            try:
                extra = json.loads(rp.read_text(encoding="utf-8"))
                if len(extra.get("distances_frac", [])) == data["vertex_count"]:
                    data["distances_frac"] = extra["distances_frac"]
                    data["residual"] = extra["residual"]
                if len(extra.get("grip_frac", [])) == data["vertex_count"]:
                    data["grip_frac"] = extra["grip_frac"]
            except (OSError, ValueError):
                pass
    return data


@app.get("/api/sessions/{session_id}/poses/{pose_id}/mesh")
def get_pose_mesh(session_id: str, pose_id: str):
    """Mesh JSON of an uploaded source pose, for the preview viewport."""
    s = _get_session(session_id)
    pose = s.poses.get(pose_id)
    if not pose:
        raise HTTPException(status_code=404, detail="Pose not found")
    return _mesh_json_response(pose.path)


@app.get("/api/sessions/{session_id}/results/{pose_id}/mesh")
def get_result_mesh(session_id: str, pose_id: str):
    """Mesh JSON of a computed deformed-target result, for the preview viewport."""
    s = _get_session(session_id)
    path = s.results.get(pose_id)
    if not path:
        raise HTTPException(status_code=404, detail="No result for that pose")
    return _mesh_json_response(path)


# -----------------------------------------------------------------------------
# Markers CRUD (whole-list replace)
# -----------------------------------------------------------------------------

@app.put("/api/sessions/{session_id}/markers")
def put_markers(session_id: str, payload: MarkersPayload):
    s = _get_session(session_id)
    s.markers = [(p.source, p.target) for p in payload.markers]
    s.save()
    return {"marker_count": len(s.markers)}


@app.get("/api/sessions/{session_id}/markers")
def get_markers(session_id: str):
    s = _get_session(session_id)
    return {"markers": [{"source": a, "target": b} for a, b in s.markers]}


# -----------------------------------------------------------------------------
# Cache key helpers
# -----------------------------------------------------------------------------

def _hash_mesh(mesh: meshlib.Mesh) -> str:
    """Content hash of a loaded mesh — same content → same key."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.vertices).tobytes())
    h.update(np.ascontiguousarray(mesh.faces).tobytes())
    return h.hexdigest()


def _hash_markers(markers: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(markers).tobytes()).hexdigest()


def _mapping_key(src_key: str, tgt_key: str, markers_key: str,
                 iterations: int, smoothness: float, identity_weight: float) -> str:
    h = hashlib.sha256()
    h.update(src_key.encode()); h.update(b"|")
    h.update(tgt_key.encode()); h.update(b"|")
    h.update(markers_key.encode()); h.update(b"|")
    h.update(f"{iterations}|{smoothness}|{identity_weight}".encode())
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Async job plumbing — /run and /wrap return a job_id immediately and compute
# on a background thread. See infra/jobs.py for the rationale.
# -----------------------------------------------------------------------------

def _detail_str(detail) -> str:
    """HTTPException.detail may be a str or a structured object; flatten it to
    a human-readable string for the job's error field."""
    return detail if isinstance(detail, str) else json.dumps(detail)


def _submit_job(s: Session, kind: str, fn):
    """Register a job for the session and run `fn(job)` on a daemon thread.

    The worker acquires the shared ComputeQueue slot (cross-session
    serialization is unchanged), flips the job to running, then calls `fn`
    which does the heavy compute and returns the endpoint's response dict.
    Terminal state is persisted into the session so a restart can still report
    the outcome.
    Raises HTTPException(409) if a job is already in progress.
    """
    try:
        job = get_registry().create(s.session_id, kind)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    def worker():
        try:
            # Blocks here while queued behind other jobs; the client sees
            # queue_position via GET /job in the meantime.
            with get_queue().occupy():
                job.mark_running()
                result = fn(job)
            job.mark_done(result)
        except HTTPException as e:
            job.mark_error(_detail_str(e.detail))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            job.mark_error(str(e))
        finally:
            try:
                s.last_job = job.persist_dict()
                s.save()
            except Exception as e:  # noqa: BLE001
                print(f"[job {job.job_id}] persist failed: {e}")

    threading.Thread(target=worker, name=f"{kind}-{job.job_id}", daemon=True).start()
    return job


def _job_response(job: Job) -> dict:
    return {"job_id": job.job_id, "kind": job.kind, "state": job.state}


@app.get("/api/sessions/{session_id}/job")
def get_job(session_id: str):
    """Status of the session's current (or last) background job.

    Returns the live job if one is registered in memory, else the persisted
    terminal snapshot from state.json (survives a backend restart), else
    404 if the session never ran a job. The browser polls this while a job
    is queued/running to render progress + queue position.
    """
    s = _get_session(session_id)
    job = get_registry().get(session_id)
    if job is not None:
        return job.snapshot(queue_position=get_registry().queue_position(job))
    if s.last_job:
        # No live job — hand back the persisted snapshot (always terminal).
        snap = dict(s.last_job)
        snap.setdefault("queue_position", -1)
        snap.setdefault("progress", {"stage": snap.get("state", "done"), "iter": 0, "total": 0})
        return snap
    raise HTTPException(status_code=404, detail="No job for this session")


# -----------------------------------------------------------------------------
# Run pipeline (with multi-level cache)
# -----------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/run")
def run_pipeline(session_id: str, payload: RunPayload):
    s = _get_session(session_id)
    if not s.source_ref_path or not s.target_ref_path:
        raise HTTPException(status_code=400, detail="Both source_ref and target_ref must be uploaded")
    if not s.poses:
        raise HTTPException(status_code=400, detail="Upload at least one source pose")
    if not s.target_dt_available:
        raise HTTPException(
            status_code=400,
            detail=("DT transfer is unavailable for this target — it's too dense "
                    "(the result would inherit the target's topology). Use Wrap "
                    "instead, or upload a lighter target."))
    # NOTE: the >=3 markers check happens inside the worker after loading the
    # meshes — a target with identical topology needs no markers at all
    # (identity mapping). A failed check surfaces as job.state == "error".
    job = _submit_job(s, "run", lambda job: _run_compute(s, payload, job))
    return _job_response(job)


def _run_compute(s: Session, payload: RunPayload, job: Job) -> dict:
    t_run = time.perf_counter()
    cache_hits: List[str] = []
    cache_misses: List[str] = []

    # The ComputeQueue slot is already held by the worker; here we only need
    # the per-session lock to protect state mutations against other endpoints.
    with s._lock:
        # ---- load reference meshes (cheap, .obj parser) ----
        source_ref = meshlib.Mesh.load(s.source_ref_path)
        target_ref = meshlib.Mesh.load(s.target_ref_path)
        markers = np.array(s.markers, dtype=np.int64)

        # Identical connectivity (e.g. target = wrap of this very source):
        # triangle i IS triangle i, correspondence would only rediscover
        # that. Skip it — markers become unnecessary. This is the
        # wrap -> blendshape-retarget workflow.
        same_topology = (
            len(source_ref.vertices) == len(target_ref.vertices)
            and source_ref.faces.shape == target_ref.faces.shape
            and bool(np.array_equal(source_ref.faces, target_ref.faces))
        )
        if not same_topology and len(markers) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 marker pairs are required (a target with "
                       "identical topology needs none — direct transfer)")

        src_key = _hash_mesh(source_ref)
        tgt_key = _hash_mesh(target_ref)
        markers_key = _hash_markers(markers)
        if same_topology:
            mapping_key = f"identity|{src_key}|{tgt_key}"
        else:
            mapping_key = _mapping_key(
                src_key, tgt_key, markers_key,
                payload.iterations, payload.smoothness, payload.identity_weight,
            )
        transf_key = mapping_key + f"|{payload.smoothness}"

        # ---- correspondence mapping ----
        if s.mapping_key == mapping_key and s.mapping is not None:
            cache_hits.append("mapping")
            mapping = s.mapping
        elif same_topology:
            cache_misses.append("mapping")
            n_tris = len(source_ref.faces)
            mapping = np.column_stack([np.arange(n_tris), np.arange(n_tris)])
            print(f"[run] identical topology ({n_tris} tris) -> identity mapping, "
                  f"correspondence skipped")
            s.mapping_key = mapping_key
            s.mapping = mapping
            s.transformation_key = None
            s.transformation = None
        else:
            # source setup (only depends on source mesh)
            if s.source_setup_key == src_key and s.source_setup is not None:
                cache_hits.append("source_setup")
                src_setup: SourceSetup = s.source_setup
            else:
                cache_misses.append("source_setup")
                src_setup = precompute_source(source_ref)
                s.source_setup_key = src_key
                s.source_setup = src_setup

            # target setup (only depends on target mesh)
            if s.target_setup_key == tgt_key and s.target_setup is not None:
                cache_hits.append("target_setup")
                tgt_setup: TargetSetup = s.target_setup
            else:
                cache_misses.append("target_setup")
                tgt_setup = precompute_target(target_ref)
                s.target_setup_key = tgt_key
                s.target_setup = tgt_setup

            cache_misses.append("mapping")
            job.set_progress("iterations", 0, payload.iterations)
            _, mapping = compute_correspondence_with_setup(
                src_setup, tgt_setup, markers,
                iterations=payload.iterations,
                smoothness=payload.smoothness,
                identity_weight=payload.identity_weight,
                progress_callback=lambda i, total: job.set_progress("iterations", i, total),
            )
            if mapping is None or len(mapping) == 0:
                raise HTTPException(status_code=500, detail="Correspondence produced an empty mapping")
            s.mapping_key = mapping_key
            s.mapping = mapping
            # invalidate downstream transformation cache
            s.transformation_key = None
            s.transformation = None

        # ---- transformation (mapping + smoothness) ----
        if s.transformation_key == transf_key and s.transformation is not None:
            cache_hits.append("transformation")
            transf: Transformation = s.transformation
        else:
            cache_misses.append("transformation")
            transf = Transformation(source_ref, target_ref, mapping, smoothness=payload.smoothness)
            s.transformation_key = transf_key
            s.transformation = transf

        # ---- clear previous result files ----
        for old in s.results.values():
            try:
                Path(old).unlink(missing_ok=True)
            except OSError:
                pass
        s.results.clear()

        # ---- per-pose transfer ----
        # The DT result has the TARGET's topology, so — exactly like /wrap does
        # for the source — we can restore the target author's original quads /
        # n-gons / UVs by rewriting the target OBJ's `v` lines with the deformed
        # positions. Falls back to the triangulated save on any mismatch.
        t_poses = time.perf_counter()
        n_poses = len(s.poses)
        polygons_preserved = True
        job.set_progress("transfer", 0, n_poses)
        for i, (pose_id, pose) in enumerate(s.poses.items()):
            pose_mesh = meshlib.Mesh.load(pose.path)
            deformed = transf(pose_mesh)
            out_path = s.workspace / "results" / f"{pose_id}.obj"
            try:
                obj_io.rewrite_obj_vertices(
                    s.target_ref_path,
                    deformed.to_third_dimension(copy=False).vertices,
                    out_path,
                    header="# Chameleon Warp (transfer): target topology + UVs, "
                           "deformed vertex positions",
                )
            except obj_io.ObjRewriteError as e:
                deformed.save_obj(str(out_path))
                polygons_preserved = False
                if i == 0:
                    print(f"[run] original-topology export skipped: {e}")
            s.results[pose_id] = str(out_path)
            job.set_progress("transfer", i + 1, n_poses)
        print(f"[run] {n_poses} poses transferred in {time.perf_counter() - t_poses:.2f}s "
              f"(polygons_preserved={polygons_preserved})")

    s.save()
    elapsed = time.perf_counter() - t_run
    print(f"[run] total {elapsed:.2f}s  hits={cache_hits}  misses={cache_misses}")

    return {
        "session": s.to_summary(),
        "mapping_size": int(len(mapping)),
        "identity_mapping": same_topology,
        "results": [
            {"pose_id": pid, "name": s.poses[pid].name} for pid in s.results.keys()
        ],
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "elapsed_seconds": elapsed,
        "polygons_preserved": polygons_preserved,
    }


# -----------------------------------------------------------------------------
# Wrap mode — character-variant shape key generation
#
# Workflow:
#   source_ref = the topology you want to preserve (e.g. your base character)
#   target_ref = the shape you want (e.g. wider-cheeks variant, different
#                topology is fine)
#   markers    = correspond paired feature points
#
# /wrap runs only the correspondence step from compute_correspondence_with_setup
# and persists its first return value (the inflated source mesh = source
# topology in target shape). The transformation/per-pose phase is skipped
# entirely — wrap doesn't need source poses.
#
# Use case: user downloads the wrap OBJ and in Blender uses "Join as Shapes"
# / "Blend From Shape" to attach it as a customization slider on top of the
# original character mesh.
# -----------------------------------------------------------------------------

def _check_marker_coverage(source: meshlib.Mesh, markers: np.ndarray) -> None:
    """Fit-mode sanity check: every connected island of the source mesh must
    contain at least one marker.

    Without the closest-point term, identity + smoothness only constrain
    deformation *gradients* — an island with no pinned vertex has a free
    translation (null space of A), so the solver would place it arbitrarily.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = source.faces[:, :3]
    n = len(source.vertices)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)

    used = np.unique(faces)  # ignore orphan vertices that belong to no face
    marked_labels = set(labels[markers[:, 0]].tolist())
    unmarked_sizes = [
        int((labels[used] == comp).sum())
        for comp in np.unique(labels[used])
        if comp not in marked_labels
    ]
    if unmarked_sizes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Fit mode: {len(unmarked_sizes)} disconnected island(s) of the "
                f"source mesh have no markers (sizes: {unmarked_sizes} verts). "
                f"Every island needs at least one source marker (ideally 3+), "
                f"otherwise its position is undetermined."
            ),
        )


def _build_region(source: meshlib.Mesh, rp: RegionPayload):
    """Turn a RegionPayload (clicked waypoints + seed) into a RegionSpec the
    solver understands. Returns (RegionSpec, loop_indices, interior_indices)
    so callers can also surface the stitched loop / editable set to the UI.

    Raises HTTPException(400) with an actionable message on any geometric
    problem (out-of-range index, loop that won't stitch, loop that doesn't
    actually separate the mesh, seed on the boundary, ...).
    """
    n = len(source.vertices)
    wp = np.asarray(rp.waypoints, dtype=np.int64)
    if wp.min() < 0 or wp.max() >= n:
        raise HTTPException(status_code=400, detail="A waypoint index is out of range")
    if rp.seed is not None and not (0 <= rp.seed < n):
        raise HTTPException(status_code=400, detail="Seed index is out of range")

    try:
        loop = regions.loop_from_waypoints(source.vertices, source.faces, wp.tolist())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Could not stitch boundary loop: {e}")

    adj = regions.vertex_adjacency(source.faces, n)
    labels = regions.label_sides(adj, loop, n)
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels) if c != -1}
    # On coarse meshes a stitched loop can shave off tiny 1-triangle specks;
    # ignore those when reasoning about "sides". A real side is a sizeable
    # connected region.
    floor = max(8, int(0.005 * n))
    significant = [c for c, sz in sizes.items() if sz >= floor]
    if len(significant) < 2:
        raise HTTPException(
            status_code=400,
            detail=("Boundary loop does not separate the mesh into two real "
                    "regions. Make sure the outline is closed and forms a ring "
                    "around the patch you want to edit."),
        )

    # Choose the editable side: the seed's component if given, else the
    # SMALLEST significant side (the patch you drew around). `invert` flips it.
    if rp.seed is not None:
        if labels[rp.seed] == -1:
            raise HTTPException(status_code=400,
                                detail="Seed lies on the boundary loop — pick a vertex to one side")
        chosen = int(labels[rp.seed])
    else:
        chosen = min(significant, key=lambda c: sizes[c])

    if rp.invert:
        interior = np.where((labels != chosen) & (labels != -1))[0].astype(np.int64)
    else:
        interior = np.where(labels == chosen)[0].astype(np.int64)

    frozen = np.setdiff1d(np.arange(n), interior)
    if len(interior) == 0:
        raise HTTPException(status_code=400, detail="Editable side is empty")
    if len(frozen) == 0:
        raise HTTPException(status_code=400, detail="Frozen side is empty")

    soft_w_full = regions.feather_weights(source.vertices, source.faces,
                                          interior, loop, rp.feather_width,
                                          exponent=rp.feather_exponent)
    keep = soft_w_full > 1e-6
    editable_mask = np.zeros(n, dtype=bool)
    editable_mask[interior] = True
    region = RegionSpec(
        frozen_idx=frozen,
        editable_mask=editable_mask,
        soft_idx=interior[keep],
        soft_w=soft_w_full[keep],
    )
    return region, loop, interior


@app.post("/api/sessions/{session_id}/region/preview")
def preview_region(session_id: str, payload: RegionPayload):
    """Cheap (no-solve) preview: stitch the loop and flood-fill the interior so
    the UI can colour editable vs frozen before committing to a wrap."""
    s = _get_session(session_id)
    if not s.source_ref_path:
        raise HTTPException(status_code=400, detail="Upload a source reference first")
    source = meshlib.Mesh.load(s.source_ref_path)
    region, loop, interior = _build_region(source, payload)
    return {
        "loop": loop.astype(int).tolist(),
        "interior": interior.astype(int).tolist(),
        "interior_count": int(len(interior)),
        "frozen_count": int(len(region.frozen_idx)),
        "feather_count": int(len(region.soft_idx)),
        "total": int(len(source.vertices)),
    }


@app.post("/api/sessions/{session_id}/wrap")
def run_wrap(session_id: str, payload: WrapPayload):
    s = _get_session(session_id)
    if not s.source_ref_path or not s.target_ref_path:
        raise HTTPException(status_code=400, detail="Both source_ref and target_ref must be uploaded")
    # NOTE: the >=3 markers check moved into the worker (_wrap_compute) — it
    # needs the loaded meshes to know whether source/target share topology, in
    # which case Wrap/Region need no markers (like same-topology /run). A failed
    # check surfaces as job.state == "error".
    job = _submit_job(s, "wrap", lambda job: _wrap_compute(s, payload, job))
    return _job_response(job)


def _wrap_compute(s: Session, payload: WrapPayload, job: Job) -> dict:
    # Without the closest-point term the system is the same on every
    # iteration (nothing depends on the current vertices), so extra
    # iterations would just re-solve identical equations.
    iterations = 1 if not payload.use_closest_point else payload.iterations

    t_wrap = time.perf_counter()
    cache_hits: List[str] = []
    cache_misses: List[str] = []

    # The ComputeQueue slot is already held by the worker; take only the
    # per-session lock here.
    with s._lock:
        source_ref = meshlib.Mesh.load(s.source_ref_path)
        target_ref = meshlib.Mesh.load(s.target_ref_path)
        # reshape(-1, 2) keeps an empty marker list as shape (0, 2) so the
        # correspondence's markers[:, 0] indexing stays valid with zero markers.
        markers = np.array(s.markers, dtype=np.int64).reshape(-1, 2)

        # Markers are required UNLESS source and target already share topology:
        # then the closest-point term (Wrap) or the frozen-region pins (Region)
        # register the meshes on their own — same rationale as same-topology
        # /run. Fit (no closest-point) still needs markers to anchor each island.
        same_topology = (
            len(source_ref.vertices) == len(target_ref.vertices)
            and source_ref.faces.shape == target_ref.faces.shape
            and bool(np.array_equal(source_ref.faces, target_ref.faces))
        )
        if len(markers) < 3 and not (same_topology and payload.use_closest_point):
            raise HTTPException(
                status_code=400,
                detail="At least 3 marker pairs are required (not needed when "
                       "the source and target already share topology).")

        # Marker-based alignment: bring the target into source space so the
        # closest-point step has overlapping geometry. Skipped if disabled or
        # too few markers. Must run BEFORE hashing/precomputing the target so
        # the setup cache and marker pin positions use the aligned coords.
        align_scale = None
        if payload.align_to_source and len(markers) >= 3:
            src_pts = source_ref.vertices[markers[:, 0]]
            tgt_pts = target_ref.vertices[markers[:, 1]]
            sc, Rm, tm = umeyama(tgt_pts, src_pts)          # target -> source
            target_ref.vertices = apply_similarity(target_ref.vertices, sc, Rm, tm)
            align_scale = float(sc)
            print(f"[wrap] aligned target->source via {len(markers)} markers: scale={sc:.4f}")

        # Regional wrap: build the freeze/feather partition from the loop.
        region_spec = None
        if payload.region is not None:
            region_spec, _loop, _interior = _build_region(source_ref, payload.region)
        elif not payload.use_closest_point:
            # Fit mode (no region): every island still needs a marker.
            _check_marker_coverage(source_ref, markers)

        src_key = _hash_mesh(source_ref)
        tgt_key = _hash_mesh(target_ref)

        # Reuse the same source/target setup caches as /run — they're pure
        # functions of the meshes and don't care which mode triggered them.
        if s.source_setup_key == src_key and s.source_setup is not None:
            cache_hits.append("source_setup")
            src_setup: SourceSetup = s.source_setup
        else:
            cache_misses.append("source_setup")
            src_setup = precompute_source(source_ref)
            s.source_setup_key = src_key
            s.source_setup = src_setup

        if s.target_setup_key == tgt_key and s.target_setup is not None:
            cache_hits.append("target_setup")
            tgt_setup: TargetSetup = s.target_setup
        else:
            cache_misses.append("target_setup")
            tgt_setup = precompute_target(target_ref)
            s.target_setup_key = tgt_key
            s.target_setup = tgt_setup

        # Run correspondence. Wrap only needs the warped source; the triangle
        # mapping is /run-only, so skip its per-triangle pass entirely.
        surf = tgt_setup.surface
        job.set_progress("iterations", 0, iterations)
        warped_source, _ = compute_correspondence_with_setup(
            src_setup, tgt_setup, markers,
            iterations=iterations,
            smoothness=payload.smoothness,
            identity_weight=payload.identity_weight,
            use_closest_point=payload.use_closest_point,
            region=region_spec,
            match_max_angle_deg=payload.match_max_angle_deg,
            match_dist_floor=payload.match_distance_frac * surf.diag,
            compute_mapping=False,
            soft_marker_weight=(payload.soft_markers_weight
                                if payload.soft_markers else 0.0),
            progress_callback=lambda i, total: job.set_progress("iterations", i, total),
        )

        if region_spec is not None:
            movable = region_spec.editable_mask               # frozen stays exact
        else:
            movable = np.ones(len(warped_source.vertices), dtype=bool)

        job.set_progress("projection", 0, 0)
        if payload.use_closest_point and payload.project_result:
            # Final shrink-wrap. A single projection snaps everything the
            # filters trust onto the surface; alternating smooth->re-project
            # (when smoothing is requested) removes residual noise WITHOUT
            # lifting the result off the surface the way a trailing Taubin
            # pass alone would.
            faces3 = warped_source.faces[:, :3]
            proj_w = movable.astype(float)
            if region_spec is not None and len(region_spec.soft_idx) > 0:
                proj_w[region_spec.soft_idx] = 1.0 - region_spec.soft_w
            max_pd = payload.project_distance_frac * surf.diag

            def _project(mesh: meshlib.Mesh):
                vn = get_vertex_normals(mesh.vertices, faces3)
                v2, snapped = project_onto_surface(
                    mesh.vertices, vn, surf, max_pd,
                    max_angle_deg=payload.match_max_angle_deg, weight=proj_w)
                v2, n_flip = damp_folds(mesh.vertices, v2, faces3)
                return meshlib.Mesh(vertices=v2, faces=mesh.faces), int(snapped.sum()), n_flip

            warped_source, n_snap, n_flip = _project(warped_source)
            smooth_cycles = 2 if payload.smooth_result > 0 else 0
            for _ in range(smooth_cycles):
                warped_source = taubin_smooth(warped_source, movable, payload.smooth_result)
                warped_source, n_snap, n_flip = _project(warped_source)
            print(f"[wrap] projection: snapped {n_snap}/{len(warped_source.vertices)} verts  "
                  f"(smooth cycles={smooth_cycles} x {payload.smooth_result} passes, "
                  f"damped folds={n_flip})")
        elif payload.smooth_result > 0:
            # Fit mode (or projection disabled): plain Taubin denoise.
            warped_source = taubin_smooth(warped_source, movable, payload.smooth_result)
            print(f"[wrap] taubin smooth: {payload.smooth_result} passes "
                  f"on {int(movable.sum())} verts")

        # Residual: how far the (movable part of the) result sits from the
        # target surface. Reported to the client so parameter tuning isn't
        # blind. Per-vertex distances (as fractions of the target's bbox
        # diagonal) feed the viewport heatmap; frozen verts get -1 = "not
        # supposed to match the target", rendered as the neutral base color.
        residual = None
        dist_frac = None
        if payload.use_closest_point:
            _, _, r_all, _ = closest_points_on_surface(warped_source.vertices, surf)
            r_dist = r_all[movable]
            residual = {
                "mean": float(r_dist.mean()),
                "p95": float(np.percentile(r_dist, 95)),
                "max": float(r_dist.max()),
                "bbox_diag": float(surf.diag),
            }
            dist_frac = np.where(movable, r_all / surf.diag, -1.0)
            print(f"[wrap] residual: mean={residual['mean']:.4g}  "
                  f"p95={residual['p95']:.4g}  max={residual['max']:.4g}  "
                  f"(diag={surf.diag:.4g})")

        # Persist
        job.set_progress("export", 0, 0)
        wrap_dir = s.workspace / "wrap"
        wrap_dir.mkdir(exist_ok=True)
        wrap_path = wrap_dir / "warped_source.obj"
        warped_source.save_obj(str(wrap_path))

        # Polygon-preserving export: wrap only moves vertices, so the
        # author's original topology (quads / n-gons) and UVs can be
        # restored exactly by rewriting the `v` lines of the uploaded OBJ.
        # Falls back to the triangulated file on any mismatch (e.g. an OBJ
        # with duplicated `v` lines the parser collapsed).
        polygons_preserved = False
        try:
            quad_path = wrap_dir / "warped_source_orig_topology.obj"
            rw_stats = obj_io.rewrite_obj_vertices(
                s.source_ref_path,
                warped_source.to_third_dimension(copy=False).vertices,
                quad_path,
            )
            wrap_path = quad_path
            polygons_preserved = True
            print(f"[wrap] original-topology export: {rw_stats['faces']} faces "
                  f"({rw_stats['quads']} quads, {rw_stats['ngons']} n-gons), "
                  f"uvs={'yes' if rw_stats['has_uvs'] else 'no'}")
        except obj_io.ObjRewriteError as e:
            print(f"[wrap] original-topology export skipped: {e}")

        # Per-vertex residual for the viewport heatmap. Written next to the
        # wrap OBJ (same lifecycle); removed when this wrap didn't compute
        # one (Fit mode) so a stale file can't describe the new result.
        residual_path = wrap_dir / "residual.json"
        if dist_frac is not None:
            residual_path.write_text(json.dumps({
                "residual": residual,
                "distances_frac": np.round(dist_frac, 6).tolist(),
            }), encoding="utf-8")
        else:
            residual_path.unlink(missing_ok=True)

        s.wrap_path = str(wrap_path)
        if region_spec is not None:
            s.wrap_mode = "region"
            s.wrap_region = {
                "waypoints": list(payload.region.waypoints),
                "seed": (int(payload.region.seed) if payload.region.seed is not None else None),
                "invert": bool(payload.region.invert),
                "feather_width": float(payload.region.feather_width),
            }
        else:
            s.wrap_mode = "wrap" if payload.use_closest_point else "fit"
            s.wrap_region = None

    s.save()
    elapsed = time.perf_counter() - t_wrap
    print(f"[wrap] total {elapsed:.2f}s  hits={cache_hits}  misses={cache_misses}  "
          f"verts={len(warped_source.vertices)}")
    return {
        "session": s.to_summary(),
        "vertex_count": int(len(warped_source.vertices)),
        "face_count": int(len(warped_source.faces)),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "elapsed_seconds": elapsed,
        "align_scale": align_scale,
        "residual": residual,
        "polygons_preserved": polygons_preserved,
    }


@app.post("/api/sessions/{session_id}/refit")
def run_refit_endpoint(session_id: str, payload: RefitPayload):
    """Fit a garment / accessory (source_ref) onto a body (target_ref).

    Reuses the wrap result slot: the fitted garment is saved as the session's
    'wrap' result, so the existing preview / download UI shows it unchanged.
    """
    s = _get_session(session_id)
    if not s.source_ref_path or not s.target_ref_path:
        raise HTTPException(status_code=400,
                            detail="Both source_ref (garment) and target_ref (body) must be uploaded")
    if payload.preset not in REFIT_PRESETS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown preset {payload.preset!r}; "
                                   f"expected one of {sorted(REFIT_PRESETS)}")
    if payload.collision is not None and payload.collision not in ("none", "push_out", "hide_body"):
        raise HTTPException(status_code=400, detail="collision must be none|push_out|hide_body")
    job = _submit_job(s, "wrap", lambda job: _refit_compute(s, payload, job))
    return _job_response(job)


@app.post("/api/sessions/{session_id}/refit_field")
def refit_field_endpoint(session_id: str, payload: RefitPayload):
    """Dry-run "preview grip": build ONLY the conform field (no solve) and
    return it, so the user can see what the fit will pull and what it leaves
    free before committing to a solve. Synchronous — it's just signed_offset +
    occlusion raycasts (seconds), so it skips the job queue."""
    s = _get_session(session_id)
    if not s.source_ref_path or not s.target_ref_path:
        raise HTTPException(status_code=400,
                            detail="Both source_ref (garment) and target_ref (body) must be uploaded")
    if payload.preset not in REFIT_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset {payload.preset!r}")
    with s._lock:
        garment = meshlib.Mesh.load(s.source_ref_path)
        body = meshlib.Mesh.load(s.target_ref_path)
        pre = (np.array(payload.pre_transform, dtype=float).reshape(4, 4)
               if payload.pre_transform is not None else None)
        ngv = len(garment.vertices)
        preserve = [v for v in payload.preserve if 0 <= v < ngv]
        preserve = np.array(preserve, dtype=np.int64) if preserve else None
        frozen = [v for v in payload.frozen if 0 <= v < ngv]
        frozen = np.array(frozen, dtype=np.int64) if frozen else None
        rf = build_refit_field(
            garment, body, payload.preset,
            pre_transform=pre, grip_width=payload.grip_width,
            seam_depth=payload.seam_depth, preserve=preserve, frozen=frozen,
            pin_radius=payload.pin_radius, stiffness=payload.stiffness,
            smoothness=payload.smoothness, collision=payload.collision,
            contact_tight=payload.contact_tight, contact_free=payload.contact_free,
            thickness=payload.thickness, auto_polish=payload.auto_polish,
            layer_mode=payload.layer_mode, layer_dry_run=True)
    resp = {
        "grip_frac": np.round(rf.field, 4).tolist(),
        "stats": {
            "contact_verts": int((rf.contact > 0).sum()),
            "occluded_verts": int(rf.occluded.sum()),
            "opposed_verts": int(rf.opposed.sum()),
            "seam_grip_verts": int((rf.field >= 1.0 - 1e-9).sum()),
            "layers": rf.layer_stats,
        },
    }
    # Auto mode: hand the per-vertex layer id back for the overlay + prefill.
    if rf.layer is not None:
        resp["layer"] = rf.layer.astype(int).tolist()
    return resp


def _refit_compute(s: Session, payload: RefitPayload, job: Job) -> dict:
    t0 = time.perf_counter()
    with s._lock:
        garment = meshlib.Mesh.load(s.source_ref_path)     # source = the thing we deform
        body = meshlib.Mesh.load(s.target_ref_path)        # target = the body to fit onto

        pre = (np.array(payload.pre_transform, dtype=float).reshape(4, 4)
               if payload.pre_transform is not None else None)
        # Existing markers act as optional pins (garment vertex -> body vertex).
        markers = np.array(s.markers, dtype=np.int64).reshape(-1, 2)
        markers = markers if len(markers) else None

        # Pin preserve (outer-wall) vertices to RELEASE, dropping any index
        # outside the garment (stale client state).
        ngv = len(garment.vertices)
        preserve = [v for v in payload.preserve if 0 <= v < ngv]
        preserve = np.array(preserve, dtype=np.int64) if preserve else None
        frozen = [v for v in payload.frozen if 0 <= v < ngv]
        frozen = np.array(frozen, dtype=np.int64) if frozen else None

        job.set_progress("iterations", 0, payload.iterations)
        res = run_refit(
            garment, body, payload.preset,
            pre_transform=pre, grip_width=payload.grip_width, offset=payload.offset,
            seam_depth=payload.seam_depth,
            preserve=preserve, frozen=frozen, pin_radius=payload.pin_radius,
            iterations=payload.iterations, markers=markers,
            stiffness=payload.stiffness, smoothness=payload.smoothness,
            collision=payload.collision,
            contact_tight=payload.contact_tight, contact_free=payload.contact_free,
            thickness=payload.thickness, auto_polish=payload.auto_polish,
            growth_steps=payload.growth_steps, layer_mode=payload.layer_mode,
            progress_callback=lambda i, total: job.set_progress("iterations", i, total),
        )
        return _persist_refit_result(s, res, body, job, t0, payload.preset)


def _persist_refit_result(s: Session, res, body, job: Job, t0: float,
                          preset_label: str) -> dict:
    """Save a RefitResult into the session wrap slot (mesh + residual + hide
    mask) and build the response. Shared by /refit and /refit_via_proxy.
    Caller holds s._lock."""
    fitted = meshlib.Mesh(vertices=res.verts, faces=res.faces)

    # Persist into the wrap slot so the preview / download UI just works.
    job.set_progress("export", 0, 0)
    wrap_dir = s.workspace / "wrap"
    wrap_dir.mkdir(exist_ok=True)
    wrap_path = wrap_dir / "warped_source.obj"
    fitted.save_obj(str(wrap_path))
    polygons_preserved = False
    try:
        quad_path = wrap_dir / "warped_source_orig_topology.obj"
        obj_io.rewrite_obj_vertices(s.source_ref_path, res.verts, quad_path)
        wrap_path = quad_path
        polygons_preserved = True
    except obj_io.ObjRewriteError as e:
        print(f"[refit] original-topology export skipped: {e}")

    # Distance of the fitted garment to the body, for the viewport heatmap
    # (here it reads as grip depth: blue = gripping, red = standing off —
    # which for a graft is expected, not an error).
    body_surf = build_surface(body.vertices, body.faces)
    _, _, dist, _ = closest_points_on_surface(res.verts, body_surf)
    residual = {"mean": float(dist.mean()), "p95": float(np.percentile(dist, 95)),
                "max": float(dist.max()), "bbox_diag": float(body_surf.diag)}
    # grip_frac = the conform field the auto-detection built (contact +
    # seam − released layers), 0..1 — shown as a second heatmap so the
    # user can SEE what gripped and what was left free before re-running.
    (wrap_dir / "residual.json").write_text(json.dumps({
        "residual": residual,
        "distances_frac": np.round(dist / body_surf.diag, 6).tolist(),
        "grip_frac": np.round(res.conform_field, 4).tolist(),
    }), encoding="utf-8")

    # Body-hide mask (armor): save the covered body vertex indices as a
    # per-outfit hide list. The body mesh itself is never modified.
    hidden_count = 0
    if res.body_hide_mask is not None:
        hidden_idx = np.nonzero(res.body_hide_mask)[0]
        hidden_count = int(len(hidden_idx))
        (wrap_dir / "body_hide.json").write_text(json.dumps({
            "hidden_vertices": hidden_idx.astype(int).tolist(),
            "body_vertex_count": int(len(body.vertices)),
        }), encoding="utf-8")
    else:
        (wrap_dir / "body_hide.json").unlink(missing_ok=True)

    s.wrap_path = str(wrap_path)
    s.wrap_mode = "refit"
    s.wrap_region = None
    s.save()

    elapsed = time.perf_counter() - t0
    print(f"[refit] preset={preset_label} total {elapsed:.2f}s verts={len(res.verts)} "
          f"collision={res.stats.get('collision')} hidden_body={hidden_count}")
    return {
        "session": s.to_summary(),
        "vertex_count": int(len(res.verts)),
        "face_count": int(len(res.faces)),
        "elapsed_seconds": elapsed,
        "residual": residual,
        "polygons_preserved": polygons_preserved,
        "refit": {**res.stats, "hidden_body_verts": hidden_count},
    }


@app.post("/api/sessions/{session_id}/refit_via_proxy")
def refit_via_proxy_endpoint(session_id: str, payload: RefitPayload):
    """Proxy wardrobe (refit D): wear a garment fitted to a proxy basemesh (P)
    on the body (target_ref), transported through the current wrap result (P',
    P deformed into the body's shape). Then a short cleanup refit.

    Preconditions: a wrap result exists (P' = the wrap slot, same topology as P),
    and source_ref (garment G), target_ref (body B) and proxy_ref (basemesh P)
    are all uploaded. pre_transform / placement pins are ignored — the placement
    comes from the binding.
    """
    s = _get_session(session_id)
    if not s.source_ref_path or not s.target_ref_path:
        raise HTTPException(status_code=400,
                            detail="Both source_ref (garment) and target_ref (body) must be uploaded")
    if not s.proxy_ref_path:
        raise HTTPException(status_code=400,
                            detail="Upload the proxy basemesh (proxy_ref) first")
    if not s.wrap_path or s.wrap_mode not in ("wrap", "region"):
        raise HTTPException(
            status_code=400,
            detail="Need a Wrap result of the proxy basemesh onto the body "
                   "(run Wrap with the basemesh as source and the body as target first)")
    if payload.preset not in REFIT_PRESETS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown preset {payload.preset!r}; "
                                   f"expected one of {sorted(REFIT_PRESETS)}")
    job = _submit_job(s, "wrap", lambda job: _refit_via_proxy_compute(s, payload, job))
    return _job_response(job)


def _refit_via_proxy_compute(s: Session, payload: RefitPayload, job: Job) -> dict:
    from dt_core.wardrobe import (bind_garment, apply_binding, binding_hash,
                                  save_binding, load_binding)
    t0 = time.perf_counter()
    with s._lock:
        garment = meshlib.Mesh.load(s.source_ref_path)     # G — fitted to P
        body = meshlib.Mesh.load(s.target_ref_path)        # B — the body
        proxy = meshlib.Mesh.load(s.proxy_ref_path)        # P — basemesh rest
        wrapped = meshlib.Mesh.load(s.wrap_path)           # P' — P in B's shape

        # P' must share P's topology (it IS P deformed by the wrap).
        if len(wrapped.vertices) != len(proxy.vertices):
            raise HTTPException(
                status_code=400,
                detail=f"Wrap result has {len(wrapped.vertices)} verts but the proxy "
                       f"basemesh has {len(proxy.vertices)} — the wrap must be of THIS "
                       f"basemesh onto the body.")

        # Binding is a pure function of (garment rest, proxy rest): cache it.
        job.set_progress("bind", 0, 0)
        wardrobe_dir = s.workspace / "wardrobe"
        wardrobe_dir.mkdir(exist_ok=True)
        h = binding_hash(garment.vertices, proxy.vertices)
        cache_path = wardrobe_dir / f"bind_{h}.npz"
        if cache_path.exists():
            binding = load_binding(cache_path)
        else:
            proxy_surf = build_surface(proxy.vertices, proxy.faces)
            binding = bind_garment(garment.vertices, proxy_surf)
            save_binding(cache_path, binding)

        # Transport the garment onto the deformed proxy, then a short cleanup.
        placed = apply_binding(binding, wrapped.vertices, proxy.faces)
        markers = np.array(s.markers, dtype=np.int64).reshape(-1, 2)
        markers = markers if len(markers) else None
        ngv = len(garment.vertices)
        frozen = [v for v in payload.frozen if 0 <= v < ngv]
        frozen = np.array(frozen, dtype=np.int64) if frozen else None

        iters = min(payload.iterations, 6)
        job.set_progress("iterations", 0, iters)
        res = run_refit(
            garment, body, payload.preset,
            placed_verts=placed, grip_width=payload.grip_width,
            offset=payload.offset, seam_depth=payload.seam_depth,
            frozen=frozen, iterations=iters, markers=markers,
            stiffness=payload.stiffness, smoothness=payload.smoothness,
            collision=payload.collision,
            contact_tight=payload.contact_tight, contact_free=payload.contact_free,
            thickness=payload.thickness, growth_steps=payload.growth_steps,
            layer_mode=payload.layer_mode,
            progress_callback=lambda i, total: job.set_progress("iterations", i, total),
        )
        return _persist_refit_result(s, res, body, job, t0,
                                     f"{payload.preset} (via proxy)")


@app.get("/api/sessions/{session_id}/wrap.obj")
def download_wrap(session_id: str):
    s = _get_session(session_id)
    if not s.wrap_path:
        raise HTTPException(status_code=404, detail="No wrap result yet")
    src_name = Path(s.source_ref_path).stem if s.source_ref_path else "source"
    tgt_name = Path(s.target_ref_path).stem if s.target_ref_path else "target"
    verb = {"fit": "fitted", "region": "regional", "refit": "refitted"}.get(s.wrap_mode, "wrapped")
    download_name = f"{src_name}_{verb}_to_{tgt_name}.obj"
    return FileResponse(
        s.wrap_path,
        media_type="application/octet-stream",
        filename=download_name,
    )


@app.get("/api/sessions/{session_id}/body_mask")
def get_body_mask(session_id: str):
    """The covered-body-vertex mask for the current refit result.

    Every refit writes `wrap/body_hide.json` (see _refit_finish); this returns
    it verbatim so the preview can hide the skin under the garment and the mask
    editor (item 2) can load it for painting. 404 when there's no refit yet.
    """
    s = _get_session(session_id)
    path = s.workspace / "wrap" / "body_hide.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No body mask for this session")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_body_mask(s: Session, hidden: np.ndarray, body_vertex_count: int) -> dict:
    """Persist a covered-body mask into the session's wrap slot and return the
    JSON payload. `hidden` is any int array; deduped, sorted, range-checked."""
    hidden = np.unique(np.asarray(hidden, dtype=np.int64))
    hidden = hidden[(hidden >= 0) & (hidden < body_vertex_count)]
    payload = {"hidden_vertices": hidden.tolist(),
               "body_vertex_count": int(body_vertex_count)}
    wrap_dir = s.workspace / "wrap"
    wrap_dir.mkdir(exist_ok=True)
    (wrap_dir / "body_hide.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


@app.put("/api/sessions/{session_id}/body_mask")
def put_body_mask(session_id: str, payload: BodyMaskPayload):
    """Replace the covered-body mask (how paint edits persist). Validated:
    ints in [0, body_vertex_count)."""
    s = _get_session(session_id)
    if not s.target_ref_path:
        raise HTTPException(status_code=400, detail="No target (body) mesh in this session")
    body = meshlib.Mesh.load(s.target_ref_path)
    with s._lock:
        return _write_body_mask(s, np.asarray(payload.hidden_vertices), len(body.vertices))


@app.post("/api/sessions/{session_id}/body_mask/dilate")
def dilate_body_mask(session_id: str, payload: BodyMaskDilatePayload):
    """Grow (rings > 0) or shrink (rings < 0) the mask by vertex-adjacency rings
    on the body. The manual "make the mask generous" knob. Returns the new mask."""
    s = _get_session(session_id)
    if not s.target_ref_path:
        raise HTTPException(status_code=400, detail="No target (body) mesh in this session")
    path = s.workspace / "wrap" / "body_hide.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No body mask for this session")
    body = meshlib.Mesh.load(s.target_ref_path)
    n = len(body.vertices)
    cur = json.loads(path.read_text(encoding="utf-8"))
    mask = np.zeros(n, dtype=bool)
    idx = np.asarray(cur.get("hidden_vertices", []), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    mask[idx] = True
    grown = regions.dilate_vertex_mask(body.faces, n, mask, payload.rings)
    with s._lock:
        return _write_body_mask(s, np.nonzero(grown)[0], n)


@app.get("/api/sessions/{session_id}/body_mask.json")
def download_body_mask(session_id: str):
    """Download the covered-body mask as a JSON sidecar (accompanies the OBJ)."""
    s = _get_session(session_id)
    path = s.workspace / "wrap" / "body_hide.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No body mask for this session")
    src_name = Path(s.source_ref_path).stem if s.source_ref_path else "garment"
    return FileResponse(path, media_type="application/json",
                        filename=f"{src_name}_body_mask.json")


@app.put("/api/sessions/{session_id}/refit_state")
def put_refit_state(session_id: str, payload: RefitStatePayload):
    """Persist the refit working state (placement, pins, frozen, preset, knobs)
    with the session so it survives F5 / view switches / restore. Opaque blob;
    the backend only bounds its size and round-trips it in the summary."""
    s = _get_session(session_id)
    blob = payload.refit_state
    if blob is not None:
        if len(json.dumps(blob)) > 2 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="refit_state too large (2 MB max)")
    with s._lock:
        s.refit_state = blob
        s.save()
    return {"ok": True}


# -----------------------------------------------------------------------------
# Download results
# -----------------------------------------------------------------------------

@app.get("/api/sessions/{session_id}/results/{pose_id}.obj")
def download_result(session_id: str, pose_id: str):
    s = _get_session(session_id)
    path = s.results.get(pose_id)
    if not path:
        raise HTTPException(status_code=404, detail="No result for that pose")
    pose = s.poses.get(pose_id)
    download_name = (pose.name.rsplit(".", 1)[0] if pose else pose_id) + "_deformed.obj"
    return FileResponse(path, media_type="application/octet-stream", filename=download_name)


@app.get("/api/sessions/{session_id}/results.fbx")
def download_results_fbx(session_id: str):
    """
    Build a single FBX whose basis is the target_ref mesh and which carries one
    shape key per pose result (named after the pose).
    """
    s = _get_session(session_id)
    if not s.results:
        raise HTTPException(status_code=404, detail="No results yet")
    if not s.target_ref_path:
        raise HTTPException(status_code=400, detail="target_ref not uploaded")
    if not fbx_io.blender_available():
        raise HTTPException(
            status_code=503,
            detail=("FBX download requires a Blender installation. Install Blender "
                    "or set the BLENDER_PATH env var to its executable."),
        )

    def _shape_key_name(pose_name: str) -> str:
        # Drop trailing extension if present so "eyeBlinkLeft.obj" → "eyeBlinkLeft"
        if pose_name.lower().endswith(".obj"):
            return pose_name[:-4]
        return pose_name

    poses_for_build = []
    for pose_id, res_path in s.results.items():
        pose = s.poses.get(pose_id)
        sk_name = _shape_key_name(pose.name) if pose else pose_id
        poses_for_build.append((sk_name, res_path))

    out_fbx = s.workspace / "results.fbx"
    try:
        result = fbx_io.build_fbx_with_shape_keys(
            basis_obj_path=s.target_ref_path,
            poses=poses_for_build,
            out_fbx_path=str(out_fbx),
            mesh_name="DeformedTarget",
        )
    except fbx_io.BlenderError as e:
        raise HTTPException(status_code=500, detail=f"FBX build failed: {e}")

    return FileResponse(
        result.out_fbx,
        media_type="application/octet-stream",
        filename="deformed_target.fbx",
        headers={"X-Added-Shape-Keys": str(len(result.added_shape_keys))},
    )


@app.get("/api/sessions/{session_id}/results.zip")
def download_all_results(session_id: str):
    s = _get_session(session_id)
    if not s.results:
        raise HTTPException(status_code=404, detail="No results yet")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pose_id, path in s.results.items():
            pose = s.poses.get(pose_id)
            arcname = (pose.name.rsplit(".", 1)[0] if pose else pose_id) + "_deformed.obj"
            zf.write(path, arcname=arcname)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="results.zip"'},
    )


@app.get("/api/health")
def health():
    """Cheap liveness probe (used by the Docker healthcheck and the launch scripts)."""
    return {"name": "Chameleon Warp", "status": "ok"}


# -----------------------------------------------------------------------------
# Static serving — one process serves the built SPA AND /api, so there's no
# CORS and only one port to open. Enabled only when the frontend has been built
# (`frontend/dist` exists, or DT_FRONTEND_DIST points at a build); in dev the
# Vite server serves the UI and proxies /api here, so this stays dormant.
#
# Registered LAST so the catch-all can't shadow any /api route (FastAPI matches
# in registration order).
# -----------------------------------------------------------------------------

FRONTEND_DIST = Path(os.environ.get("DT_FRONTEND_DIST")
                     or BACKEND_DIR.parent / "frontend" / "dist").resolve()

if FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    _assets_dir = FRONTEND_DIST / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        # An unknown /api/* path should 404 as JSON, not silently return the
        # SPA shell (that would mask client bugs and break error handling).
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        # Serve a real static file if it exists (favicon, version.json, …),
        # otherwise fall back to index.html so client-side routes resolve.
        if full_path:
            candidate = (FRONTEND_DIST / full_path).resolve()
            # Guard against path traversal escaping the dist dir.
            if candidate.is_relative_to(FRONTEND_DIST) and candidate.is_file():
                return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    print(f"[static] serving built frontend from {FRONTEND_DIST}")
else:
    @app.get("/")
    def root():
        return JSONResponse({
            "name": "Chameleon Warp",
            "status": "ok",
            "note": "frontend/dist not built; run `npm run build` in frontend/ (or use the Vite dev server)",
        })
