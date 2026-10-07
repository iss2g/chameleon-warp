"""
FBX I/O via Blender as a subprocess.

Why subprocess: Python 3.9 has no good pip-installable FBX library that handles
shape keys (the official Autodesk FBX SDK is not on PyPI; pyfbx & pyassimp
have weak morph-target support). Blender ships a robust FBX importer/exporter
and we just spawn it in `--background` mode for each conversion.

Public API:
  - blender_available() -> bool
  - find_blender() -> Path | None
  - extract_shape_keys(fbx_path, out_dir) -> ExtractResult
  - build_fbx_with_shape_keys(basis_obj_path, poses, out_fbx_path) -> BuildResult
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BACKEND_DIR / "blender_scripts"
EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_shape_keys.py"
BUILD_SCRIPT = SCRIPTS_DIR / "build_fbx.py"

# How long we wait for Blender to do its thing. Extraction of an ARKit-size
# FBX (~50 shape keys × ~24k verts) finishes in ~30s on a modern machine —
# 180s is a generous cap that still surfaces a hung process.
BLENDER_TIMEOUT_SECONDS = 180


def find_blender() -> Optional[Path]:
    """Locate the Blender executable, returning None if not found.

    Order:
      1. BLENDER_PATH env var (full path to blender.exe / blender)
      2. PATH lookup via `shutil.which`
      3. Standard Windows install locations under Program Files
      4. macOS / Linux common spots
    """
    # 1. env override
    env = os.environ.get("BLENDER_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    # 2. PATH
    found = shutil.which("blender")
    if found:
        return Path(found)

    # 3. Windows program files
    for base in (
        Path("C:/Program Files/Blender Foundation"),
        Path("C:/Program Files (x86)/Blender Foundation"),
    ):
        if base.is_dir():
            candidates = sorted(base.glob("Blender */blender.exe"))
            if candidates:
                return candidates[-1]  # highest version

    # 4. macOS / Linux
    for guess in (
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
    ):
        if guess.is_file():
            return guess

    return None


def blender_available() -> bool:
    return find_blender() is not None


# -----------------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------------

@dataclass
class PoseExtract:
    name: str           # original shape-key name (preserved for display)
    filename: str       # sanitized .obj filename
    path: str           # absolute path to the OBJ
    size: int           # bytes


@dataclass
class ExtractResult:
    mesh_name: str
    vertex_count: int
    polygon_count: int
    basis_path: str
    poses: List[PoseExtract]


@dataclass
class BuildResult:
    out_fbx: str
    added_shape_keys: List[str]
    skipped: List[Dict]        # [{name, reason}]
    vertex_count: int


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------

class BlenderError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Internal: run blender, parse the DT_RESULT line
# -----------------------------------------------------------------------------

def _run_blender(script: Path, script_args: List[str]) -> dict:
    blender = find_blender()
    if blender is None:
        raise BlenderError(
            "Blender executable not found. "
            "Install Blender, OR set BLENDER_PATH=<path/to/blender.exe>."
        )

    cmd = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python", str(script),
        "--",
    ] + script_args

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BLENDER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise BlenderError(f"Blender timed out after {BLENDER_TIMEOUT_SECONDS}s") from e

    # Hunt for the manifest line in stdout (may be surrounded by Blender chatter)
    manifest: Optional[dict] = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("DT_RESULT:"):
            try:
                manifest = json.loads(line[len("DT_RESULT:"):])
            except json.JSONDecodeError:
                continue
            break

    if manifest is None:
        raise BlenderError(
            f"Blender subprocess produced no DT_RESULT line (rc={proc.returncode}).\n"
            f"stdout tail:\n{proc.stdout[-1200:]}\n"
            f"stderr tail:\n{proc.stderr[-1200:]}"
        )

    if not manifest.get("ok"):
        raise BlenderError(f"Blender script reported error: {manifest.get('error')}")

    return manifest


# -----------------------------------------------------------------------------
# Public: extract shape keys from an FBX
# -----------------------------------------------------------------------------

def extract_shape_keys(fbx_path: str, out_dir: str) -> ExtractResult:
    """
    Run Blender to import `fbx_path` and write a `basis.obj` + one OBJ per
    non-basis shape key into `out_dir`. All meshes are written in world space.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = _run_blender(EXTRACT_SCRIPT, [fbx_path, out_dir])
    return ExtractResult(
        mesh_name=manifest["mesh_name"],
        vertex_count=int(manifest["vertex_count"]),
        polygon_count=int(manifest["polygon_count"]),
        basis_path=manifest["basis_path"],
        poses=[
            PoseExtract(
                name=p["name"], filename=p["filename"],
                path=p["path"], size=int(p["size"]),
            )
            for p in manifest["poses"]
        ],
    )


# -----------------------------------------------------------------------------
# Public: build an FBX from a basis OBJ + per-pose OBJs
# -----------------------------------------------------------------------------

def build_fbx_with_shape_keys(
        basis_obj_path: str,
        poses: List[Tuple[str, str]],   # [(shape_key_name, obj_path), ...]
        out_fbx_path: str,
        mesh_name: str = "DeformedTarget",
) -> BuildResult:
    """
    Run Blender to create an FBX whose mesh is `basis_obj_path` and which has
    one shape key per entry in `poses` (named `shape_key_name`, vertex positions
    taken from the matching `obj_path`).
    """
    cfg = {
        "basis_path": basis_obj_path,
        "poses": [{"name": name, "obj_path": path} for (name, path) in poses],
        "out_fbx": out_fbx_path,
        "mesh_name": mesh_name,
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="dt_build_fbx_",
        delete=False, encoding="utf-8",
    ) as fp:
        json.dump(cfg, fp)
        cfg_path = fp.name

    try:
        manifest = _run_blender(BUILD_SCRIPT, [cfg_path])
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass

    return BuildResult(
        out_fbx=manifest["out_fbx"],
        added_shape_keys=list(manifest["added_shape_keys"]),
        skipped=list(manifest["skipped"]),
        vertex_count=int(manifest["vertex_count"]),
    )
