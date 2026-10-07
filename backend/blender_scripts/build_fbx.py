"""
Blender script: take a basis OBJ + a list of per-shape-key OBJs, build a single
FBX file where each pose OBJ becomes a shape key on the basis mesh.

Invocation:
  blender --background --python build_fbx.py -- <config.json>

config.json:
{
  "basis_path": "C:/.../basis.obj",
  "poses": [
    {"name": "eyeSquintLeft",  "obj_path": "C:/.../eyeSquintLeft.obj"},
    {"name": "mouthSmileLeft", "obj_path": "C:/.../mouthSmileLeft.obj"},
    ...
  ],
  "out_fbx": "C:/.../result.fbx",
  "mesh_name": "DeformedTarget"   // optional
}

The basis OBJ is loaded once and becomes the FBX's mesh. Each pose OBJ is
then loaded, its vertex positions copied into a new shape key on the basis,
and the temporary pose object discarded.

The pose OBJ MUST have the same vertex count and order as the basis OBJ —
which is guaranteed in our pipeline because all results come from running
the same Transformation on the same target topology.
"""
import json
import os
import sys

import bpy  # type: ignore


def fail(msg: str) -> None:
    print("DT_RESULT:" + json.dumps({"ok": False, "error": msg}), flush=True)
    sys.exit(1)


def _import_obj_get_object(path: str):
    """Import an OBJ via wm.obj_import and return the resulting object.

    Our pipeline writes OBJs in **Y-up / forward -Z** (the OBJ standard, what
    three.js and the deformation-transfer step both expect). Tell Blender so
    explicitly — otherwise the importer applies its own axis guess and the
    head ends up sideways when we re-export to FBX.
    """
    if not os.path.isfile(path):
        fail(f"OBJ not found: {path}")
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(
        filepath=path,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    )
    after = set(bpy.context.scene.objects)
    new_objs = list(after - before)
    if not new_objs:
        fail(f"OBJ import produced no objects: {path}")
    return new_objs[0]


def main() -> None:
    if "--" not in sys.argv:
        fail("missing '--' separator before script args")
    args = sys.argv[sys.argv.index("--") + 1:]
    if not args:
        fail("need <config.json>")
    config_path = args[0]
    if not os.path.isfile(config_path):
        fail(f"config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    basis_path: str = cfg["basis_path"]
    poses: list = cfg.get("poses", [])
    out_fbx: str = cfg["out_fbx"]
    mesh_name: str = cfg.get("mesh_name", "DeformedTarget")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the basis mesh
    basis_obj = _import_obj_get_object(basis_path)
    basis_obj.name = mesh_name
    basis_obj.data.name = mesh_name + "_mesh"

    # Add the basis shape key
    basis_obj.shape_key_add(name="Basis", from_mix=False)

    basis_n = len(basis_obj.data.vertices)

    skipped = []
    added = []

    for pose in poses:
        name: str = pose["name"]
        obj_path: str = pose["obj_path"]

        pose_obj = _import_obj_get_object(obj_path)
        pose_n = len(pose_obj.data.vertices)
        if pose_n != basis_n:
            skipped.append({
                "name": name,
                "reason": f"vertex count mismatch (pose {pose_n} vs basis {basis_n})",
            })
            bpy.data.objects.remove(pose_obj, do_unlink=True)
            continue

        # Add new shape key on basis
        sk = basis_obj.shape_key_add(name=name, from_mix=False)
        # Copy pose vertex positions into the shape key
        for i, v in enumerate(pose_obj.data.vertices):
            sk.data[i].co = v.co

        bpy.data.objects.remove(pose_obj, do_unlink=True)
        added.append(name)

    # Make sure basis is selected and active for export
    bpy.ops.object.select_all(action="DESELECT")
    basis_obj.select_set(True)
    bpy.context.view_layer.objects.active = basis_obj

    # Export FBX with shape keys
    bpy.ops.export_scene.fbx(
        filepath=out_fbx,
        use_selection=True,
        object_types={"MESH"},
        add_leaf_bones=False,
        bake_anim=False,
        # Shape-key (morph) export
        use_armature_deform_only=False,
        # Axis: keep Blender's default Z-up; FBX consumers (Maya/Unity/Unreal)
        # do their own conversion on import.
    )

    print("DT_RESULT:" + json.dumps({
        "ok": True,
        "out_fbx": out_fbx,
        "added_shape_keys": added,
        "skipped": skipped,
        "vertex_count": basis_n,
    }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        fail(f"unexpected error: {e}")
