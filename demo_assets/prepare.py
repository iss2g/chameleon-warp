"""
Reproducible prep for the bundled demo assets.

Every derived file in this folder is produced by a command in here, so the chain
from the upstream download to what we ship stays auditable (which matters: the
licenses below require us to be able to say exactly what we changed).

Usage:
    python prepare.py fetch     # re-download the upstream originals (~19 MB)
    python prepare.py ict       # ict-facekit/*.obj  -> ict-facekit/skin/*.obj
    python prepare.py body      # makehuman/base.obj -> makehuman/body.obj
    python prepare.py head      # makehuman/body.obj -> makehuman/head.obj
    python prepare.py all       # fetch + all of the above

The upstream originals are NOT committed (see .gitignore) — only the trimmed
results we actually ship. `fetch` brings them back when a step needs rerunning.

No dependencies — plain text OBJ surgery, so it runs with any Python 3.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

ICT_RAW = "https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master"
MH_RAW = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master"
MHA_RAW = "https://raw.githubusercontent.com/makehumancommunity/makehuman-assets/master"

DOWNLOADS: list[tuple[str, str]] = [
    # (url, path relative to this folder)
    *[(f"{ICT_RAW}/FaceXModel/{n}", f"ict-facekit/{n}") for n in (
        "generic_neutral_mesh.obj", "jawOpen.obj", "mouthSmile_L.obj",
        "mouthSmile_R.obj", "eyeBlink_L.obj", "eyeBlink_R.obj",
        "vertex_indices.json",
    )],
    (f"{ICT_RAW}/LICENSE", "ict-facekit/LICENSE-ICT-FaceKit.txt"),
    (f"{MH_RAW}/makehuman/data/3dobjs/base.obj", "makehuman/base.obj"),
    (f"{MHA_RAW}/base/clothes/male_casualsuit01/male_casualsuit01.obj",
     "makehuman/male_casualsuit01.obj"),
    (f"{MHA_RAW}/LICENSE.txt", "makehuman/LICENSE-MakeHuman-assets.txt"),
]


def fetch() -> None:
    for url, rel in DOWNLOADS:
        dst = HERE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            dst.write_bytes(r.read())
        print(f"{rel}: {dst.stat().st_size:,} bytes")


def keep_groups(
    src: Path, dst: Path, keep_prefixes: tuple[str, ...], tag: str = "g "
) -> None:
    """Copy `src` to `dst` keeping only faces whose current `tag` value starts
    with one of `keep_prefixes`, then drop the v/vt/vn records no surviving face
    references and renumber accordingly.

    `tag` is the line prefix that partitions the file: "g " for OBJ groups
    (MakeHuman) or "usemtl " for material runs (ICT-FaceKit splits its parts
    that way — face / back head / teeth / eyeballs …).

    Face lines are rewritten (indices shift), everything else is dropped — these
    are demo assets, not a general OBJ round-trip.

    Applying the SAME filter to a set of shape siblings keeps them
    vertex-compatible, which is what the pose OBJs need.
    """
    verts: list[str] = []
    uvs: list[str] = []
    norms: list[str] = []
    faces: list[list[tuple[int, int, int]]] = []   # 1-based, 0 = absent

    current_kept = False
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            verts.append(line)
        elif line.startswith("vt "):
            uvs.append(line)
        elif line.startswith("vn "):
            norms.append(line)
        elif line.startswith(tag):
            name = line[len(tag):].strip()
            current_kept = name.startswith(keep_prefixes)
        elif line.startswith("f ") and current_kept:
            corners = []
            for tok in line.split()[1:]:
                parts = (tok.split("/") + ["", ""])[:3]
                v = int(parts[0])
                vt = int(parts[1]) if parts[1] else 0
                vn = int(parts[2]) if parts[2] else 0
                corners.append((v, vt, vn))
            faces.append(corners)

    used_v: dict[int, int] = {}
    used_vt: dict[int, int] = {}
    used_vn: dict[int, int] = {}
    for corners in faces:
        for v, vt, vn in corners:
            used_v.setdefault(v, 0)
            if vt:
                used_vt.setdefault(vt, 0)
            if vn:
                used_vn.setdefault(vn, 0)
    # Renumber in the original order so the result keeps the source's vertex
    # ordering — vertex order is what a wrap result is judged on downstream.
    for new, old in enumerate(sorted(used_v), start=1):
        used_v[old] = new
    for new, old in enumerate(sorted(used_vt), start=1):
        used_vt[old] = new
    for new, old in enumerate(sorted(used_vn), start=1):
        used_vn[old] = new

    out: list[str] = [f"# derived from {src.name} by demo_assets/prepare.py",
                      f"# kept {tag.strip()}: {', '.join(keep_prefixes)}"]
    out += [verts[i - 1] for i in sorted(used_v)]
    out += [uvs[i - 1] for i in sorted(used_vt)]
    out += [norms[i - 1] for i in sorted(used_vn)]
    for corners in faces:
        toks = []
        for v, vt, vn in corners:
            t = str(used_v[v])
            if vt or vn:
                t += "/" + (str(used_vt[vt]) if vt else "")
            if vn:
                t += "/" + str(used_vn[vn])
            toks.append(t)
        out.append("f " + " ".join(toks))
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{dst.name}: {len(used_v)} verts, {len(faces)} faces")


def cut_above(src: Path, dst: Path, axis: int, threshold: float) -> None:
    """Keep only the faces whose every corner sits above `threshold` on `axis`
    (0=x, 1=y, 2=z). Used to lift a head off a full body."""
    verts: list[str] = []
    coords: list[tuple[float, float, float]] = []
    uvs: list[str] = []
    faces: list[list[tuple[int, int]]] = []

    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            verts.append(line)
            x, y, z = (float(t) for t in line.split()[1:4])
            coords.append((x, y, z))
        elif line.startswith("vt "):
            uvs.append(line)
        elif line.startswith("f "):
            corners = []
            for tok in line.split()[1:]:
                parts = (tok.split("/") + [""])[:2]
                corners.append((int(parts[0]), int(parts[1]) if parts[1] else 0))
            if all(coords[v - 1][axis] >= threshold for v, _ in corners):
                faces.append(corners)

    used_v: dict[int, int] = {}
    used_vt: dict[int, int] = {}
    for corners in faces:
        for v, vt in corners:
            used_v.setdefault(v, 0)
            if vt:
                used_vt.setdefault(vt, 0)
    for new, old in enumerate(sorted(used_v), start=1):
        used_v[old] = new
    for new, old in enumerate(sorted(used_vt), start=1):
        used_vt[old] = new

    out = [f"# derived from {src.name} by demo_assets/prepare.py",
           f"# cut: axis {axis} >= {threshold}"]
    out += [verts[i - 1] for i in sorted(used_v)]
    out += [uvs[i - 1] for i in sorted(used_vt)]
    for corners in faces:
        toks = []
        for v, vt in corners:
            toks.append(f"{used_v[v]}/{used_vt[vt]}" if vt else str(used_v[v]))
        out.append("f " + " ".join(toks))
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{dst.name}: {len(used_v)} verts, {len(faces)} faces")


def ict_skin() -> None:
    # Keep the skin only. The full model also carries teeth, tongue, gums,
    # eyeballs, lacrimal fluid and lashes as separate material runs — they
    # float around unshaded in the viewport and double the file size.
    # The SAME filter runs over the neutral and every expression, so they stay
    # vertex-compatible (a pose must match the reference exactly).
    ict = HERE / "ict-facekit"
    out = ict / "skin"
    out.mkdir(exist_ok=True)
    for obj in sorted(ict.glob("*.obj")):
        keep_groups(obj, out / obj.name, ("M_Face", "M_BackHead"), tag="usemtl ")


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    mh = HERE / "makehuman"
    if what == "fetch":
        fetch()
    elif what == "all":
        fetch()
        ict_skin()
        keep_groups(mh / "base.obj", mh / "body.obj", ("body",))
        cut_above(mh / "body.obj", mh / "head.obj", axis=1, threshold=6.0)
    elif what == "body":
        # MakeHuman ships helper geometry (hair/tights/teeth/eyelash proxies)
        # inside the same OBJ; only the "body" group is the actual skin.
        keep_groups(mh / "base.obj", mh / "body.obj", ("body",))
    elif what == "head":
        # y is up in the MakeHuman base (body spans -8.17..8.49); 6.0 lands in
        # the neck, so the head keeps a collar of neck to wrap against.
        cut_above(mh / "body.obj", mh / "head.obj", axis=1, threshold=6.0)
    elif what == "ict":
        ict_skin()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
