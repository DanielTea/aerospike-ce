"""
Build a print-ready file for the complete engine.

    python print_ready.py --spec ../spec/regen.json --out ../docs/print

"Printable" is a claim, not a file extension, so this checks it rather than
asserting it. Every part written here is:

- watertight, with no boundary and no non-manifold edges
- a single connected solid, not a shell that happens to look like one
- free of the defects a slicer sees and a topologist does not: zero-area
  triangles, duplicated faces, a patch wound inside out, a pinch point
- within tolerance of the distance field it came from, by volume
- meshed fine enough to resolve its own narrowest feature

and it is refused if any of that fails, because a slicer will happily accept a
broken mesh and print something that is not the part.

Resolution
----------
Set from the geometry, not from taste. Marching cubes needs about three samples
across a feature, and the narrowest thing in this engine is a 0.4 mm cooling
channel, so the voxel wants to be around 0.13 mm. That is not negotiable
downwards without losing the channels, which is the whole reason the part is
interesting.

Decimation
----------
Marching cubes tessellates smooth ground at voxel resolution, which is most of
the triangles and none of the information. Quadric collapse removes it -- but it
also reads a half-millimetre channel as noise against a smooth wall and closes
it, so every step is checked against the topology and the volume, and the
decimation stops at the last ratio that survived rather than the one that was
asked for.

Format
------
3MF. STL is a triangle soup with no units in it; every slicer guesses
millimetres and is usually right, which is not the same as being told.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from build_plan import build_plan, narrowest_feature
from engine_design import design_engine
from manifold_ref import design_manifolds, geometry_features
from mesh_export import manifold_report, mesh_volume, slicing_report, write_3mf
from mesh_solid import (
    _weld,
    build_mesh_streaming,
    centrebody_channels,
    cowl_channels,
    decimate,
    feed_ports,
)
from shell_ref import shell_features


def surfaces(verts: np.ndarray, faces: np.ndarray):
    """
    The mesh's closed boundary surfaces, with the signed volume of each.

    Not the same as counting solids. A part with sealed internal cavities has
    one outward-facing surface and one inward-facing surface per cavity, and the
    inward ones carry negative volume. Requiring a single surface would reject
    every hollow part ever printed; what actually matters is whether those
    cavities have a way out.
    """
    m = len(faces)
    rows = np.empty(3 * m, dtype=np.int32)
    cols = np.empty(3 * m, dtype=np.int32)
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        rows[k * m:(k + 1) * m] = faces[:, i]
        cols[k * m:(k + 1) * m] = faces[:, j]
    # int32 indices and int8 weights. At print resolution this graph has 72
    # million entries in it, and the difference between that and the default
    # int64/float64 is three gigabytes on a machine that has already been
    # killed once for wanting them.
    g = coo_matrix((np.ones(3 * m, dtype=np.int8), (rows, cols)),
                   shape=(len(verts), len(verts)))
    del rows, cols
    _, label = connected_components(g, directed=False)
    del g
    per_face = label[faces[:, 0]]
    marks = np.unique(per_face)
    if len(marks) == 1:
        # The ordinary case, and worth not gathering a copy of every triangle
        # in the part to discover it.
        return [(m, mesh_volume(verts, faces))]
    out = []
    for k in marks:
        sub = faces[per_face == k]
        if len(sub):
            out.append((len(sub), mesh_volume(verts, sub)))
    return out


def loose_pieces(verts: np.ndarray, faces: np.ndarray) -> list:
    """
    Solids that are not the part, by their positive volume.

    A part is one solid plus however many cavity surfaces it has. A *second*
    positive-volume surface is a piece of metal connected to nothing: it falls
    out of the powder at depowdering, or worse, does not, and rattles around
    inside a cooling jacket.

    This is the check that was missing. Watertightness does not see it -- both
    pieces are closed -- and neither does the sealed-void test, because the
    fragment is solid rather than hollow. A 27 cm3 ring of copper shipped
    inside the cowl of a print file that passed every gate it was given.
    """
    vols = sorted((v for _, v in surfaces(verts, faces) if v > 0.0), reverse=True)
    return vols[1:]


def sealed_voids(verts: np.ndarray, faces: np.ndarray) -> list:
    """
    Enclosed cavities with no way out, by their negative signed volume.

    A sealed void in a powder-bed part stays full of powder for ever. It adds
    mass nobody accounted for, it can shake loose later, and in a cooling
    jacket or a manifold it blocks the thing the cavity exists for. This is the
    check that matters for printability, and it is the one that catches an
    injector orifice that has stopped reaching the plenum it feeds.
    """
    return [v for _, v in surfaces(verts, faces) if v < 0.0]


def check(verts, faces, label: str, deep: bool = False) -> dict:
    rep = manifold_report(verts, faces)
    rep.update(slicing_report(verts, faces, deep=deep))
    surf = surfaces(verts, faces)
    rep["surfaces"] = len(surf)
    rep["sealed"] = [v for _, v in surf if v < 0.0]
    rep["loose"] = sorted((v for _, v in surf if v > 0.0), reverse=True)[1:]
    rep["genus"] = (2 * len(surf) - rep["euler"]) // 2
    rep["volume_mm3"] = mesh_volume(verts, faces)
    rep["label"] = label
    return rep


def reduce_safely(verts, faces, target_ratio: float, tol_volume: float = 0.02):
    """
    Decimate as far as topology and volume survive, and no further.

    Returns the last mesh that passed. Asking for a ratio and taking whatever
    comes back is how a cooling channel gets quietly closed: the result is still
    watertight, still looks like an engine, and no longer has the feature the
    part exists for.
    """
    def repair(v, f):
        """
        Weld coincident vertices and drop what is left with no area.

        A quadric collapse can slide three vertices into a line. The face it
        leaves has no normal, which is the one property a slicer reads and none
        of the topology checks do -- and refusing the step outright costs the
        whole decimation: on the centrebody every rung produces one, so the
        part shipped all 16 million of its triangles. Welding merges the
        vertices the collapse made coincident, and the faces that then have
        two identical corners come out without opening the surface.
        """
        v2, f2 = _weld(v, f)
        a, b, c = v2[f2[:, 0]], v2[f2[:, 1]], v2[f2[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        return v2, f2[area > 0.0]

    base = check(verts, faces, "base")
    best = (verts, faces, base)
    # Mildest first, stopping at the target. Walking the other way -- skipping
    # every ratio gentler than the target and opening with the target itself --
    # means a part that cannot survive an eight-fold collapse gets no
    # decimation at all, when it would have taken a halving without complaint.
    # Finely stepped near the top, because that is where the channelled
    # parts stop. The centrebody survives 0.50 and not 0.30; with only
    # those two rungs the 0.40 that it would also have taken is never
    # tried, and a fifth of the file is carried for nothing.
    # Starts at 0.85, not 0.5. The head and the cowl both carry 1.2 mm
    # passages now, and halving the triangle count closes them -- so with
    # 0.5 as the mildest rung the guard correctly refused, and both parts
    # shipped every one of their triangles. A ladder is only as good as
    # its gentlest step.
    for ratio in (0.85, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25,
                  0.2, 0.15, 0.12, 0.08, 0.05):
        if ratio < target_ratio:
            continue
        v2, f2 = repair(*decimate(verts, faces, ratio))
        rep = check(v2, f2, f"keep {ratio:.2f}")
        if not rep["watertight"] or rep["surfaces"] != base["surfaces"]:
            break
        # A quadric collapse can slide three vertices into a line without
        # breaking a single edge pairing, so this has to be asked separately.
        # The field carries no zero-area triangles once it is meshed off the
        # bias; the decimator puts them back, and every other check here says
        # the result is fine. export_cooled applies the same criterion.
        # After the repair this should be zero; if it is not, the collapse did
        # something a weld cannot undo and the step is not safe to take.
        if rep["degenerate_faces"] > base["degenerate_faces"]:
            break
        # A collapse can also duplicate a face onto its neighbour or flip one
        # against it. Both keep every edge in two triangles, so the watertight
        # test above passes them, and both are a surface a slicer reads the
        # wrong way round. The fan count is left to the deep check on the
        # finished part: a new pinch moves the genus, which is the next line.
        if rep["duplicate_faces"] > base["duplicate_faces"]:
            break
        if rep["inconsistent_edges"] > base["inconsistent_edges"]:
            break
        if rep["genus"] != base["genus"]:
            break
        if abs(rep["volume_mm3"] - base["volume_mm3"]) > tol_volume * abs(base["volume_mm3"]):
            break
        best = (v2, f2, rep)
    return best


def features_for(design, part):
    a = design.assembly
    gf = geometry_features(design, design_manifolds(design))
    shell, _, _ = shell_features(design)
    ch, pt, hl = [], [], []
    if part == "cowl" and design.circuits.get("cowl"):
        ch = [cowl_channels(a, design.circuits["cowl"].channel)]
        pt = [feed_ports(a, ch[0])]
    if part == "centrebody" and design.circuits.get("centrebody"):
        ch = [centrebody_channels(a, design.circuits["centrebody"].channel)]
        pt = [feed_ports(a, ch[0])]
    # The injector orifices come from geometry_features, which knows where the
    # plenums ended up. Generating them here as well would put a second,
    # differently-placed set of orifices in the same disc.
    hl = list(gf.get(part, {}).get("holes", []))
    pt = list(pt) + list(gf.get(part, {}).get("ports", []))
    return dict(channels=ch, ports=pt, holes=hl,
                bosses=gf.get(part, {}).get("bosses"),
                lugs=gf.get(part, {}).get("lugs"),
                plenums=gf.get(part, {}).get("plenums"),
                ribs=shell.get(part, {}).get("ribs"),
                legs=shell.get(part, {}).get("legs"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    ap.add_argument("--out", default="../docs/print")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="override; default is the narrowest feature over three")
    ap.add_argument("--keep", type=float, default=0.12,
                    help="decimation target; the run stops earlier if topology moves")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    design = design_engine(spec)
    plan = build_plan(spec, design)
    narrow = narrowest_feature(plan)
    voxel = args.voxel if args.voxel > 0 else round(narrow / 3.0, 3)

    os.makedirs(args.out, exist_ok=True)
    print(f"{spec.get('name')}: narrowest feature {narrow:.2f} mm -> "
          f"meshing at {voxel:.3f} mm")

    parts, reports = {}, []
    for part in design.assembly.profiles:
        t0 = time.time()
        v, f = build_mesh_streaming(design.assembly.profiles[part], voxel_mm=voxel,
                                    **features_for(design, part))
        raw = len(f)
        v, f, rep = reduce_safely(v, f, args.keep)
        # The ladder runs on the cheap checks; the part that actually gets
        # written is asked the expensive question once, here.
        rep = check(v, f, rep["label"], deep=True)
        parts[part] = (v, f)
        reports.append((part, raw, rep, time.time() - t0))
        print(f"  {part:11s} {time.time() - t0:6.0f}s  {raw:9d} -> {len(f):8d} tris  "
              f"{rep['label']:10s} watertight {str(rep['watertight']):5s} "
              f"genus {rep['genus']:5d}  sealed {len(rep['sealed'])}  "
              f"loose {len(rep['loose'])}  flat {rep['degenerate_faces']}  "
              f"pinch {rep['nonmanifold_vertices']}  "
              f"{rep['volume_mm3'] / 1000:8.2f} cm3", flush=True)

    bad = []
    for part, _, r, _ in reports:
        if not r["watertight"]:
            bad.append(f"{part} is not watertight ({r['boundary_edges']} boundary edges)")
        if r["sealed"]:
            trapped = sum(abs(v) for v in r["sealed"]) / 1000.0
            bad.append(f"{part} has {len(r['sealed'])} sealed void(s) holding "
                       f"{trapped:.1f} cm3 of powder with no way out")
        if r["degenerate_faces"]:
            # Invisible to everything else here. A zero-area triangle is a
            # perfectly ordinary face by index -- two neighbours per edge, so
            # watertightness passes, genus passes, volume passes. A slicer has
            # no normal to offset and no side to be inside of, and a plane of
            # them is what Cura means by missing or extraneous surfaces.
            bad.append(f"{part} has {r['degenerate_faces']} zero-area "
                       f"triangle(s); a slicer will refuse it")
        if r["loose"]:
            adrift = sum(r["loose"]) / 1000.0
            bad.append(f"{part} is in {len(r['loose']) + 1} pieces: "
                       f"{adrift:.1f} cm3 of metal is attached to nothing")
        if r["nonmanifold_vertices"]:
            # The surface touching itself at a point. Every edge is still in
            # two triangles so watertightness passes, and the slicer has two
            # answers for what is inside there.
            bad.append(f"{part} pinches at {r['nonmanifold_vertices']} "
                       f"vertex/vertices; the surface meets itself")
        if r["duplicate_faces"]:
            bad.append(f"{part} has {r['duplicate_faces']} duplicated triangle(s)")
        if r["inconsistent_edges"]:
            bad.append(f"{part} has {r['inconsistent_edges']} edge(s) whose two "
                       f"triangles are wound the same way; a patch is inside out")
        if not r["outward"]:
            bad.append(f"{part} encloses negative volume: the solid is inside out")
        if not r["finite"] or not r["indices_in_range"]:
            bad.append(f"{part} has non-finite coordinates or a triangle "
                       f"referencing a vertex that is not there")
    if bad:
        raise SystemExit("refusing to write a print file:\n  " + "\n  ".join(bad))

    path = os.path.join(args.out, f"{spec.get('name', 'engine')}.3mf")
    write_3mf(path, parts)
    total = sum(len(f) for _, f in parts.values())
    print(f"\nwrote {path}  {os.path.getsize(path) / 1e6:.1f} MB, "
          f"{total:,} triangles, {len(parts)} solids")
    print(f"equivalent binary STL would be {(84 + total * 50) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
