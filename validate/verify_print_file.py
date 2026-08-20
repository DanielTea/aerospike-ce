"""
Check a written print file by reading it back.

    python verify_print_file.py ../docs/print/regen-spike-75.3mf

`print_ready.py` checks the meshes it holds in memory and then writes them. That
leaves the writer itself unchecked, which is the part a slicer actually sees: a
dropped triangle, an off-by-one vertex index or a unit tag that says inches all
produce a file that opens cleanly and is not the part.

Every gate is re-derived here, including the ones that only a slicer would
notice: a zero-area triangle, a duplicated face, a patch wound inside out, and
a vertex where the surface pinches against itself.

So this reopens the file as a stranger would, and re-derives every property from
what is on disk rather than from what was meant.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mesh_export import manifold_report, mesh_volume, read_3mf, slicing_report
from print_ready import loose_pieces, surfaces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--density", type=float, default=8756.0,
                    help="kg/m3, for the mass column; GRCop-42 by default")
    args = ap.parse_args()

    doc = read_3mf(args.path)
    unit, parts = doc["unit"], doc["objects"]
    print(f"{args.path}")
    print(f"  declared unit: {unit}")
    if unit != "millimeter":
        print("  FAIL: the model must be dimensioned in millimetres")
        return 1
    if not parts:
        print("  FAIL: no objects in the model")
        return 1
    bad, total_tris, total_mass = [], 0, 0.0
    print()
    print(f"  {'part':11s} {'triangles':>10s} {'watertight':>11s} {'genus':>7s} "
          f"{'sealed':>7s} {'loose':>6s} {'flat':>6s} {'pinch':>6s} {'wound':>6s} "
          f"{'volume cm3':>11s} {'mass kg':>8s}")
    for name, (v, f) in parts.items():
        rep = manifold_report(v, f)
        rep.update(slicing_report(v, f))
        surf = surfaces(v, f)
        sealed = [vol for _, vol in surf if vol < 0.0]
        loose = sorted((vol for _, vol in surf if vol > 0.0), reverse=True)[1:]
        vol = mesh_volume(v, f)
        genus = (2 * len(surf) - rep["euler"]) // 2
        mass = vol * 1e-9 * args.density
        total_tris += len(f)
        total_mass += mass
        print(f"  {name:11s} {len(f):10d} {str(rep['watertight']):>11s} {genus:7d} "
              f"{len(sealed):7d} {len(loose):6d} {rep['degenerate_faces']:6d} "
              f"{rep['nonmanifold_vertices']:6d} {rep['inconsistent_edges']:6d} "
              f"{vol / 1000:11.2f} {mass:8.3f}")
        if not rep["watertight"]:
            bad.append(f"{name}: {rep['boundary_edges']} boundary edge(s), "
                       f"{rep['nonmanifold_edges']} edge(s) in more than two "
                       f"triangles"
                       + ("" if rep["boundary_edges"] else
                          "  (no hole: the surface meets itself)"))
        if sealed:
            bad.append(f"{name}: {len(sealed)} sealed void(s)")
        if loose:
            # The check that was missing. Both pieces are closed, so
            # watertightness passes; the fragment is solid, so the sealed-void
            # test passes too. 27 cm3 of copper shipped attached to nothing.
            bad.append(f"{name}: in {len(loose) + 1} pieces, "
                       f"{sum(loose) / 1000:.1f} cm3 attached to nothing")
        if rep["degenerate_faces"]:
            bad.append(f"{name}: {rep['degenerate_faces']} zero-area triangles")
        if rep["nonmanifold_vertices"]:
            bad.append(f"{name}: pinches at {rep['nonmanifold_vertices']} vertices")
        if rep["duplicate_faces"]:
            bad.append(f"{name}: {rep['duplicate_faces']} duplicated triangles")
        if rep["inconsistent_edges"]:
            bad.append(f"{name}: {rep['inconsistent_edges']} edges wound the same "
                       f"way by both their triangles")
        if not rep["outward"]:
            bad.append(f"{name}: encloses negative volume, the solid is inside out")
        if not rep["finite"]:
            bad.append(f"{name}: non-finite vertex coordinates")
        if not rep["indices_in_range"]:
            bad.append(f"{name}: triangle references vertex {f.max()} of {len(v)}")

    lo = np.min([v.min(axis=0) for v, _ in parts.values()], axis=0)
    hi = np.max([v.max(axis=0) for v, _ in parts.values()], axis=0)
    print()
    print(f"  assembled envelope {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} x "
          f"{hi[2] - lo[2]:.1f} mm")
    print(f"  {len(parts)} solid{'s' if len(parts) != 1 else ''}, "
          f"{total_tris:,} triangles, {total_mass:.3f} kg "
          f"at {args.density:.0f} kg/m3")

    if bad:
        print("\n  FAIL")
        for b in bad:
            print(f"    {b}")
        return 1
    print("\n  OK: every solid closed, in one piece and consistently wound, "
          "no cavity without a way out, nothing a slicer will refuse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
